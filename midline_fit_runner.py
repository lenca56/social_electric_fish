"""Three midlines on every video frame of a recording, with per-frame metrics, run in parallel over chunks
of consecutive frames (consecutive frames are needed for the frame-to-frame jitter).

Used by Midline_fitting_12_32_58.ipynb. Masks: efish4-track (Kaden) visible masks, raw (fins kept). Approaches:

  slice    slice spline: through-body distance from a body tip -> 1-px slice centroids -> one penalised spline
  md8      mask-denis 8: 8 points from snout tip to tail tip -> Denis's interpolation
  denis    Denis's method on the 4 SLEAP keypoints (only on frames with a SLEAP row whose keypoints lie on the mask)

Head end of the mask-based midlines: area rule (wider third of the body) outside contact, continuity during contact.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per worker process
import time
from itertools import combinations
import numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt
import midline_methods as mm
from io_utils import rle_decode
from five_way_runner import PARTS, MATCH_MAX_PX, dist_to_poly, keypoint_to_mask, head_point

APPROACHES = {'slice': 'slice spline (mask)', 'md8': 'mask-denis 8 (mask)', 'denis': 'denis (4 SLEAP keypoints)'}
N = 200                                  # points per midline
FLIP_PX = 25                             # head end moving further than this between consecutive frames = a head/tail flip


def midlines(m, kp, cont_head_is_a):
    '''{name: curve px (N,2) tail -> head or None}, the 8 mask-denis points (px, tail -> head), head decision'''
    P = m.shape[0]
    pix, g, a, b = mm.mask_tips(m)
    ta, tb = pix[a].astype(float), pix[b].astype(float)
    pts = mm.mask_slice_points(pix, g, a, b, 8)                                    # tip a -> tip b
    md8 = mm.fish_midline_denis(pts, m.shape, n_ctrl=N) * (P - 1)
    area_a, area_b = mm.end_areas(m, md8)
    head_is_a = (area_a > area_b) if cont_head_is_a is None else cont_head_is_a
    if head_is_a: md8, pts = md8[::-1], pts[::-1]
    head = ta if head_is_a else tb
    out = {'md8': md8}
    try: out['slice'] = mm._orient(mm.smooth_midline(mm.slice_centroids(pix, g), lam=10, order=3, n_knots=40, n_out=N), head)
    except Exception: out['slice'] = None
    out['denis'] = None
    if kp is not None and not np.isnan(kp).any():
        dk = keypoint_to_mask(m, kp)
        if dk[1] + dk[2] <= MATCH_MAX_PX: out['denis'] = mm.fish_midline_denis(kp, m.shape, n_ctrl=N) * (P - 1)
    return out, pts, dict(head_is_a=bool(head_is_a), head_rule='continuity' if cont_head_is_a is not None else 'area')


def wiggle_px(c):
    '''RMS distance (px) between a midline and a much smoother version of itself (same order-3 penalty, lam 1000):
       small-scale jiggle along the body; a steady bend barely changes under this smoothing'''
    return float(np.sqrt(np.mean(np.sum((c - mm.smooth_midline(c, lam=1000, order=3, n_knots=40)) ** 2, axis=1))))


def process_chunk(task):
    '''task = (letter, frames (list, consecutive), {frame: rle or None}, {frame: SLEAP kp (4,2)}, {frame: continuity head_is_a},
               first frame to report, last frame to report). The chunk carries one extra frame on each side, so
       jitter can be computed for every reported frame. Returns one row per reported frame.'''
    L, frames, rles, kps, cont, f_lo, f_hi = task
    cur, rows = {}, {}
    for f in frames:
        rle = rles.get(f)
        if rle is None: cur[f] = None; continue
        m = rle_decode(rle)
        if m.sum() < 50: cur[f] = None; continue
        kp = kps.get(f)
        try: cs, pts, info = midlines(m, kp, cont.get(f))
        except Exception: cur[f] = None; continue
        cur[f] = cs
        if not (f_lo <= f <= f_hi): continue
        row = dict(frame=f, fish=L, mask_px=int(m.sum()), sleap_row=kp is not None, **info)
        dout = distance_transform_edt(~m); F = m.astype(float)
        for k, c in cs.items():
            row[f'{k}_ok'] = c is not None
            if c is None: continue
            ins = map_coordinates(F, c.T, order=1) >= 0.5
            row.update({f'{k}_inside': ins.mean(), f'{k}_inside_interior': ins[10:-10].mean(),
                        f'{k}_px_out': map_coordinates(dout, c.T, order=1).max(), f'{k}_turning': mm.turning_deg(c),
                        f'{k}_wiggle': wiggle_px(c), f'{k}_length': np.linalg.norm(np.diff(c, axis=0), axis=1).sum()})
            if kp is not None and head_point(kp) is not None:
                row[f'{k}_head_err'] = np.linalg.norm(c[-1] - head_point(kp))
                if not np.isnan(kp[0]).any(): row[f'{k}_tail_err'] = np.linalg.norm(c[0] - kp[0])
        for a, b in combinations(APPROACHES, 2):
            if cs[a] is None or cs[b] is None: continue
            d = dist_to_poly(cs[a], cs[b]); row[f'd_{a}__{b}_mean'], row[f'd_{a}__{b}_max'] = d.mean(), d.max()
        rows[f] = row
    for f, row in rows.items():                                 # frame-to-frame: needs the frames before and after
        prev, nxt = cur.get(f - 1), cur.get(f + 1)
        for k in APPROACHES:
            c = cur[f][k] if cur[f] is not None else None
            if c is None: continue
            cp = prev[k] if prev is not None else None
            if cp is not None: row[f'{k}_head_step'] = np.linalg.norm(c[-1] - cp[-1])
            cn = nxt[k] if nxt is not None else None
            if cp is not None and cn is not None:            # 2nd difference in time: 0 for steady motion, large for flicker
                row[f'{k}_jitter'] = float(np.linalg.norm(cn - 2 * c + cp, axis=1).mean())
    return list(rows.values())


def tasks_for(K, csv, cont, first, last, letters, chunk=300, mask_kind='mask_visible'):
    '''chunks of consecutive frames per fish, each with one frame of margin on both sides'''
    n = K['n_frames']; csvf = set(csv.index); out = []
    for i, L in enumerate(letters):
        for s in range(first, last + 1, chunk):
            e = min(s + chunk - 1, last); frames = list(range(max(s - 1, 0), min(e + 1, n - 1) + 1))
            kps = {f: np.array([[csv.at[f, f'{p}_y_{L}'], csv.at[f, f'{p}_x_{L}']] for p in PARTS], float) for f in frames if f in csvf}
            out.append((L, frames, {f: K[mask_kind][f][i] for f in frames}, kps, {f: cont[(f, L)] for f in frames if (f, L) in cont}, s, e))
    return out


def run(tasks, processes=8, progress_every=50):
    from multiprocessing import get_context
    rows, t0 = [], time.time()
    with get_context('spawn').Pool(processes) as pool:
        for i, r in enumerate(pool.imap_unordered(process_chunk, tasks, chunksize=1), 1):
            rows.extend(r)
            if progress_every and i % progress_every == 0: print(f'{i}/{len(tasks)} chunks, {time.time() - t0:.0f} s', flush=True)
    return rows
