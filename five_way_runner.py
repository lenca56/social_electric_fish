"""Midline comparison on every mask of a recording, run in parallel over frames.

Used by Midline_five_way_12_32_58.ipynb. Masks come from the efish4-track (Kaden) .pkl, fish index 0-3 = A-D,
the same letters as the SLEAP keypoints. Raw masks (fins kept). Three approaches:

  iso_raw   isomap+ : Isomap midline + smoothing, oriented by the mask-only head decision
  md8_raw   mask-denis 8 : 8 slice points from snout tip to tail tip -> Denis's interpolation; head = wider end
  denis     Denis's method on the 4 SLEAP keypoints (tail, middle, head, mouth)

The mask-only head decision (fish_midline_mask_denis_auto): the head is the end whose third of the body has
more mask area. It is compared with SLEAP's: the body tip nearer SLEAP's mouth (and further from its tail).
One row of metrics per mask (no curves, so 60k+ masks stay small).
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

APPROACHES = {'iso_raw': 'isomap+ (raw mask)', 'md8_raw': 'mask-denis 8 (raw mask)', 'denis': 'denis (4 SLEAP keypoints)'}
PARTS = ('tail', 'middle', 'head', 'mouth')
SMOOTH = dict(lam=10, order=3)
MATCH_MAX_PX = 10       # SLEAP middle + head keypoints must lie within this (summed) distance of the mask


def dist_to_poly(p, poly):
    return np.min(np.linalg.norm(poly[None] - p[:, None], axis=2), axis=1)


def keypoint_to_mask(mask, kp):
    '''distance (px) from each keypoint tail, middle, head, mouth to the nearest mask pixel (NaN if missing)'''
    pix = np.argwhere(mask)
    return np.array([np.nan if np.isnan(k).any() else np.linalg.norm(pix - k, axis=1).min() for k in kp])


def head_point(kp):
    '''mouth keypoint, else head keypoint, else None (there is no chin in this CSV)'''
    for p in (kp[3], kp[2]):
        if not np.isnan(p).any(): return p
    return None


def sleap_head_is_a(kp, tip_a, tip_b):
    '''SLEAP's head/tail assignment of the two body tips: True if tip a is the head end'''
    head, tail = head_point(kp), kp[0]
    if np.isnan(tail).any():
        return np.linalg.norm(head - tip_a) < np.linalg.norm(head - tip_b)
    return (np.linalg.norm(head - tip_a) + np.linalg.norm(tail - tip_b)
            < np.linalg.norm(head - tip_b) + np.linalg.norm(tail - tip_a))


def curves_for(mask, kp):
    '''{short name: (curve px tail -> head or None, ms)}, the 8 mask points (px, tail -> head), head info'''
    P = mask.shape[0]; out = {}
    t0 = time.perf_counter()
    try: md8, pts, info = mm.fish_midline_mask_denis_auto(mask, 8)
    except Exception: md8, pts, info = None, None, None
    t_md8 = 1e3 * (time.perf_counter() - t0)
    out['md8_raw'] = (None if md8 is None else md8 * (P - 1), t_md8)
    if info is not None:                      # isomap+ oriented by the same mask-only head decision
        head_tip = info['tip_a'] if info['head_is_a'] else info['tip_b']
        t0 = time.perf_counter()
        try: iso = mm.smooth_midline(mm.fish_midline_isomap(mask, head_tip) * (P - 1), **SMOOTH)
        except Exception: iso = None
        out['iso_raw'] = (iso, 1e3 * (time.perf_counter() - t0))
    else: out['iso_raw'] = (None, np.nan)
    if np.isnan(kp).any(): out['denis'] = (None, np.nan)
    else:
        t0 = time.perf_counter(); out['denis'] = (mm.fish_midline_denis(kp, mask.shape) * (P - 1), 1e3 * (time.perf_counter() - t0))
    return out, pts, info


def process_frame(task):
    '''task = (frame, {letter: rle or None}, {letter: (4,2) SLEAP keypoints tail..mouth}) -> list of metric rows'''
    frame, rles, kps = task
    rows = []
    for letter, rle in rles.items():
        row = dict(frame=frame, fish=letter)
        if rle is None: rows.append(dict(row, mask_px=0, status='no mask')); continue
        m = rle_decode(rle); row['mask_px'] = int(m.sum())
        if m.sum() < 50: rows.append(dict(row, status='mask < 50 px')); continue
        kp = kps[letter]; dk = keypoint_to_mask(m, kp)
        row.update({f'kp_{p}_to_mask': v for p, v in zip(PARTS, dk)})
        if np.isnan(dk[1:3]).any() or dk[1] + dk[2] > MATCH_MAX_PX: rows.append(dict(row, status='SLEAP off mask')); continue
        if head_point(kp) is None: rows.append(dict(row, status='no head keypoint')); continue
        curves, pts, info = curves_for(m, kp)
        row['status'] = 'ok'
        if info is not None:
            s_head_a = sleap_head_is_a(kp, info['tip_a'], info['tip_b'])
            a_head, a_tail = (info['area_a'], info['area_b']) if info['head_is_a'] else (info['area_b'], info['area_a'])
            hp = head_point(kp)
            row.update(area_head_third=a_head, area_tail_third=a_tail, area_ratio=a_head / max(a_tail, 1),
                       head_agree=bool(info['head_is_a'] == s_head_a),
                       mouth_to_nearest_tip=min(np.linalg.norm(hp - info['tip_a']), np.linalg.norm(hp - info['tip_b'])),
                       body_tip_distance=np.linalg.norm(info['tip_a'] - info['tip_b']))
        dout = distance_transform_edt(~m); F = m.astype(float)
        for k, (cur, ms) in curves.items():
            if cur is None: row[f'{k}_ok'] = False; continue
            ins = map_coordinates(F, cur.T, order=1) >= 0.5
            row.update({f'{k}_ok': True, f'{k}_inside': ins.mean(), f'{k}_inside_interior': ins[10:-10].mean(),
                        f'{k}_px_out': map_coordinates(dout, cur.T, order=1).max(),
                        f'{k}_tail_err': np.linalg.norm(cur[0] - kp[0]), f'{k}_head_err': np.linalg.norm(cur[-1] - head_point(kp)),
                        f'{k}_length': np.linalg.norm(np.diff(cur, axis=0), axis=1).sum(), f'{k}_turning': mm.turning_deg(cur), f'{k}_ms': ms})
        for a, b in combinations(APPROACHES, 2):
            ca, cb = curves[a][0], curves[b][0]
            if ca is None or cb is None: continue
            d = dist_to_poly(ca, cb)
            row[f'd_{a}__{b}_mean'], row[f'd_{a}__{b}_max'] = d.mean(), d.max()
        rows.append(row)
    return rows


def contact_episodes(occluded, letters):
    '''(n_frames, n_fish) bool -> list of (letter, first, last) runs of consecutive contact frames'''
    out = []
    for i, L in enumerate(letters):
        d = np.diff(np.r_[0, occluded[:, i].astype(int), 0])
        out += [(L, int(s), int(e) - 1) for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]
    return out


def _tips(rle):
    if rle is None: return None
    m = rle_decode(rle)
    if m.sum() < 50: return None
    _, _, info = mm.fish_midline_mask_denis_auto(m, 8)
    return info['tip_a'], info['tip_b'], info['head_is_a']


def episode_heads(task):
    '''Head by continuity through one contact episode.
       task = (letter, first, last, {frame: rle or None} for first-1 .. last+1, {frame: SLEAP kp (4,2)} for CSV frames,
               {frame: Kaden skeleton (4,2) (row, col) mouth..tail})
       The frames just before and after the episode (anchors) are contact-free, so their head comes from the area rule.
       From each anchor the head is chained frame by frame: in each frame, the body tip nearest the previous frame's head.
       Both chains agree -> that tip; they disagree -> the chain whose anchor is nearer in time; one chain -> that one.
       Returns one row per episode frame that has a mask.'''
    L, s, e, rles, kps, skel = task
    tips = {f: _tips(rles.get(f)) for f in range(s - 1, e + 2)}
    anchor = {f: (None if tips[f] is None else (tips[f][0] if tips[f][2] else tips[f][1])) for f in (s - 1, e + 1)}

    def chain(frames, head):
        out = {}
        for f in frames:
            if tips[f] is None or head is None: out[f] = None; continue
            ta, tb, _ = tips[f]
            out[f] = np.linalg.norm(ta - head) <= np.linalg.norm(tb - head)        # True: tip a is the head
            head = ta if out[f] else tb
        return out
    fwd = chain(range(s, e + 1), anchor[s - 1])
    bwd = chain(range(e, s - 1, -1), anchor[e + 1])
    rows = []
    for f in range(s, e + 1):
        if tips[f] is None: continue
        ta, tb, area_a = tips[f]; F, B = fwd[f], bwd[f]
        if F is not None and B is not None and F == B: head_a, how = F, 'both chains agree'
        elif F is not None and B is not None: head_a, how = (F, 'chains differ: earlier anchor nearer') if f - (s - 1) <= (e + 1) - f else (B, 'chains differ: later anchor nearer')
        elif F is not None: head_a, how = F, 'earlier anchor only'
        elif B is not None: head_a, how = B, 'later anchor only'
        else: head_a, how = area_a, 'no anchor: area rule'
        row = dict(fish=L, frame=f, episode_first=s, episode_last=e, frames_to_anchor=min(f - (s - 1), (e + 1) - f),
                   area_head_is_a=bool(area_a), cont_head_is_a=bool(head_a), how=how)
        if f in skel and not np.isnan(skel[f][[0, 3]]).any():
            km, kt = skel[f][0], skel[f][3]
            row['kaden_head_is_a'] = bool(np.linalg.norm(km - ta) + np.linalg.norm(kt - tb) < np.linalg.norm(km - tb) + np.linalg.norm(kt - ta))
        if f in kps and head_point(kps[f]) is not None:
            row['sleap_head_is_a'] = bool(sleap_head_is_a(kps[f], ta, tb))
        rows.append(row)
    return rows


def run(tasks, processes=8, chunksize=4, progress_every=2000, worker=process_frame):
    '''worker (default process_frame) over tasks in `processes` worker processes; returns all rows'''
    from multiprocessing import get_context
    rows, t0 = [], time.time()
    with get_context('spawn').Pool(processes) as pool:
        for i, r in enumerate(pool.imap_unordered(worker, tasks, chunksize), 1):
            rows.extend(r)
            if progress_every and i % progress_every == 0:
                print(f'{i}/{len(tasks)} frames, {time.time() - t0:.0f} s', flush=True)
    return rows
