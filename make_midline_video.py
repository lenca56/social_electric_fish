"""Videos of the efish4-track (Kaden) masks with three midlines per fish:

  variant B     (blue)    distance from a body tip -> 1-px slice centres -> smoothing spline (fish_midline_slices)
  mask-denis 8  (orange)  8 points along the raw mask, snout tip to tail tip -> Denis's interpolation (points drawn)
  denis         (aqua)    Denis's method on the 4 SLEAP keypoints (dots); only on frames that have a SLEAP row

Masks are drawn exactly as tracked (Kaden's visible mask, plain grey). Mask-based head ends: area rule outside
contact, continuity during contact (cache from Midline_five_way_12_32_58.ipynb §4c). A filled circle marks each
midline's head end.

  --mode fish (default): one zoomed video per fish. The view is a CROP x CROP tank-pixel window shown at ZOOM x; it
      follows the fish's mask centroid averaged over SMOOTH_FRAMES frames (a smoothed camera, so the view glides
      instead of shaking). Only the camera is smoothed; masks and midlines are drawn at their true positions.
      Other fish in the window are drawn darker, without midlines.
  --mode tank: all four fish on the whole tank at 2x.

    python make_midline_video.py --start 0 --seconds 180                 # fish A-D, one video each
    python make_midline_video.py --fish B --start 9000 --seconds 30
    python make_midline_video.py --mode tank --start 0 --seconds 60
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, pickle, subprocess, time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import midline_methods as mm
from io_utils import rle_decode
import five_way_runner as fw

RUN_DIR = 'data/track_Kaden/results_12_32_58_full/video2024-02-20T12_32_58/outputs'
PKL = f'{RUN_DIR}/video2024-02-20T12_32_58.pkl'
CSV = 'data/Denis Assignment/Data_4fish_2024-02-20T12_32_58_no_ref_per.csv'
HEAD_CACHE = 'results/head_continuity_12_32_58_mask_visible.csv.gz'
P, FPS, HEADER = 912, 50, 84                    # tank px, frames / s, px of legend bar above the picture
TANK_SCALE = 2                                  # --mode tank
CROP, ZOOM, SMOOTH_FRAMES = 320, 4, 25          # --mode fish: window (tank px), magnification, camera averaging (frames)
COL = {'B': (42, 120, 214), 'md8': (235, 104, 52), 'denis': (27, 175, 122)}
MASK_RGB, OTHER_RGB, BG = (150, 149, 144), (72, 72, 70), (16, 16, 16)
_font = {}


def font(size):
    if size not in _font: _font[size] = ImageFont.load_default(size=size)
    return _font[size]


def midlines(m, kp, cont_head_is_a):
    '''(variant B px, mask-denis 8 px, its 8 points px, denis px or None), all tail -> head'''
    md8, pts, info = mm.fish_midline_mask_denis_auto(m, 8)
    md8 = md8 * (P - 1)
    head_is_a = info['head_is_a'] if cont_head_is_a is None else cont_head_is_a
    if head_is_a != info['head_is_a']: md8, pts = md8[::-1], pts[::-1]          # continuity overrides the area rule
    head_tip = info['tip_a'] if head_is_a else info['tip_b']
    try: slc = mm.fish_midline_slices(m, head_tip) * (P - 1)
    except Exception: slc = None
    den = None if kp is None or np.isnan(kp).any() else mm.fish_midline_denis(kp, m.shape) * (P - 1)
    return slc, md8, pts, den


def draw_fish(d, to_xy, scale, m, kp, cont, label=None, label_size=30):
    '''the three midlines, the 8 mask points and the SLEAP keypoints of one fish'''
    dot = lambda p, r, fill, outline=(0, 0, 0): d.ellipse([to_xy(p)[0] - r, to_xy(p)[1] - r, to_xy(p)[0] + r, to_xy(p)[1] + r], fill=fill, outline=outline)
    try: slc, md8, pts, den = midlines(m, kp, cont)
    except Exception: return
    for c, k, w in ((den, 'denis', 3), (slc, 'B', 2.5), (md8, 'md8', 1.5)):           # wider lines underneath, so all three show
        if c is None: continue
        d.line([to_xy(p) for p in c], fill=COL[k], width=max(1, round(w * scale / 2)), joint='curve'); dot(c[-1], 3 * scale, COL[k])
    for p in pts: dot(p, 1.8 * scale, COL['md8'], outline=(255, 255, 255))
    if kp is not None:
        for p in kp:
            if not np.isnan(p).any(): dot(p, 1.8 * scale, COL['denis'], outline=(255, 255, 255))
    if label:
        cy, cx = np.argwhere(m).mean(0); x, y = to_xy((cy - 12, cx + 12))
        d.text((x, y), label, fill=(255, 255, 255), font=font(label_size), stroke_width=3, stroke_fill=(0, 0, 0))


def header(d, frame, sleap, title=''):
    d.text((16, 10), f'{title}frame {frame}   t = {frame / FPS:6.2f} s', fill=(235, 235, 230), font=font(26))
    d.text((16, 46), 'SLEAP row: ' + ('yes' if sleap else 'no (no denis midline in this frame)'), fill=COL['denis'] if sleap else (160, 160, 155), font=font(20))
    x = max(d.textlength(f'{title}frame {frame}   t = {frame / FPS:6.2f} s', font=font(26)) + 60, 470)
    for k, lab in (('B', 'variant B'), ('md8', 'mask-denis 8 (+ 8 points)'), ('denis', 'denis (4 SLEAP points)')):
        d.line([(x, 26), (x + 34, 26)], fill=COL[k], width=5); d.text((x + 42, 14), lab, fill=(235, 235, 230), font=font(20))
        x += 42 + d.textlength(lab, font=font(20)) + 30
    d.text((max(d.textlength('SLEAP row: no (no denis midline in this frame)', font=font(20)) + 60, 470), 52),
           'filled circle = head end', fill=(160, 160, 155), font=font(18))


def render_tank(task):
    frame, rles, kps, cont = task
    S = TANK_SCALE
    img = np.empty((P, P, 3), np.uint8); img[:] = BG
    masks = {L: (None if r is None else rle_decode(r)) for L, r in rles.items()}
    for m in masks.values():
        if m is not None: img[m] = MASK_RGB
    im = Image.new('RGB', (P * S, P * S + HEADER), BG); im.paste(Image.fromarray(img.repeat(S, 0).repeat(S, 1)), (0, HEADER))
    d = ImageDraw.Draw(im)
    to_xy = lambda p: ((p[1] + 0.5) * S, (p[0] + 0.5) * S + HEADER)          # pixel (row, col) centre -> canvas (x, y)
    for L, m in masks.items():
        if m is not None and m.sum() >= 50: draw_fish(d, to_xy, S, m, kps.get(L), cont.get(L), label=L)
    header(d, frame, len(kps) > 0)
    return np.asarray(im, np.uint8).tobytes()


def render_fish(task):
    frame, L, rles, kp, cont, center = task
    S = ZOOM; r0, c0 = int(round(center[0] - CROP / 2)), int(round(center[1] - CROP / 2))
    img = np.empty((CROP, CROP, 3), np.uint8); img[:] = BG
    rs, re_, cs, ce = max(r0, 0), min(r0 + CROP, P), max(c0, 0), min(c0 + CROP, P)      # window part inside the tank
    me = None
    for Lo, rle in rles.items():
        if rle is None: continue
        m = rle_decode(rle)
        if Lo == L: me = m
        if rs < re_ and cs < ce:
            sub = m[rs:re_, cs:ce]; win = img[rs - r0:re_ - r0, cs - c0:ce - c0]
            win[sub] = MASK_RGB if Lo == L else OTHER_RGB
    im = Image.new('RGB', (CROP * S, CROP * S + HEADER), BG); im.paste(Image.fromarray(img.repeat(S, 0).repeat(S, 1)), (0, HEADER))
    d = ImageDraw.Draw(im)
    to_xy = lambda p: ((p[1] - c0 + 0.5) * S, (p[0] - r0 + 0.5) * S + HEADER)
    if me is not None and me.sum() >= 50: draw_fish(d, to_xy, S, me, kp, cont)
    elif me is None: d.text((20, HEADER + 20), 'no mask for this fish in this frame', fill=(200, 200, 195), font=font(22))
    header(d, frame, kp is not None, title=f'fish {L}   ')
    return np.asarray(im, np.uint8).tobytes()


def smoothed_centres(K, i, first, last):
    '''mask centroid (row, col) of fish i for frames first..last, gaps interpolated, averaged over SMOOTH_FRAMES frames'''
    h = SMOOTH_FRAMES // 2; lo, hi = max(first - h, 0), min(last + h, K['n_frames'] - 1)
    c = pd.DataFrame(K['centroid_visible'][lo:hi + 1, i][:, ::-1].astype(float), index=range(lo, hi + 1))     # (x, y) -> (row, col)
    c = c.interpolate(limit_direction='both').rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean()
    return {f: c.loc[f].to_numpy() for f in range(first, last + 1)}


def encode(tasks, worker, size, out, processes):
    from multiprocessing import get_context
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}',
                           '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'medium', out],
                          stdin=subprocess.PIPE)
    t0 = time.time()
    with get_context('spawn').Pool(processes) as pool:
        for i, buf in enumerate(pool.imap(worker, tasks, chunksize=8), 1):
            ff.stdin.write(buf)
            if i % 1500 == 0: print(f'  {i}/{len(tasks)} frames, {time.time() - t0:.0f} s', flush=True)
    ff.stdin.close(); ff.wait()
    print(f'wrote {out}: {len(tasks)} frames ({len(tasks) / FPS:.0f} s of video) in {time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('fish', 'tank'), default='fish'); ap.add_argument('--fish', default='ABCD')
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--seconds', type=float, default=180)
    ap.add_argument('--processes', type=int, default=8); ap.add_argument('--outdir', default='figures')
    a = ap.parse_args()
    K = pickle.load(open(PKL, 'rb')); LET = K['letters']
    first = a.start; last = min(a.start + int(round(a.seconds * FPS)) - 1, K['n_frames'] - 1)
    csv = pd.read_csv(CSV, low_memory=False).groupby('Frame').first()
    H = pd.read_csv(HEAD_CACHE)
    cont = {(int(f), L): bool(v) for f, L, v in zip(H.frame, H.fish, H.cont_head_is_a)}
    kp = lambda f, L: np.array([[csv.at[f, f'{p}_y_{L}'], csv.at[f, f'{p}_x_{L}']] for p in fw.PARTS], float) if f in csv.index else None
    if a.mode == 'tank':
        tasks = [(f, dict(zip(LET, K['mask_visible'][f])), {L: kp(f, L) for L in LET} if f in csv.index else {},
                  {L: cont.get((f, L)) for L in LET}) for f in range(first, last + 1)]
        encode(tasks, render_tank, (P * TANK_SCALE, P * TANK_SCALE + HEADER), f'{a.outdir}/midline_video_B_md8_denis_12_32_58_tank_frames{first}-{last}.mp4', a.processes)
        return
    for L in a.fish:
        i = LET.index(L); centre = smoothed_centres(K, i, first, last)
        tasks = [(f, L, dict(zip(LET, K['mask_visible'][f])), kp(f, L), cont.get((f, L)), centre[f]) for f in range(first, last + 1)]
        encode(tasks, render_fish, (CROP * ZOOM, CROP * ZOOM + HEADER), f'{a.outdir}/midline_video_B_md8_denis_12_32_58_fish{L}_frames{first}-{last}.mp4', a.processes)


if __name__ == '__main__':
    main()
