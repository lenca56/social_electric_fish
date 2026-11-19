"""Where the fish masks are, against the model's tank: occupancy of every fish's ZEUS mask (*_Shiv_tracking_final.pkl, fish labels
only, every STEP-th video frame) drawn over the video background, the tank outline (tank_outline_updated_<n>-fish.csv), the N 200
grid as the model sees it (nodes outside the walls dark) and the updated electrodes; mask pixels outside the outline in red.
Also: how far the masks reach past each wall.

    python plot_mask_occupancy_tank.py
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/: mask_occupancy_vs_tank.png, mask_overshoot_by_wall.csv, and the
occupancy counts mask_occupancy_counts.npz (P x P, number of sampled frames each pixel is covered by a fish).
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.path import Path as MPath
from scipy.ndimage import distance_transform_edt
import eod_assignment_control_groups as g
from io_utils import load_masks, load_tank_outline, load_electrodes
import tank_outline_electrodes as te
from compute_electric_field import tank_inside, electrode_points, PX_PER_CM

STEP, N, P = 2, 200, 912
MIN_FRAMES = 3                       # a pixel counts as occupied when a fish covers it in >= MIN_FRAMES sampled frames
WALLS = {0: 'top (right of the cut-out)', 1: 'right', 2: 'bottom', 3: 'left (below the cut-out)', 4: 'cut-out, horizontal',
         5: 'cut-out, vertical'}       # outline edge j runs from vertex j to vertex j + 1
TR = {'2-fish': '2-fish', '3-fish': '3-fish'}

for rec in ('2-fish', '3-fish'):
    c = g.REC[rec]; out = c['out']
    cache = f'{out}/mask_occupancy_counts.npz'
    if os.path.exists(cache):
        z = np.load(cache); cnt, n_frames = z['counts'], int(z['n_frames'])
    else:
        masks = load_masks(c['track'], 'shiv', fish_order=c['fish']); cnt = np.zeros((P, P), np.int32); keys = range(0, len(masks), STEP)
        for k in keys:
            for (y0, y1, x0, x1), crop in masks.crops(k).values(): cnt[y0:y1, x0:x1] += crop
        n_frames = len(keys); np.savez_compressed(cache, counts=cnt, n_frames=n_frames, step=STEP)
    V = load_tank_outline(c['outline']); poly = MPath(V)
    yy, xx = np.mgrid[0:P, 0:P]
    inside_px = poly.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(P, P)
    occ = cnt >= MIN_FRAMES; outside = occ & ~inside_px

    # how far past each wall: distance of each outside occupied pixel to the nearest outline edge, grouped by that edge
    E = np.roll(V, -1, 0) - V; oy, ox = np.nonzero(outside); Q = np.c_[ox, oy].astype(float)
    t = np.clip(((Q[:, None] - V[None]) * E[None]).sum(-1) / (E ** 2).sum(-1), 0, 1)
    dist = np.linalg.norm(Q[:, None] - (V[None] + t[..., None] * E[None]), axis=-1); j = dist.argmin(1); dmin = dist.min(1)
    rows = []
    for e, name in WALLS.items():
        s = j == e; w = cnt[oy[s], ox[s]]
        rows.append(dict(wall=name, outside_pixels=int(s.sum()), frames_covered_outside=int(w.sum()),
                         max_px=dmin[s].max() if s.any() else 0, p99_px=np.percentile(dmin[s], 99) if s.any() else 0,
                         weighted_median_px=np.median(np.repeat(dmin[s], np.minimum(w, 1000))) if s.any() else 0))
    W = pd.DataFrame(rows); W['max_cm'] = W.max_px / PX_PER_CM; W['p99_cm'] = W.p99_px / PX_PER_CM
    W.to_csv(f'{out}/mask_overshoot_by_wall.csv', index=False)
    frac_out = cnt[~inside_px].sum() / cnt.sum()
    print(f'\n{rec}: {n_frames} sampled frames (every {STEP}th); {frac_out * 100:.2f}% of all mask pixel-frames lie outside the outline')
    print(W.round(2).to_string(index=False))

    # figure: whole tank + four corners + four mid-wall zooms
    bg = te.background(te.REC[TR[rec]]['video'], te.REC[TR[rec]]['n_frames'])
    inside = tank_inside(V, N, P); gpx = np.arange(N) / (N - 1) * (P - 1); cell = gpx[1] - gpx[0]
    ext = (gpx[0] - cell / 2, gpx[-1] + cell / 2, gpx[-1] + cell / 2, gpx[0] - cell / 2)
    pts = electrode_points(load_electrodes(c['electrodes_updated'], P))[:, ::-1] * (P - 1)
    fig = plt.figure(figsize=(22, 13)); gs = fig.add_gridspec(2, 6, width_ratios=[1.6, 1.6, 1, 1, 1, 1])
    big = fig.add_subplot(gs[:, :2]); zs = [fig.add_subplot(gs[i, 2 + k]) for i in range(2) for k in range(4)]
    boxes = {'top left': (0, 130, 0, 130), 'top right': (782, 912, 0, 130), 'bottom left': (0, 130, 782, 912),
             'bottom right': (782, 912, 782, 912), 'top, middle': (391, 521, 0, 130), 'right, middle': (782, 912, 391, 521),
             'bottom, middle': (391, 521, 782, 912), 'left, middle': (0, 130, 391, 521)}
    occ_show = np.ma.masked_less(cnt.astype(float), MIN_FRAMES)
    red = np.ma.masked_where(~outside, np.ones((P, P)))
    for a, (name, (x0, x1, y0, y1)) in zip([big] + zs, [('whole tank', (0, P, 0, P))] + list(boxes.items())):
        lo, hi = np.percentile(bg[y0:y1, x0:x1], [1, 99.5]); a.imshow(bg, cmap='gray', vmin=lo, vmax=hi)
        a.imshow(np.ma.masked_where(inside, np.ones((N, N))), cmap=ListedColormap(['#3a3a38']), alpha=0.35, extent=ext,
                 interpolation='nearest')
        im = a.imshow(occ_show, cmap='viridis', norm=LogNorm(MIN_FRAMES, max(cnt.max(), MIN_FRAMES + 1)), alpha=0.6)
        a.imshow(red, cmap=ListedColormap(['#d03b2b']), alpha=0.85)
        a.plot(*np.vstack([V, V[:1]]).T, '-', color='#eda100', lw=1.4)
        for x, y in pts: a.add_patch(plt.Circle((x, y), 0.5 * PX_PER_CM, fill=False, ec='w', lw=1))
        if a is not big:
            for xg in np.arange(N + 1) * cell - cell / 2:
                a.axvline(xg, color='w', lw=0.25, alpha=0.4); a.axhline(xg, color='w', lw=0.25, alpha=0.4)
        a.set_xlim(x0, x1 - 1); a.set_ylim(y1 - 1, y0); a.set_title(name, fontsize=10); a.tick_params(labelsize=7)
    fig.colorbar(im, ax=big, shrink=0.6, label=f'sampled frames covered by a fish (of {n_frames})')
    big.set_xlabel('x (px, video column)'); big.set_ylabel('y (px, video row, down)')
    fig.suptitle(f'{rec}: fish-mask occupancy (every {STEP}th frame; pixels covered in >= {MIN_FRAMES} frames) over the model tank\n'
                 f'orange: tank outline; shaded: grid nodes outside the walls (N {N}, cells drawn in the zooms); red: mask pixels outside '
                 f'the outline ({frac_out * 100:.2f}% of all mask pixel-frames); white circles: electrodes (r 0.5 cm)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f'{out}/mask_occupancy_vs_tank.png', dpi=110); plt.close(fig)
    print('saved', f'{out}/mask_occupancy_vs_tank.png')
