"""Figure of the model geometry of the 2- and 3-fish controls (--geometry tank of eod_assignment_control_groups.py), in video
orientation: the video background with the tank outline and the updated electrodes with their discs (radius 0.25 and 0.5 cm at
compute_electric_field.PX_PER_CM), and the N x N grid as the model sees it (nodes outside the walls, disc nodes).

    python plot_tank_model_geometry.py [--N 200]
Output: results/eod_assignment_<n>_fish_silent/Nov_10_2026/model_geometry_N<N>.png
"""
import argparse, subprocess
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from compute_electric_field import tank_inside, electrode_occupancy, electrode_points, PX_PER_CM, cell_cm
from io_utils import load_electrodes, load_tank_outline
import eod_assignment_control_groups as g
import tank_outline_electrodes as te

ap = argparse.ArgumentParser(); ap.add_argument('--N', type=int, default=200); N = ap.parse_args().N; P = 912
for rec, tr in (('2-fish', '2-fish'), ('3-fish', '3-fish')):
    c = g.REC[rec]; bg = te.background(te.REC[tr]['video'], te.REC[tr]['n_frames'])
    V = load_tank_outline(c['outline'])
    pairs = load_electrodes(c['electrodes_updated'], P); pts = electrode_points(pairs)
    inside = tank_inside(V, N, P); px = np.arange(N) / (N - 1) * (P - 1)
    fig = plt.figure(figsize=(19, 10.5)); gsp = fig.add_gridspec(2, 4, width_ratios=[2.1, 2.1, 1, 1])
    a_vid, a_grid = fig.add_subplot(gsp[:, 0]), fig.add_subplot(gsp[:, 1])
    zooms = [fig.add_subplot(gsp[i, 2 + j]) for i in range(2) for j in range(2)]
    lo, hi = np.percentile(bg, [1, 99.5])
    a_vid.imshow(bg, cmap='gray', vmin=lo, vmax=hi)
    a_vid.plot(*np.vstack([V, V[:1]]).T, '-', color='#eb6834', lw=1.5)
    for pid, pp, pm in load_electrodes(c['electrodes_updated']):           # poles in video px (x, y)
        for (x, y), sign in ((pp, '+'), (pm, '-')):
            for rad, ls in ((0.25, ':'), (0.5, '-')):
                a_vid.add_patch(plt.Circle((x, y), rad * PX_PER_CM, fill=False, ec='#2a78d6', lw=1.0, ls=ls))
            a_vid.text(x + (10 if x < 456 else -10), y + (14 if y < 456 else -8), f'{pid}{sign}', color='#2a78d6',
                       fontsize=8, fontweight='bold', ha='left' if x < 456 else 'right')
    a_vid.set_title('video: tank outline (orange), electrode discs\n0.25 cm (dotted) and 0.5 cm (solid)', fontsize=10)
    # model grid: 0 outside the walls, 1 water, 2 disc of 0.25 cm, 3 disc of 0.5 cm (the 0.25 cm disc nodes are inside the 0.5 ones)
    lab = inside.astype(int)
    lab[electrode_occupancy(pts, N, 0.5, P) > 0.5] = 3; lab[electrode_occupancy(pts, N, 0.25, P) > 0.5] = 2
    cmap = ListedColormap(['#3a3a38', '#e9f1fb', '#2a78d6', '#86b6ef'])
    for a in [a_grid] + zooms:
        a.imshow(lab, cmap=cmap, vmin=-0.5, vmax=3.5, extent=(px[0] - (px[1] - px[0]) / 2, px[-1] + (px[1] - px[0]) / 2,
                                                               px[-1] + (px[1] - px[0]) / 2, px[0] - (px[1] - px[0]) / 2),
                 interpolation='nearest')
        a.plot(*np.vstack([V, V[:1]]).T, '-', color='#eb6834', lw=1.0)
        for x, y in pts[:, ::-1] * (P - 1):
            for rad, ls in ((0.25, ':'), (0.5, '-')):
                a.add_patch(plt.Circle((x, y), rad * PX_PER_CM, fill=False, ec='k', lw=0.6, ls=ls))
    a_grid.set_title(f'model grid, N {N} (cell {cell_cm(N, P):.3f} cm)\ndark: outside the walls; blue: disc nodes (dark: 0.25 cm, '
                     'light: 0.5 cm only)', fontsize=10)
    fig.suptitle(f'{rec}: model geometry (--geometry tank), {PX_PER_CM} px/cm, video orientation', fontsize=12)
    for a, (name, (x0, x1, y0, y1)) in zip(zooms, dict(top_left=(0, 110, 0, 110), top_right=(802, 912, 0, 110),
                                                      bottom_left=(0, 110, 802, 912), bottom_right=(802, 912, 802, 912)).items()):
        a.set_xlim(x0, x1); a.set_ylim(y1, y0); a.set_title(name.replace('_', ' ') + ' (grid)', fontsize=9); a.tick_params(labelsize=7)
    for a in (a_vid, a_grid):
        a.set_xlim(0, P - 1); a.set_ylim(P - 1, 0); a.set_xlabel('x (px, video column)'); a.set_ylabel('y (px, video row, down)')
    fig.tight_layout(); out = f"{c['out']}/model_geometry_N{N}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    n25, n50 = (electrode_occupancy(pts, N, 0.25, P) > 0.5).sum(), (electrode_occupancy(pts, N, 0.5, P) > 0.5).sum()
    print(f'{rec}: {inside.sum()} of {N * N} nodes inside the walls; disc nodes {n25} (0.25 cm) / {n50} (0.5 cm) over 20 discs; '
          f'disc nodes outside the walls: {(~inside & (lab >= 2)).sum()}; figure {out}', flush=True)
