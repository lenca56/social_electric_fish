"""Tank geometry of each 3-fish recording, per video (the tank and camera moved between recordings): tank outline, both electrode
layouts and the four plastic L-barriers, for the 3-fish mask-stage jobs (cluster_kaden_3fish_*.py).

1. Background: median of 41 frames of the .avi (tank_outline_electrodes.background).
2. Outline: wall lines fitted to the background and the 2 cm cut-out at the top left (tank_outline_electrodes.wall_lines / outline),
   as for the stored 14_12_25 outline (identical on that video).
3. Electrodes: Denis' clicked positions (data/electrodes/electrodes_signed_px_coord_3-fish.csv, on 14_12_25) carried to this video
   by the similarity transform (rotation, scale, shift; least squares) that maps the 4 wall corners of 14_12_25 onto this video's;
   then my model's layout from them as in tank_outline_electrodes.update (side poles pushed MARGIN_PX inside the walls, pairs 9 and
   10 to the corners). Polarities unchanged.
4. Barriers: four white plastic L-pieces at the corners of an inner rectangle. Their rough corners are read off the backgrounds once
   per group of recordings that share them (L_PRIORS; they moved between groups); per video each L is refined on the high-pass
   background: corner shift within +-SEARCH_PX and arm lengths that maximise the brightness along both arms (the white bars).
   Each L = 2 arms (segments from the corner), like data/electrodes/tank_barriers_4-fish.csv.

Output per video in data/electrodes/3-fish/<rec>/: tank_outline.csv (x, y), electrodes_signed_px_coord.csv (Denis' layout, moved),
electrode_locations_updated.csv (my model's layout), tank_barriers.csv (piece, arm, x1, y1, x2, y2); all video px (x = column,
y = row). Figures: results/eod_assignment_3_fish_silent/mask_stages/tank_check/geometry_<rec>.png and geometry_all_videos.png.

    python tank_geometry_3fish.py [--workers 4]
"""
import os, sys, subprocess, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from tank_outline_electrodes import background, wall_lines, corners, outline, push_inside, corner_point, MARGIN_PX
import cluster_kaden_3fish_common as C

# ================================================================================================================ hyperparameters
REFERENCE = '2024-10-15T14_12_25'                 # Denis clicked the electrodes on this video
DENIS_REF_ELECTRODES = 'data/electrodes/electrodes_signed_px_coord_3-fish.csv'
N_BACKGROUND = 41                                 # frames in the median background
SEARCH_PX = 12                                    # barrier refinement: corner shift searched within +-SEARCH_PX
ARM_PX = (25, 60)                                 # barrier refinement: arm length range searched
# rough L corners (x, y) per group of recordings and the directions of their two arms (+-1 in x for the horizontal arm, +-1 in y
# for the vertical arm), read off the high-pass backgrounds
L_PRIORS = {
    '2024-10-14': {'top-left': ((253, 230), (+1, +1)), 'top-right': ((685, 219), (-1, +1)),
                   'bottom-left': ((259, 588), (+1, -1)), 'bottom-right': ((694, 576), (-1, -1))},
    '2024-10-15T14_12_25': {'top-left': ((170, 270), (+1, +1)), 'top-right': ((596, 240), (-1, +1)),
                            'bottom-left': ((197, 630), (+1, -1)), 'bottom-right': ((625, 593), (-1, -1))},
    '2024-10-15': {'top-left': ((195, 273), (+1, +1)), 'top-right': ((612, 252), (-1, +1)),
                   'bottom-left': ((202, 629), (+1, -1)), 'bottom-right': ((632, 606), (-1, -1))},
}
GEOMETRY_DIR = 'data/electrodes/3-fish/{video}'
FIG_DIR = 'results/eod_assignment_3_fish_silent/mask_stages/tank_check'


def priors(video):
    return L_PRIORS[video] if video in L_PRIORS else L_PRIORS[video[:10]]


def n_frames(vid):
    return int(subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries',
                               'stream=nb_read_packets', '-of', 'csv=p=0', vid], capture_output=True, text=True).stdout.strip())


def similarity(src, dst):
    '''least-squares similarity transform (scale s, rotation R, shift t) with dst ~ s R src + t; returns f(points)'''
    ms, md = src.mean(0), dst.mean(0); a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a); D = np.diag([1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum()
    return lambda p: s * (np.asarray(p, float) - ms) @ R.T + md


def line_mean(img, p, q, n=40):
    t = np.linspace(0, 1, n); x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
    return map_coordinates(img, [y, x], order=1).mean()


def refine_L(hp, c0, dirs):
    '''corner and arm ends of one L: corner within +-SEARCH_PX of c0 and arm lengths in ARM_PX maximising the brightness along both
       arms (each arm scored separately, so a bright arm does not carry a dark one)'''
    sx, sy = dirs; best = (-np.inf, None)
    for dx in range(-SEARCH_PX, SEARCH_PX + 1):
        for dy in range(-SEARCH_PX, SEARCH_PX + 1):
            c = np.array([c0[0] + dx, c0[1] + dy], float)
            h = line_mean(hp, c, c + (sx * 35, 0)); v = line_mean(hp, c, c + (0, sy * 35))
            if min(h, v) > best[0]: best = (min(h, v), c)
    c = best[1]
    def length(d):                                   # arm length: last step where the bar is still at least half as bright
        vals = [map_coordinates(hp, [[c[1] + d[1] * L], [c[0] + d[0] * L]], order=1)[0] for L in range(0, ARM_PX[1] + 1)]
        top = np.median(vals[3:ARM_PX[0]]); ok = [L for L, v in enumerate(vals) if L >= ARM_PX[0] and v >= 0.5 * top]
        return max(ok) if ok else ARM_PX[0]
    return c, c + (sx * length((sx, 0)), 0), c + (0, sy * length((0, sy)))


def one(video):
    vid = f'data/3_fish_silent/video{video}.avi'
    bg = background(vid, n_frames(vid), n=N_BACKGROUND); fit = wall_lines(bg); cn = corners(fit); V, notch = outline(cn)
    hp = bg - gaussian_filter(bg, 15)
    Ls = {name: refine_L(hp, c0, d) for name, (c0, d) in priors(video).items()}
    return video, bg, fit, cn, V, notch, Ls


def updated_layout(denis, fit, V, notch):
    '''my model's electrodes from Denis' layout of this video (tank_outline_electrodes.update, 'updated' variant)'''
    new = denis.copy()
    for i, r in denis.iterrows():
        for k in (1, 2):
            q = np.array([r[f'x{k}'], r[f'y{k}']], float)
            if r.pair in (9, 10):
                cands = [corner_point(fit, 'tr'), corner_point(fit, 'br'), corner_point(fit, 'bl'), notch + MARGIN_PX / np.sqrt(2)]
                q = min(cands, key=lambda p: np.linalg.norm(p - q))
            else:
                q = push_inside(V, q)
            new.loc[i, [f'x{k}', f'y{k}']] = np.round(q, 1)
    return new


def draw(ax, bg, V, denis, mine, Ls, title, labels=True):
    ax.imshow(bg, cmap='gray', vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99.5))
    ax.plot(*np.vstack([V, V[:1]]).T, color='#eb6834', lw=1.4)
    for (_, a), (_, b) in zip(mine.sort_values('pair').iterrows(), denis.sort_values('pair').iterrows()):
        for k in (1, 2):
            plus = str(a[f'pol{k}']).strip() == '+'
            ax.plot([a[f'x{k}'], b[f'x{k}']], [a[f'y{k}'], b[f'y{k}']], color='#f2c94c', lw=0.8)
            ax.plot(b[f'x{k}'], b[f'y{k}'], 'x', color='#ff4fa3', ms=7, mew=1.8)
            ax.plot(a[f'x{k}'], a[f'y{k}'], 'o', ms=7, mfc='#1baf7a' if plus else 'white', mec='#1baf7a', mew=1.6)
            if labels:
                dx = 14 if a[f'x{k}'] < 456 else -40; dy = 22 if a[f'y{k}'] < 456 else -12
                ax.text(a[f'x{k}'] + dx, a[f'y{k}'] + dy, f"{int(a.pair)}{'+' if plus else '−'}", color='#ffe066', fontsize=9, weight='bold')
    for c, e1, e2 in Ls.values():
        ax.plot([e1[0], c[0], e2[0]], [e1[1], c[1], e2[1]], color='#2fd3ff', lw=2.6, solid_capstyle='round')
    ax.set(xlim=(-15, 927), ylim=(927, -15), xticks=[], yticks=[]); ax.set_title(title, fontsize=11)


LEGEND = ([plt.Line2D([], [], color='#eb6834', lw=1.5), plt.Line2D([], [], color='#2fd3ff', lw=2.6),
           plt.Line2D([], [], marker='o', ls='', mfc='#1baf7a', mec='#1baf7a'), plt.Line2D([], [], marker='o', ls='', mfc='white', mec='#1baf7a'),
           plt.Line2D([], [], marker='x', ls='', color='#ff4fa3', mew=1.8)],
          ['tank outline (walls fitted on this video)', 'plastic L-barriers', 'my model electrode, + pole', 'my model electrode, − pole',
           "Denis' electrode (moved with the tank), joined to mine"])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    with mp.get_context('fork').Pool(a.workers) as pool: R = {r[0]: r[1:] for r in pool.map(one, C.VIDEOS)}
    ref = R[REFERENCE][2]; ref_c = np.array([ref[k] for k in ('tl', 'tr', 'br', 'bl')])
    D0 = pd.read_csv(DENIS_REF_ELECTRODES); os.makedirs(FIG_DIR, exist_ok=True)
    fig_all, axs_all = plt.subplots(2, 4, figsize=(30, 15.5)); rows = []
    for ax_all, v in zip(axs_all.ravel(), C.VIDEOS):
        bg, fit, cn, V, notch, Ls = R[v]
        f = similarity(ref_c, np.array([cn[k] for k in ('tl', 'tr', 'br', 'bl')]))
        denis = D0.copy().astype({c: float for c in ('x1', 'y1', 'x2', 'y2')})
        for k in (1, 2): denis[[f'x{k}', f'y{k}']] = np.round(f(D0[[f'x{k}', f'y{k}']].to_numpy(float)), 1)
        mine = updated_layout(denis, fit, V, notch)
        out = GEOMETRY_DIR.format(video=v); os.makedirs(out, exist_ok=True)
        pd.DataFrame(np.round(V, 2), columns=['x', 'y']).to_csv(f'{out}/tank_outline.csv', index=False)
        denis.to_csv(f'{out}/electrodes_signed_px_coord.csv', index=False); mine.to_csv(f'{out}/electrode_locations_updated.csv', index=False)
        bar = [dict(piece=f'inner {name}', arm=j + 1, x1=round(c[0], 1), y1=round(c[1], 1), x2=round(e[0], 1), y2=round(e[1], 1))
               for name, (c, e1, e2) in Ls.items() for j, e in enumerate((e1, e2))]
        pd.DataFrame(bar).to_csv(f'{out}/tank_barriers.csv', index=False)
        shift = np.linalg.norm(denis[['x1', 'y1']].to_numpy(float) - D0[['x1', 'y1']].to_numpy(float), axis=1)
        rows.append(dict(video=v, electrode_shift_px_median=np.median(shift).round(1), electrode_shift_px_max=shift.max().round(1),
                         **{f'{n}_corner': f'({c[0]:.0f}, {c[1]:.0f})' for n, (c, _, _) in Ls.items()}))
        fig, ax = plt.subplots(figsize=(12, 12.6)); draw(ax, bg, V, denis, mine, Ls, v)
        fig.legend(*LEGEND, loc='lower center', ncol=3, fontsize=10, frameon=False); fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(f'{FIG_DIR}/geometry_{v}.png', dpi=80); plt.close(fig)
        draw(ax_all, bg, V, denis, mine, Ls, v, labels=False)
    fig_all.legend(*LEGEND, loc='lower center', ncol=5, fontsize=12, frameon=False)
    fig_all.suptitle('3-fish recordings: tank outline, electrodes and L-barriers fitted per video', fontsize=15)
    fig_all.tight_layout(rect=(0, 0.04, 1, 0.97)); fig_all.savefig(f'{FIG_DIR}/geometry_all_videos.png', dpi=60)
    T = pd.DataFrame(rows); T.to_csv(f'{FIG_DIR}/geometry_per_video.csv', index=False); print(T.to_string(index=False))
