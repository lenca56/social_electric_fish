"""Tank outline and updated electrode positions of the 2- and 3-fish silent controls, in video pixels (x = column, y = row down,
as the .avi shows it in a normal viewer).

1. Background: median of 41 frames of the .avi (fish removed).
2. Walls: on the middle part of each side (rows / columns 250-662, away from corners and electrode rods), the strongest dark|bright
   brightness step marks the water edge; a straight line is fitted to each side (robust: points > 6 px from the median dropped).
3. Outline: the quadrilateral of the four wall lines, with a square of CUTOUT_PX = 30 px (2.0 cm) cut out of the top-left corner,
   aligned with the image axes (the walls follow the small square). The plastic walls are non-conductive.
4. Electrodes (original positions: data/electrodes/electrodes_signed_px_coord_<n>-fish.csv):
   - side electrodes (pairs 1-8): each pole is moved straight in from its nearest wall, only as far as needed to sit MARGIN_PX
     inside every wall (poles already that far inside stay where they are);
   - diagonal pairs 9 and 10: each pole goes to its corner of the outline, MARGIN_PX inside both walls: pair 9 to the top-right
     and bottom-left corners, pair 10 to the bottom-right corner and to the corner of the cut-out square that points into the
     tank (MARGIN_PX from that corner, on the diagonal).
   MARGIN_PX = 8 px (0.53 cm): a disc of radius 0.5 cm around the new centre stays inside the tank.

Output: data/electrodes/electrode_locations_updated_<n>-fish.csv (same columns and polarities as the original file, x/y in px),
data/electrodes/tank_outline_updated_<n>-fish.csv (outline vertices x, y in px, in order), and the figure
results/eod_assignment_<n>_fish_silent/Nov_10_2026/electrode_locations_updated.png.

--variant enlarged (2026-11-10): the fish masks reach a few px past the fitted walls (fish swim in 3D and the camera looks down
at a slight angle; plot_mask_occupancy_tank.py), so each wall line is moved outwards by that wall's typical overshoot (the
weighted median of mask_overshoot_by_wall.csv, px); the 30 px cut-out is kept at the new top-left corner. Side electrodes are
then placed right at the new walls: each pole goes onto its nearest wall and MARGIN_PX in from it (moved outwards or inwards as
needed), and pairs 9 and 10 to the new corners as above. Files: *_enlarged_<n>-fish.csv and electrode_locations_enlarged.png.

    python tank_outline_electrodes.py [--variant updated|enlarged]
"""
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter

from compute_electric_field import PX_PER_CM as PX_MODEL     # model scale (14.2 px/cm), for the discs in the figures
P, PX_PER_CM = 912, 912 / 60
CUTOUT_PX, MARGIN_PX = 30, 8
REC = {
    '2-fish': dict(video='data/2_fish_silent/video2024-02-13T14_33_21.avi', n_frames=51447, res='results/eod_assignment_2_fish_silent'),
    '3-fish': dict(video='data/3_fish_silent/video2024-10-15T14_12_25.avi', n_frames=38863, res='results/eod_assignment_3_fish_silent'),
    # 4-fish (2026-11-11): one video of the 2024-02-20 day (the camera does not move); the .avi is not in data/
    '4-fish': dict(video='/Users/lencacuturela/Downloads/video2024-02-20T12_56_22.avi', n_frames=59985, res='results/eod_assign_4fish_kaden'),
}


def background(video, n_frames, n=41):
    imgs = []
    for k in np.linspace(500, n_frames - 500, n).astype(int):
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{k / 50:.3f}', '-i', video, '-frames:v', '1', '-f', 'rawvideo',
                              '-pix_fmt', 'gray', '-'], capture_output=True, check=True).stdout
        imgs.append(np.frombuffer(raw, np.uint8).reshape(P, P))
    return np.median(np.stack(imgs), axis=0)


def wall_lines(bg):
    """Fitted wall lines: top / bottom y = a x + b, left / right x = a y + b (px)."""
    b = gaussian_filter(bg.astype(float), 1.0); gy, gx = np.diff(b, axis=0), np.diff(b, axis=1); span = np.arange(250, 662, 4)
    edges = dict(top=[(x, 5 + np.argmax(gy[5:80, x]) + 0.5) for x in span],
                 bottom=[(x, 820 + np.argmax(-gy[820:905, x]) + 0.5) for x in span],
                 left=[(y, 5 + np.argmax(gx[y, 5:80]) + 0.5) for y in span],
                 right=[(y, 820 + np.argmax(-gx[y, 820:905]) + 0.5) for y in span])
    fit = {}
    for w, e in edges.items():
        t, v = np.array(e).T; keep = np.abs(v - np.median(v)) < 6
        fit[w] = np.polyfit(t[keep], v[keep], 1)
    return fit


def corners(fit):
    def meet(h, v):                                                     # y = a1 x + b1 and x = a2 y + b2
        (a1, b1), (a2, b2) = h, v; y = (a1 * b2 + b1) / (1 - a1 * a2); return np.array([a2 * y + b2, y])
    return dict(tl=meet(fit['top'], fit['left']), tr=meet(fit['top'], fit['right']),
                br=meet(fit['bottom'], fit['right']), bl=meet(fit['bottom'], fit['left']))


def outline(c, s=CUTOUT_PX):
    """Outline vertices (x, y), clockwise in the video, starting at the top end of the cut-out; returns (vertices, cut-out corner)."""
    tl = c['tl']; notch = tl + (s, s)
    return np.array([tl + (s, 0), c['tr'], c['br'], c['bl'], tl + (0, s), notch]), notch


def inward_offsets(V, q):
    """Signed distance of point q to each outline edge (positive inside) and the inward unit normal of each edge."""
    inside = MPath(V).contains_point(q); d, nrm = [], []
    for a, b in zip(V, np.roll(V, -1, axis=0)):
        t = b - a; n = np.array([-t[1], t[0]]) / np.linalg.norm(t)       # one of the two normals
        mid = (a + b) / 2
        if not MPath(V).contains_point(mid + 0.5 * n): n = -n            # make it point into the tank
        u = np.clip((q - a) @ t / (t @ t), 0, 1); dist = np.linalg.norm(q - (a + u * t))
        d.append(dist if inside else -dist); nrm.append(n)
    return np.array(d), np.array(nrm)


def push_inside(V, q, m=MARGIN_PX):
    """Move q straight in from the walls until it is at least m from every outline edge (a few passes for corners)."""
    q = np.asarray(q, float).copy()
    for _ in range(10):
        d, n = inward_offsets(V, q); j = np.argmin(np.abs(d))            # nearest wall (d < 0 for every edge when outside)
        if d.min() >= m - 1e-9: break
        q = q + (m - d[j]) * n[j]
    assert inward_offsets(V, q)[0].min() >= m - 1e-6
    return q


def corner_point(fit, which, m=MARGIN_PX):
    """Point m inside both walls at outline corner `which` (tr, br, bl): corner of the wall lines moved m inwards."""
    sh = dict(top=m, bottom=-m, left=m, right=-m)                         # inward shift of each wall line's intercept
    f = {w: np.array([fit[w][0], fit[w][1] + sh[w]]) for w in fit}
    return corners(f)[which]


def snap_to_wall(V, q, m=MARGIN_PX):
    """Put q onto its nearest outline edge and m in from it (then at least m from every edge)."""
    q = np.asarray(q, float); best = None
    for a, b in zip(V, np.roll(V, -1, axis=0)):
        t = b - a; u = np.clip((q - a) @ t / (t @ t), 0, 1); p = a + u * t; dist = np.linalg.norm(q - p)
        if best is None or dist < best[0]:
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            if not MPath(V).contains_point((a + b) / 2 + 0.5 * n): n = -n
            best = (dist, p, n)
    return push_inside(V, best[1] + m * best[2], m)


WALL_NAMES = {'top': 'top (right of the cut-out)', 'right': 'right', 'bottom': 'bottom', 'left': 'left (below the cut-out)'}


def enlarge(fit, rec):
    """Wall lines moved outwards by each wall's typical mask overshoot (px)."""
    W = pd.read_csv(f"{REC[rec]['res']}/Nov_10_2026/mask_overshoot_by_wall.csv").set_index('wall').weighted_median_px
    sh = {w: W[name] for w, name in WALL_NAMES.items()}
    out = {w: np.array([fit[w][0], fit[w][1] + (-sh[w] if w in ('top', 'left') else sh[w])]) for w in fit}
    return out, sh


def update(rec, variant='updated'):
    c = REC[rec]; bg = background(c['video'], c['n_frames']); fit = wall_lines(bg)
    if variant == 'enlarged':
        fit, sh = enlarge(fit, rec); print(f'{rec}: walls moved out by (px)', {w: round(v, 2) for w, v in sh.items()}, flush=True)
    cn = corners(fit)
    V, notch = outline(cn)
    old = pd.read_csv(f'data/electrodes/electrodes_signed_px_coord_{rec}.csv'); new = old.copy().astype({'x1': float, 'y1': float,
                                                                                                         'x2': float, 'y2': float})
    for i, r in old.iterrows():
        for k in (1, 2):
            q = np.array([r[f'x{k}'], r[f'y{k}']], float)
            if r.pair in (9, 10):                                         # diagonal pairs: to the nearest outline corner
                cands = dict(tr=corner_point(fit, 'tr'), br=corner_point(fit, 'br'), bl=corner_point(fit, 'bl'),
                             notch=notch + MARGIN_PX / np.sqrt(2))
                q = min(cands.values(), key=lambda p: np.linalg.norm(p - q))
            else:
                q = snap_to_wall(V, q) if variant == 'enlarged' else push_inside(V, q)
            new.loc[i, [f'x{k}', f'y{k}']] = np.round(q, 1)
    new.to_csv(f'data/electrodes/electrode_locations_{variant}_{rec}.csv', index=False)
    pd.DataFrame(np.round(V, 2), columns=['x', 'y']).to_csv(f'data/electrodes/tank_outline_{variant}_{rec}.csv', index=False)
    return bg, V, old, new


def figure(rec, bg, V, old, new, variant='updated'):
    fig = plt.figure(figsize=(18, 11)); gs = fig.add_gridspec(2, 4, width_ratios=[2, 2, 1, 1])
    big = fig.add_subplot(gs[:, :2]); zooms = [fig.add_subplot(gs[i, 2 + j]) for i in range(2) for j in range(2)]
    boxes = dict(top_left=(0, 120, 0, 120), top_right=(792, 912, 0, 120), bottom_left=(0, 120, 792, 912), bottom_right=(792, 912, 792, 912))
    pts_old = [(r[f'x{k}'], r[f'y{k}'], r.pair, r[f'pol{k}']) for _, r in old.iterrows() for k in (1, 2)]
    pts_new = [(r[f'x{k}'], r[f'y{k}'], r.pair, r[f'pol{k}']) for _, r in new.iterrows() for k in (1, 2)]
    for a, (name, box) in zip([big] + zooms, [('whole tank', (0, 912, 0, 912))] + list(boxes.items())):
        x0, x1, y0, y1 = box; sub = bg[y0:y1, x0:x1]; lo, hi = np.percentile(sub, [1, 99.5])
        a.imshow(bg, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        a.plot(*np.vstack([V, V[:1]]).T, '-', color='#eb6834', lw=1.6)
        for (xo, yo, p, s), (xn, yn, _, _) in zip(pts_old, pts_new):
            a.annotate('', (xn, yn), (xo, yo), arrowprops=dict(arrowstyle='->', color='#1baf7a', lw=1.2))
            a.plot(xo, yo, 'o', mfc='none', mec='0.55', ms=7 if a is big else 9, mew=1.2)
            a.plot(xn, yn, 'o', mfc='#2a78d6', mec='w', ms=6 if a is big else 9, mew=0.8)
            if x0 <= xn <= x1 and y0 <= yn <= y1: a.text(xn + 4, yn + (12 if yn < 456 else -6), f'{p}{s}', color='#2a78d6', fontsize=8, fontweight='bold')
        if a is not big:
            for rr in (0.25, 0.5):
                for xn, yn, _, _ in pts_new:
                    a.add_patch(plt.Circle((xn, yn), rr * PX_MODEL, fill=False, ec='#2a78d6', lw=0.8, ls=':' if rr == 0.25 else '-'))
        a.set_xlim(x0, x1); a.set_ylim(y1, y0); a.set_title(name.replace('_', ' '), fontsize=10); a.tick_params(labelsize=7)
        step = 100 if a is big else 20
        a.set_xticks(range(x0, x1 + 1, step)); a.set_yticks(range(y0, y1 + 1, step)); a.grid(alpha=0.2, color='y')
    big.set_xlabel('x (px, video column)'); big.set_ylabel('y (px, video row, down)')
    shift = np.hypot(new[['x1', 'x2']].to_numpy() - old[['x1', 'x2']].to_numpy(), new[['y1', 'y2']].to_numpy() - old[['y1', 'y2']].to_numpy())
    fig.suptitle(f'{rec}: tank outline ({variant}; orange, 30 px cut-out at the top left) and electrodes, original (grey ring) -> '
                 f'{variant} (blue, arrow), in video orientation\nzooms: discs of radius 0.25 cm (dotted) and 0.5 cm (solid) at the new '
                 f'positions ({PX_MODEL} px/cm); shift median {np.median(shift) / PX_MODEL:.2f} cm, max {shift.max() / PX_MODEL:.2f} cm',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"{REC[rec]['res']}/Nov_10_2026/electrode_locations_{variant}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    return out, shift


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', choices=['updated', 'enlarged'], default='updated')
    ap.add_argument('--rec', choices=list(REC), default=None, help='one recording only (default: the 2- and 3-fish controls)'); a = ap.parse_args()
    for rec in ([a.rec] if a.rec else ['2-fish', '3-fish']):
        bg, V, old, new = update(rec, a.variant)
        import os; os.makedirs(f"{REC[rec]['res']}/Nov_10_2026", exist_ok=True)
        out, shift = figure(rec, bg, V, old, new, a.variant)
        print(f'{rec}: outline {np.round(V, 1).tolist()}\n  shifts (cm) per pair (pole 1, pole 2):',
              {int(p): tuple(np.round(s / PX_MODEL, 2)) for p, s in zip(old.pair, shift)}, '\n  figure', out, flush=True)
