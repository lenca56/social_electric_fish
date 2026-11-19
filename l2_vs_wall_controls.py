"""L2 distance of the true emitter's model to the recording (shape only) against where the emitter is relative to the nearest tank
wall, clean good EODs of the 2- and 3-fish silent controls, for three versions of my model:
  N200, square                 walls at the frame edges, original electrodes, 15.2 px/cm (--geometry image)
  N200 tank, no electrodes     tank outline with the 2 cm top-left cut-out, updated electrodes, 14.2 px/cm (--geometry tank)
  N200 tank, discs r 0.5 cm    the same with conducting electrode discs (r 0.5 cm, 1e7 x water)
Correctly and wrongly assigned EODs (each version's own pick) go into separate figures.

Geometry, from the true emitter's mask midline (mask_midlines_by_frame_newest.npz) and the measured tank outline
(tank_outline_updated_<n>-fish.csv), the same for every version:
  distance  the closest midline point to the nearest wall (any outline edge, the cut-out's included), cm;
  heading   angle between the fish's direction (tail tip -> head tip of the midline) and the direction straight at that wall:
            0 deg = head pointing at the wall, 90 deg = parallel to it, 180 deg = head pointing away (tail to the wall).
Lines: median L2 per bin (bins with >= MIN_N EODs), band: 25th-75th percentile; in the wrong-EOD figure every EOD is also drawn.
Clean good: mask QC and every fish's length >= 0.8 x its median (eod_clean_good_synch_fix_newest.csv).

    python l2_vs_wall_controls.py
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/: l2_vs_wall_correct.png, l2_vs_wall_wrong.png (both recordings in
each), and emitter_wall_geometry.csv (per clean-good EOD: distance, heading, L2 and pick per version).
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import eod_assignment_control_groups as g
from io_utils import load_tank_outline, load_assignment
from compute_electric_field import PX_PER_CM

REC = {'2-fish': dict(res='results/eod_ground_truth', emitter=1), '3-fish': dict(res='results/eod_ground_truth_3fish', emitter=0)}
VER = {'N200, square': ('N200_image_none', '#e87ba4'), 'N200 tank, no electrodes': ('N200_tank_none', '#eb6834'),
       'N200 tank, discs r 0.5 cm': ('N200_tank_r0.5_s1e7', '#eda100')}
DIST_BINS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 22, 30])
HEAD_BINS = np.arange(0, 181, 20)
MIN_N = {'correct': 20, 'wrong': 5}


def wall_geometry(Pts, V, inward):
    """(distance to the nearest wall in cm, heading angle to it in deg) of a midline (n, 2) px, tail -> head."""
    E = np.roll(V, -1, 0) - V
    t = np.clip(((Pts[:, None] - V[None]) * E[None]).sum(-1) / (E ** 2).sum(-1), 0, 1)
    d = np.linalg.norm(Pts[:, None] - (V[None] + t[..., None] * E[None]), axis=-1)       # (points, edges)
    j = np.unravel_index(np.argmin(d), d.shape)[1]
    h = Pts[-1] - Pts[0]; h = h / np.linalg.norm(h)
    return d.min() / PX_PER_CM, np.degrees(np.arccos(np.clip(h @ -inward[j], -1, 1)))


def inward_normals(V):
    out = []
    for a, b in zip(V, np.roll(V, -1, 0)):
        t = b - a; n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        out.append(n if MPath(V).contains_point((a + b) / 2 + n) else -n)
    return np.array(out)


TABLES = {}
for rec, r in REC.items():
    c = g.REC[rec]; em = r['emitter']
    q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv"); clean = q.clean_good.to_numpy(bool)
    V = load_tank_outline(c['outline']); nin = inward_normals(V)
    z = np.load(c['midlines']); M = dict(zip(z['frames'].tolist(), z['midlines_px']))
    rows = []
    for i in np.flatnonzero(clean):
        Pts = M[int(q.Frame[i])][em]
        rows.append((i, *wall_geometry(Pts, V, nin)) if np.isfinite(Pts).all() else (i, np.nan, np.nan))
    T = pd.DataFrame(rows, columns=['eod_row', 'wall_cm', 'heading_deg']).set_index('eod_row')
    for v, (tag, _) in VER.items():
        a = load_assignment(f"{c['out']}/{tag}/eod_assignment_shape.csv")
        assert (a.table.Frame.to_numpy() == q.Frame.to_numpy()).all()
        T[f'L2 | {v}'] = a.score[T.index, em]; T[f'correct | {v}'] = (a.pick == em)[T.index]
    T = T.dropna(subset=['wall_cm']); T.to_csv(f"{c['out']}/emitter_wall_geometry.csv"); TABLES[rec] = T
    print(f"{rec}: {len(T)} clean-good EODs; wrong per version: " +
          ', '.join(f"{v} {(~T[f'correct | {v}']).sum()}" for v in VER), flush=True)


def binned(x, y, bins, min_n):
    b = np.digitize(x, bins) - 1; mid, med, lo, hi = [], [], [], []
    for j in range(len(bins) - 1):
        s = y[b == j]
        if len(s) < min_n: continue
        mid.append(np.median(x[b == j])); med.append(np.median(s)); lo.append(np.percentile(s, 25)); hi.append(np.percentile(s, 75))
    return map(np.array, (mid, med, lo, hi))


for outcome in ('correct', 'wrong'):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5), sharey='row')
    for row, (rec, T) in zip(axs, TABLES.items()):
        for a, (col, bins, xlabel) in zip(row, (('wall_cm', DIST_BINS, 'distance from the emitter to the nearest wall (cm)'),
                                                 ('heading_deg', HEAD_BINS, 'heading to the nearest wall (deg): 0 head at the wall, '
                                                                             '90 parallel, 180 tail at the wall'))):
            for j, (v, (_, colr)) in enumerate(VER.items()):
                sel = T[f'correct | {v}'].to_numpy() == (outcome == 'correct')
                x, y = T[col].to_numpy()[sel], T[f'L2 | {v}'].to_numpy()[sel]
                if outcome == 'wrong':
                    a.scatter(x * (1 + 0.02 * (j - 1)) if col == 'wall_cm' else x + 1.5 * (j - 1), y, s=10, color=colr, alpha=0.45,
                              edgecolors='none')
                mid, med, lo, hi = binned(x, y, bins, MIN_N[outcome])
                a.fill_between(mid, lo, hi, color=colr, alpha=0.15, lw=0)
                a.plot(mid, med, 'o-', color=colr, lw=2, ms=5, label=f'{v} (n = {sel.sum():,})')
            if col == 'wall_cm':
                a.set_xscale('symlog', linthresh=1); a.set_xticks([0, 0.5, 1, 2, 4, 8, 16, 30], ['0', '0.5', '1', '2', '4', '8', '16', '30'])
            else:
                a.set_xticks(range(0, 181, 30))
            a.set_xlabel(xlabel); a.grid(alpha=0.3); a.set_axisbelow(True)
            for s in ('top', 'right'): a.spines[s].set_visible(False)
            a.set_title(f'{rec}, clean good, {outcome}ly assigned EODs', fontsize=10)
        row[0].set_ylabel("true emitter's L2 (shape only)"); row[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle(f"L2 of the true emitter's model vs its position relative to the nearest wall — {outcome}ly assigned EODs "
                 f"(line: median per bin with >= {MIN_N[outcome]} EODs, band: 25-75 %" +
                 ("; dots: every EOD)" if outcome == 'wrong' else ")"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for rec in REC: fig.savefig(f"{g.REC[rec]['out']}/l2_vs_wall_{outcome}.png", dpi=125)
    plt.close(fig)
print('saved l2_vs_wall_correct.png, l2_vs_wall_wrong.png in results/eod_assignment_{2,3}_fish_silent/Nov_10_2026/')
