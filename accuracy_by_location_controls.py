"""% of EODs assigned to the true emitter, 2- and 3-fish silent controls, by where the emitter is in the tank, for my numerical
model (shape only) in several geometries and for Denis' analytic model with my mask midline.

Versions:
  Nov 5: N150, square            my model, walls at the frame edges, original electrodes, 15.2 px/cm (2026-11-05 run)
  N200, square                   the same at N 200 (--geometry image)
  N200 tank, no electrodes       N 200, tank outline with the 2 cm top-left cut-out, updated electrodes, 14.2 px/cm (--geometry tank)
  N200 tank, discs r 0.5 cm      the same with conducting electrode discs (r 0.5 cm, 1e7 x water)
  Denis analytic, my mask        Denis' pipeline (EODAssignment, silent_fish.run_pipeline) with my mask midline: 3D point currents at
                                 z = 4 cm with first-order mirror images in the frame-edge walls (912 x 912 px = 60 x 60 cm),
                                 the floor and the surface (z 0 and 10.5 cm); original electrodes; highest cosine similarity of the
                                 peak-to-peak patterns (denis_mask_midline_all_eods_synch_fix_newest.csv)
An EOD without a pick (a fish has no midline) counts as wrong.

Groups (location of the true emitter's mask midline, measured against the tank outline tank_outline_updated_<n>-fish.csv):
  all EODs; clean good (mask QC and every fish's length >= 0.8 x median, eod_clean_good_synch_fix_newest.csv), and among clean good:
  top-left corner (within 5 cm of the cut-out's inner corner), top-right / bottom-right / bottom-left corner (within 5 cm of that
  corner), wall (within 1.5 cm of a wall, not near a corner), open water (the rest).

--set enlarged: the 2026-11-10 runs in the enlarged tank (walls moved out by the typical fish-mask overshoot, electrodes at the new
walls; --geometry tank_enlarged, N 150) without and with discs (r 0.5 cm), against the Nov 5 N150 square model and the first tank
without discs (N 200, --geometry tank; N 150 and N 200 gave the same picks in the square, so N 200 stands in); locations are
measured against the enlarged outline (tank_outline_enlarged_<n>-fish.csv).

--set rules: my N200 tank model (no discs) with the other fish as bodies (N200_tank_none) or without them (--bodies no,
N200_tank_nobodies_none), each with the L2 rule (pick) and the cosine rule (pick_cosine, add_cosine_scores.py / the runner),
against Denis' analytic model; locations against the first tank outline.

    python accuracy_by_location_controls.py [--set first|enlarged|rules]
Output: results/eod_assignment_<n>_fish_silent/Nov_10_2026/accuracy_by_location[_enlarged|_rules].csv and .png (+ printed tables).
"""
import argparse
import numpy as np
import pandas as pd
import eod_assignment_control_groups as g
from compute_electric_field import PX_PER_CM
from io_utils import load_tank_outline, load_assignment, load_denis_assignment

WALL_CM, CORNER_CM = 1.5, 5.0
REC = {'2-fish': dict(res='results/eod_ground_truth', emitter=1, letters={'A': 1, 'B': 0}),
       '3-fish': dict(res='results/eod_ground_truth_3fish', emitter=0, letters={'A': 1, 'B': 2, 'C': 0})}
SETS = {
    'first': dict(suffix='', outline='updated', denis=True, versions={
        'Nov 5: N150, square': ('{res}/conductivities/my_model_shape_synch_fix_newest.csv', '#2a78d6'),
        'N200, square': ('{out}/N200_image_none/eod_assignment_shape.csv', '#e87ba4'),
        'N200 tank, no electrodes': ('{out}/N200_tank_none/eod_assignment_shape.csv', '#eb6834'),
        'N200 tank, discs r 0.5 cm': ('{out}/N200_tank_r0.5_s1e7/eod_assignment_shape.csv', '#eda100')}),
    'enlarged': dict(suffix='_enlarged', outline='enlarged', denis=False, versions={
        'Nov 5: N150, square': ('{res}/conductivities/my_model_shape_synch_fix_newest.csv', '#2a78d6'),
        'N200 first tank, no electrodes': ('{out}/N200_tank_none/eod_assignment_shape.csv', '#eb6834'),
        'N150 enlarged tank, no electrodes': ('{out}/N150_tank_enlarged_none/eod_assignment_shape.csv', '#1baf7a'),
        'N150 enlarged tank, discs r 0.5 cm': ('{out}/N150_tank_enlarged_r0.5_s1e7/eod_assignment_shape.csv', '#eda100')}),
    'rules': dict(suffix='_rules', outline='updated', denis=True, versions={
        'N200 tank, fish bodies, L2': ('{out}/N200_tank_none/eod_assignment_shape.csv', '#eb6834', 'pick'),
        'N200 tank, fish bodies, cosine': ('{out}/N200_tank_none/eod_assignment_shape.csv', '#eda100', 'pick_cosine'),
        'N200 tank, no fish bodies, L2': ('{out}/N200_tank_nobodies_none/eod_assignment_shape.csv', '#1baf7a', 'pick'),
        'N200 tank, no fish bodies, cosine': ('{out}/N200_tank_nobodies_none/eod_assignment_shape.csv', '#2a78d6', 'pick_cosine')}),
}
ap = argparse.ArgumentParser(); ap.add_argument('--set', choices=list(SETS), default='first'); SET = SETS[ap.parse_args().set]
MINE = {v: (spec[0], spec[2] if len(spec) > 2 else 'pick') for v, spec in SET['versions'].items()}   # (path, pick column)
COL = {v: spec[1] for v, spec in SET['versions'].items()}
if SET['denis']: COL['Denis analytic, my mask'] = '#008300'
DENIS = '{res}/conductivities/denis_mask_midline_offset'                  # his summary table and his synch_fix_newest/ run
CORNERS = {'top-left corner (cut-out)': 5, 'top-right corner': 1, 'bottom-right corner': 2, 'bottom-left corner': 3}   # outline vertex


def location(Pts, V):
    """Location group of a midline (n, 2) px against the outline V (vertices: 0 cut-out top, 1 tr, 2 br, 3 bl, 4 cut-out left,
       5 cut-out inner corner)."""
    for name, j in CORNERS.items():
        if np.linalg.norm(Pts - V[j], axis=1).min() / PX_PER_CM <= CORNER_CM: return name
    E = np.roll(V, -1, 0) - V
    t = np.clip(((Pts[:, None] - V[None]) * E[None]).sum(-1) / (E ** 2).sum(-1), 0, 1)
    dw = np.linalg.norm(Pts[:, None] - (V[None] + t[..., None] * E[None]), axis=-1).min() / PX_PER_CM
    return 'wall (not a corner)' if dw <= WALL_CM else 'open water'


for rec, r in REC.items():
    c = g.REC[rec]; em = r['emitter']
    q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv")
    picks = {}
    for name, (p, pc) in MINE.items():
        d = load_assignment(p.format(res=r['res'], out=c['out'])).table
        assert (d.Frame.to_numpy() == q.Frame.to_numpy()).all(); picks[name] = d[pc].to_numpy() == em
    if SET['denis']:
        ddir = DENIS.format(res=r['res'])
        dn = load_denis_assignment(f'{ddir}/denis_mask_midline_all_eods_synch_fix_newest.csv', f'{ddir}/synch_fix_newest', fish=r['letters'])
        assert (dn.table.Frame.to_numpy() == q.Frame.to_numpy()).all() and (dn.table.eod_row.to_numpy() == np.arange(len(q))).all()
        picks['Denis analytic, my mask'] = dn.pick == em
    V = load_tank_outline(c['outline'].replace('updated', SET['outline']))
    z = np.load(c['midlines']); M = dict(zip(z['frames'].tolist(), z['midlines_px']))
    loc = np.array([location(M[int(k)][em], V) if np.isfinite(M[int(k)][em]).all() else 'no midline' for k in q.Frame])
    clean = q.clean_good.to_numpy(bool)
    groups = {'all EODs': np.ones(len(q), bool), 'clean good': clean}
    for name in ['wall (not a corner)', *CORNERS, 'open water']: groups[f'clean good: {name}'] = clean & (loc == name)
    T = pd.DataFrame({gname: {'n EODs': int(sel.sum()), **{v: 100 * ok[sel].mean() if sel.any() else np.nan for v, ok in picks.items()}}
                      for gname, sel in groups.items()}).T
    T.index.name = f'% correct ({rec}, true emitter = fish {em})'
    T.to_csv(f"{c['out']}/accuracy_by_location{SET['suffix']}.csv")
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(f'\n{rec}\n' + T.assign(**{'n EODs': T['n EODs'].astype(int)}).round(2).to_string())


# ---------------------------------------------------------------------------------------------------------------------- figure
# Bars show % assigned WRONG (100 - % correct), so that every bar starts at zero and the differences between versions near
# 100 % correct stay visible; whiskers: 95 % Wilson interval of that proportion (wide where a group has few EODs).
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt



def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z ** 2 / n; mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return 100 * (mid - half), 100 * (mid + half)


fig, axs = plt.subplots(2, 1, figsize=(16, 10.5))
for a, rec in zip(axs, REC):
    T = pd.read_csv(f"{g.REC[rec]['out']}/accuracy_by_location{SET['suffix']}.csv", index_col=0)
    T = T[T['n EODs'] > 0]
    x = np.arange(len(T)); w = 0.8 / len(COL)
    for j, (v, col) in enumerate(COL.items()):
        n = T['n EODs'].to_numpy(); wrong = 100 - T[v].to_numpy(); k = np.round(wrong / 100 * n)
        lo, hi = wilson(k, n)
        xo = x + (j - (len(COL) - 1) / 2) * w
        a.bar(xo, wrong, w * 0.92, color=col, label=v)
        a.errorbar(xo, wrong, yerr=[np.clip(wrong - lo, 0, None), np.clip(hi - wrong, 0, None)], fmt='none', ecolor='0.35', elinewidth=0.8, capsize=1.5)
    NL = '\n'
    a.set_xticks(x, [gname.replace('clean good: ', 'clean good:' + NL) + NL + f'(n = {int(n):,})' for gname, n in T['n EODs'].items()],
                 fontsize=9)
    a.set_ylabel('% assigned wrong (100 − % correct)'); a.set_ylim(bottom=0)
    a.set_title(f"{rec} silent control: % of EODs assigned to the wrong fish, by where the true emitter is", fontsize=11)
    a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
axs[0].legend(fontsize=9, frameon=False, ncol=len(COL), loc='upper center', bbox_to_anchor=(0.5, 1.16))
fig.text(0.5, 0.005, 'whiskers: 95 % Wilson interval; wall = within 1.5 cm of a wall and not near a corner; corner = within 5 cm '
         '(top left: the cut-out\'s inner corner); clean good = mask QC + length check', ha='center', fontsize=8.5, color='0.35')
fig.tight_layout(rect=(0, 0.02, 1, 1))
for rec in REC: fig.savefig(f"{g.REC[rec]['out']}/accuracy_by_location{SET['suffix']}.png", dpi=130)
print(f"figure saved: results/eod_assignment_{{2,3}}_fish_silent/Nov_10_2026/accuracy_by_location{SET['suffix']}.png")
