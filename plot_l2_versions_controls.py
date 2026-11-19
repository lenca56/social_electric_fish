"""L2 distance of the true emitter (shape only: MODEL_SIGN u / max|u| vs r / max|r|) on correctly assigned EODs, per model version,
2- and 3-fish silent controls. Versions: the 2026-11-05 run (N 150, walls at the frame edges, original electrodes, 15.2 px/cm), the
same geometry at N 200 (--geometry image), and the 2026-11-10 tank runs (N 200, tank outline, updated electrodes, 14.2 px/cm; no
discs, discs of 0.25 / 0.5 cm at 1e7 x water). Also prints % correct per version on all EODs, mask-QC EODs and clean good frames.
Compared on the same EODs: those every version assigns to the true emitter, among clean good frames
(<res>/eod_clean_good_synch_fix_newest.csv: mask QC and length >= 0.8 x median; no SLEAP rule from 2026-11-10) and among
all EODs. Versions whose result file does not exist yet are skipped.

    python plot_l2_versions_controls.py
Output: results/eod_assignment_<n>_fish_silent/Nov_10_2026/l2_correct_versions.png, l2_correct_versions.csv (summary) and
accuracy_versions.csv.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from io_utils import load_assignment

V = {'Nov 5: N150, square, original electrodes': '{res}/conductivities/my_model_shape_synch_fix_newest.csv',
     'N200, square, original electrodes': '{out}/N200_image_none/eod_assignment_shape.csv',
     'N200 tank, no electrodes': '{out}/N200_tank_none/eod_assignment_shape.csv',
     'N200 tank, discs r 0.25 cm': '{out}/N200_tank_r0.25_s1e7/eod_assignment_shape.csv',
     'N200 tank, discs r 0.5 cm': '{out}/N200_tank_r0.5_s1e7/eod_assignment_shape.csv'}
COLORS = ['#2a78d6', '#e87ba4', '#eb6834', '#1baf7a', '#eda100']       # categorical slots 1, 5, 2, 3, 4: fixed per version
REC = (('2-fish', 'results/eod_ground_truth', 'results/eod_assignment_2_fish_silent/Nov_10_2026', 1),
       ('3-fish', 'results/eod_ground_truth_3fish', 'results/eod_assignment_3_fish_silent/Nov_10_2026', 0))

for rec, res, out, em in REC:
    q = pd.read_csv(f'{res}/eod_clean_good_synch_fix_newest.csv')
    avail = {name: p.format(res=res, out=out) for name, p in V.items() if os.path.exists(p.format(res=res, out=out))}
    d = {name: load_assignment(p) for name, p in avail.items()}
    cols = dict(zip(V, COLORS))                                       # colour fixed per version, also when one is missing
    for x in d.values(): assert (x.table.Frame.to_numpy() == q.Frame.to_numpy()).all()
    acc = pd.DataFrame({name: {'all': (x.pick == em).mean() * 100, 'mask QC': (x.pick == em)[q.mask_qc.astype(bool)].mean() * 100,
                               'clean good': (x.pick == em)[q.clean_good.astype(bool)].mean() * 100} for name, x in d.items()}).T
    acc.to_csv(f'{out}/accuracy_versions.csv'); print(f'\n{rec} % correct\n' + acc.round(2).to_string())
    correct_all = np.logical_and.reduce([x.pick == em for x in d.values()])
    groups = {'clean good frames': correct_all & q.clean_good.to_numpy(bool), 'all EODs': correct_all}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    rows = []
    for a, (gname, sel) in zip(ax, groups.items()):
        for name, x in d.items():
            col = cols[name]
            l2 = np.sort(x.score[sel, em])
            a.plot(l2, np.arange(1, len(l2) + 1) / len(l2), '-', color=col, lw=2, label=f'{name} (median {np.median(l2):.3f})')
            rows.append(dict(group=gname, version=name, n=len(l2), median=np.median(l2), mean=l2.mean(), p90=np.percentile(l2, 90)))
        a.set_xlim(0, np.percentile(d['N200 tank, no electrodes'].score[sel, em], 99.5))
        a.set_xlabel('L2 distance of the true emitter (shape only)'); a.set_title(f'{gname}: {sel.sum()} EODs correct in every version',
                                                                                   fontsize=10)
        a.grid(alpha=0.3, lw=0.6); a.legend(fontsize=8, frameon=False, loc='lower right')
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    ax[0].set_ylabel('fraction of EODs with L2 ≤ x')
    fig.suptitle(f'{rec} silent control: L2 on correctly assigned EODs, per model version (emitter = fish {em})', fontsize=12)
    fig.tight_layout(); fig.savefig(f'{out}/l2_correct_versions.png', dpi=130); plt.close(fig)
    t = pd.DataFrame(rows); t.to_csv(f'{out}/l2_correct_versions.csv', index=False)
    print(f'\n{rec}\n' + t.round(4).to_string(index=False))
