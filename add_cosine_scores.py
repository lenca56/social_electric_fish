"""Add the cosine-similarity rule to existing per-EOD result tables (eod_assignment_shape.csv layout of
eod_assignment_control_groups.py, or the Nov 5 my_model_shape_synch_fix_newest.csv): per fish cos<f> = cosine of MODEL_SIGN u and r
(eod_assign.cosine_similarity, from the saved raw u<f>_raw_p<q> and r_raw_p<q>), and pick_cosine (largest cosine, -1 without a
model). No new solves; L2 columns and pick are left as they are. Re-running overwrites the same columns.

    python add_cosine_scores.py                       # every result table of the 2- and 3-fish controls
    python add_cosine_scores.py path/to/table.csv ...
"""
import glob, sys
import numpy as np
import pandas as pd
from eod_assign import cosine_similarity

DEFAULT = (glob.glob('results/eod_assignment_[23]_fish_silent/Nov_10_2026/N*/eod_assignment_shape.csv')
           + glob.glob('results/eod_ground_truth*/conductivities/my_model_shape_synch_fix_newest.csv'))

for path in (sys.argv[1:] or DEFAULT):
    d = pd.read_csv(path)
    pairs = sorted(int(c.split('_p')[1]) for c in d.columns if c.startswith('r_raw_p'))
    fish = sorted({int(c[1:].split('_')[0]) for c in d.columns if c.startswith('u') and '_raw_p' in c})
    R = d[[f'r_raw_p{q}' for q in pairs]].to_numpy(float)
    Cs = np.full((len(d), len(fish)), np.nan)
    for j, f in enumerate(fish):
        U = d[[f'u{f}_raw_p{q}' for q in pairs]].to_numpy(float)
        ok = np.isfinite(U).all(axis=1)
        Cs[ok, j] = [cosine_similarity(u, r) for u, r in zip(U[ok], R[ok])]
        d[f'cos{f}'] = Cs[:, j]
    has = np.isfinite(Cs).any(axis=1)
    d['pick_cosine'] = np.where(has, np.array(fish)[np.argmax(np.where(np.isfinite(Cs), Cs, -np.inf), axis=1)], -1)
    d.to_csv(path, index=False)
    print(f'{path}: {len(d)} EODs, cosine and L2 picks agree on {(d.pick_cosine == d.pick).mean() * 100:.2f}%', flush=True)
