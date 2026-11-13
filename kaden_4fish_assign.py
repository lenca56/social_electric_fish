"""4-fish recordings of 2024-02-20 (EOD tables in data/4 fish, re-synchronised; masks from Kaden's tracking, letters A-D): Denis'
pipeline with my mask midline, and my model (BEM conductivities, sink at the tail tip) with the shape-only rule (model x -1 and
recording each scaled to max |v| = 1 across the 10 pairs, L2). My unit-current voltages are saved for a voltage comparison with
scalings still to be chosen. No frame offset: each EOD uses the masks of its own Frame.

Stages (per recording; --rec 05_52_55):
    midlines   (electric_fish env) mask midline of every fish in every EOD frame (fish_midline_with_head, 101 points)
    model      (electric_fish env) my unit-current pair voltages of every fish in every EOD frame (the same midlines)
    tables     (electric_fish env) per-EOD predicted voltages, L2 / similarity per fish and the assignment of each pipeline

    OMP_NUM_THREADS=1 python kaden_4fish_assign.py --rec 05_52_55 --stage midlines --workers 2

Denis' stages (denis, denis_table) moved on 2026-11-10 to ../EODAssignment/analysis/social_electric_fish_runners/kaden_4fish_denis.py;
the tables stage reads his outputs (run denis first).

Output in results/eod_assign_4fish_kaden/<rec>/ (Denis' own files in results/eod_assign_4fish_kaden/denis/).
"""
import argparse, os, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

P, N_CTRL, LETTERS = 912, 101, 'ABCD'
EOD = 'data/4 fish/Data_4fish_2024-02-20T{t}.csv'
KADEN = 'data/track_Kaden/results_{t}_full/video2024-02-20T{t}/outputs/video2024-02-20T{t}.pkl'
ELECTRODES = 'data/electrodes/electrodes_signed_px_coord_4-fish.csv'
ROWS = int(os.environ.get('KADEN_TEST_ROWS', 0)) or None               # test runs: first ROWS EODs, separate folder
OUT = Path('results/eod_assign_4fish_kaden' + ('_test' if ROWS else ''))


def kaden_pkl(t):
    '''Kaden's tracking pickle of recording t: video2024-02-20T<t>.pkl, or the R2 name (2024-02-20T<t>_R2-*.pkl) of a few recordings.'''
    p = Path(KADEN.format(t=t))
    if p.exists(): return str(p)
    alt = sorted(p.parent.glob(f'2024-02-20T{t}_*.pkl'))
    if len(alt) != 1: raise FileNotFoundError(f'{t}: no unique Kaden pickle in {p.parent}')
    return str(alt[0])


def paths(t):
    d = OUT / t; d.mkdir(parents=True, exist_ok=True)
    return dict(dir=d, mids=d / 'midlines_by_frame.npz', model=d / 'model_voltages_N150_sink0_bem_by_frame.pkl')


def eod_frames(t):
    return np.unique(pd.read_csv(EOD.format(t=t), usecols=['Frame'], nrows=ROWS).Frame.to_numpy())


# ------------------------------------------------------------------------------------------------------------------ midlines
def _midlines(k):
    from midline_methods import fish_midline_with_head
    fm = MASKS[k]; out = np.full((4, N_CTRL, 2), np.nan)
    for f in range(4):
        if not fm[f].any(): continue
        try:
            mid = fish_midline_with_head(fm[f], n_ctrl=N_CTRL)['midline']
        except Exception:
            continue
        if mid is not None and np.isfinite(mid).all(): out[f] = mid
    return out


def stage_midlines(t, workers):
    global MASKS
    from io_utils import load_masks
    MASKS = load_masks(kaden_pkl(t))
    frames = eod_frames(t); t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        mids = np.stack(pool.map(_midlines, frames, chunksize=16))
    np.savez_compressed(paths(t)['mids'], frames=frames, mids=mids)            # (row, col) / (P - 1), tail -> head
    print(f'{t} midlines: {len(frames)} frames, every fish in {np.isfinite(mids).all(axis=(1, 2, 3)).sum()}, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


# ------------------------------------------------------------------------------------------------------------------- model
def _voltages(k):
    from compute_electric_field import multipole_sources, build_multipole_b, occupancy_field, unflatten_to_grid
    from eod_assign import pair_voltages
    from model_voltages_4fish import build_A_fast, solve, N, H, DELTA_H, GAM, SINK
    fm = MASKS[k]; mids = MIDS[int(k)]; u = np.full((4, len(PAIRS)), np.nan)
    for f in range(4):
        if not np.isfinite(mids[f]).all(): continue
        b = build_multipole_b(*multipole_sources(mids[f], I0=1.0, sink_frac=SINK), H, N)
        u[f] = pair_voltages(unflatten_to_grid(solve(build_A_fast(occupancy_field(fm, N, skip=f), N, DELTA_H, *GAM), b), N).real, PAIRS, N)
    return int(k), u


def stage_model(t, workers):
    global MASKS, MIDS, PAIRS
    from io_utils import load_masks
    from eod_assign import load_electrode_pairs
    MASKS = load_masks(kaden_pkl(t)); PAIRS = load_electrode_pairs(ELECTRODES, P)
    z = np.load(paths(t)['mids']); MIDS = dict(zip(z['frames'].tolist(), z['mids']))
    path = paths(t)['model']
    done = pd.read_pickle(path) if path.exists() else {}
    todo = [k for k in z['frames'].tolist() if k not in done]
    print(f'{t} model: {len(z["frames"])} frames, {len(done)} cached, {len(todo)} to solve', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        for j, (k, u) in enumerate(pool.imap_unordered(_voltages, todo, chunksize=8), 1):
            done[k] = u
            if j % 2000 == 0 or j == len(todo):
                pd.to_pickle(done, str(path) + '.tmp'); os.replace(str(path) + '.tmp', path)
                el = time.time() - t0
                print(f'  {t}: {j}/{len(todo)} frames, {el / 60:.1f} min, ~{el / j * (len(todo) - j) / 60:.0f} min left', flush=True)
    print(f'{t} model done', flush=True)


# ------------------------------------------------------------------------------------------------------------------ tables
def stage_tables(t):
    from eod_assign import load_electrode_pairs, recorded_pair_traces_all
    pair_ids = [pid for pid, _, _ in load_electrode_pairs(ELECTRODES, P)]
    df = pd.read_csv(EOD.format(t=t), nrows=ROWS)
    V = recorded_pair_traces_all(df, pair_ids, 25)                            # (n, 10, 25), each pair demeaned
    ts = np.nanargmax(np.abs(V).max(axis=1), axis=1)
    R = V[np.arange(len(V)), :, ts]                                           # recorded peak pattern (volts)
    U = pd.read_pickle(paths(t)['model'])
    M = -np.stack([U[k] for k in df.Frame])                                   # (n, 4, 10) unit-current model, sign fixed (x -1)
    unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
    l2_shape = np.linalg.norm(unit(M) - unit(R)[:, None], axis=2)
    cos = (M * R[:, None]).sum(2) / (np.linalg.norm(M, axis=2) * np.linalg.norm(R, axis=1)[:, None])
    def pick(d):
        d = np.where(np.isfinite(d), d, np.inf)
        return np.where(np.isfinite(d.min(1)), d.argmin(1), -1)
    out = pd.DataFrame({'row': np.arange(len(df)), 'Frame': df.Frame, 'Time': df.Time})
    let = np.array(list(LETTERS) + [''])
    out['fish | mine, shape'] = let[pick(l2_shape)]

    d = pd.read_pickle(OUT / 'denis' / f'Assigned_Data_4fish_2024-02-20T{t}.pkl')
    has = np.load(paths(t)['dir'] / 'denis_has_midlines.npy')
    def spread(col, shape=()):
        a = np.full((len(df),) + shape, np.nan); a[has] = np.stack(d[col].to_numpy()) if shape else d[col].to_numpy(float); return a
    sim = np.stack([spread(f'similarity_{f}') for f in LETTERS], axis=1)
    pred = np.stack([spread(f'predicted_PTP_{f}', (10,)) for f in LETTERS], axis=1)
    npred = np.stack([spread(f'normalized_predicted_PTP_{f}', (10,)) for f in LETTERS], axis=1)
    nrec = spread('normalized_recorded_PTP', (10,)); rec_ptp = spread('recorded_PTP', (10,))
    l2_denis = np.linalg.norm(npred - nrec[:, None], axis=2)
    fd = np.full(len(df), '', object); fd[has] = d.discharging_fish.to_numpy(); out['fish | Denis, mask midline'] = fd
    for j, f in enumerate(LETTERS):
        out[f'L2 shape {f} | mine'] = l2_shape[:, j]
        out[f'cosine similarity {f} | mine'] = cos[:, j]
        out[f'similarity {f} | Denis'] = sim[:, j]; out[f'L2 {f} | Denis (max |v| = 1)'] = l2_denis[:, j]
    out.to_csv(paths(t)['dir'] / 'eod_assignments.csv', index=False)
    np.savez_compressed(paths(t)['dir'] / 'predicted_voltages.npz', frame=df.Frame.to_numpy(), recorded_peak=R,
                        mine_unit_current=M, mine_shape=unit(M),
                        denis_predicted_ptp=pred, denis_normalized_predicted_ptp=npred, denis_recorded_ptp=rec_ptp,
                        denis_normalized_recorded_ptp=nrec, letters=np.array(list(LETTERS)))
    s = {c: out[c].value_counts(normalize=True).reindex(list(LETTERS) + [''], fill_value=0).round(3).to_dict() for c in out.columns if c.startswith('fish |')}
    print(f'{t} tables: {len(out)} EODs; shares {s}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True)
    ap.add_argument('--stage', required=True, choices=['midlines', 'model', 'tables']); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    {'midlines': lambda: stage_midlines(a.rec, a.workers),
     'model': lambda: stage_model(a.rec, a.workers), 'tables': lambda: stage_tables(a.rec)}[a.stage]()
