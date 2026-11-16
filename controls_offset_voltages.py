"""Model voltages of the 2- and 3-fish controls with the masks of video frame Frame + offset (my model, BEM conductivities,
sink at the tail tip, the other fish as bodies; as the conductivity runs, which use offset 0).

    python controls_offset_voltages.py --rec 2-fish --sweep 300        # emitter only, 300 random better-list EODs, offsets -10..+4
    python controls_offset_voltages.py --rec 3-fish --offset -4         # every EOD, every fish, at one offset

--sweep saves <res>/offset_sweep_emitter_random<n>.csv: shape distance d (model x -1 and recording each scaled to max |v| = 1) of
the emitter per offset. --offset saves <res>/model_voltages_N150_sink0_bem_offset<k>.pkl: {Frame: (n_fish, n_pairs)} unit-current
voltages (no sign change; NaN for a fish without a mask or midline), checkpointed so that a run can resume.
"""
import argparse, os, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import eod_examples as ex
from compute_electric_field import multipole_sources, build_multipole_b, occupancy_field, unflatten_to_grid
from eod_assign import pair_voltages
from model_voltages_4fish import build_A_fast, solve, N, H, DELTA_H, GAM

SWEEP_OFFSETS = list(range(-10, 5))


def voltages(k, fish=None):
    '''(n_fish, n_pairs) unit-current voltages with the masks of frame k (only the fish in `fish`; NaN for the others).'''
    u = np.full((len(C['fish']), len(C['pairs'])), np.nan)
    if not 0 <= k < len(C['masks']): return u
    fm = C['masks'][k]; mids = ex.fish_midlines(fm)
    for f in (C['fish'] if fish is None else fish):
        if mids[f] is None: continue
        b = build_multipole_b(*multipole_sources(mids[f], I0=1.0, sink_frac=0.0), H, N)
        u[f] = pair_voltages(unflatten_to_grid(solve(build_A_fast(occupancy_field(fm, N, skip=f), N, DELTA_H, *GAM), b), N).real, C['pairs'], N)
    return u


def sweep_one(i):
    frame = int(C['df'].Frame[i]); r = np.asarray(C['cache'].v_rec[i]); r = r / np.abs(r).max()
    row = dict(eod_row=i, frame=frame)
    for off in SWEEP_OFFSETS:
        u = -voltages(frame + off, [C['emitter']])[C['emitter']]
        row[f'd_{off}'] = np.linalg.norm(u / np.abs(u).max() - r) if np.isfinite(u).all() else np.nan
    return row


def frame_one(k):
    return k, voltages(k + OFFSET)


def main():
    global C, OFFSET
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True, choices=list(ex.REC))
    ap.add_argument('--sweep', type=int); ap.add_argument('--offset', type=int); ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    C = ex.load_recording(a.rec); res = C['res']
    if a.sweep:
        better = C['df'].Frame.isin(pd.read_csv(f"{res.rsplit('/', 1)[0]}/good_frames_better_list.csv").frame).to_numpy()
        rows = np.random.default_rng(0).choice(np.flatnonzero(better), a.sweep, replace=False)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('fork')) as pool:
            out = pd.DataFrame(list(pool.map(sweep_one, rows, chunksize=4)))
        out.to_csv(f'{res}/offset_sweep_emitter_random{a.sweep}.csv', index=False)
        print(out[[f'd_{o}' for o in SWEEP_OFFSETS]].median().round(3).to_string(), flush=True)
        return
    OFFSET = a.offset
    path = f'{res}/model_voltages_N{N}_sink0_bem_offset{OFFSET}.pkl'
    done = pd.read_pickle(path) if os.path.exists(path) else {}
    todo = [int(k) for k in np.unique(C['df'].Frame) if int(k) not in done]
    print(f'{a.rec}, offset {OFFSET}: {len(todo)} frames to solve, {len(done)} cached', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('fork')) as pool:
        for j, (k, u) in enumerate(pool.map(frame_one, todo, chunksize=8), 1):
            done[k] = u
            if j % 3000 == 0 or j == len(todo):
                pd.to_pickle(done, path + '.tmp'); os.replace(path + '.tmp', path)
                print(f'  {j}/{len(todo)} frames, {(time.time() - t0) / 60:.1f} min', flush=True)
    pd.to_pickle(done, path + '.tmp'); os.replace(path + '.tmp', path)
    print('done', path, flush=True)


if __name__ == '__main__':
    main()
