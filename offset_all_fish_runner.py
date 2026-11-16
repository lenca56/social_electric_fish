"""All candidate fish at shifted video frames, 3-fish recording: for every good-frame EOD, my model with fish 1 or 2 emitting
(BEM σ, sink at the tail tip, the other fish as bodies) rebuilt from the masks and midlines of video frame Frame + offset, for
offsets -5, -4, -3 frames, and its L2 distance d to the recorded EOD (patterns x -1, max |v| = 1). Fish 0 at these offsets is
already in offset_sweep_all_good.csv; with it, the pick (argmin d over the fish) at each offset can be compared with the pick at
offset 0. Predicted patterns are saved (u<fish>_<offset>_p<pair>). Offset 0 for fish 1 and 2 is solved on the first EODs of the
run as a check against the cached model (--check). Partial results are saved every SAVE_EVERY EODs and a rerun resumes from them.

    OMP_NUM_THREADS=1 python offset_all_fish_runner.py [n_workers] [--check n]
"""
import os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import eod_examples as ex
from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, solve_potential, unflatten_to_grid,
                                    theta_outer)
from eod_assign import pair_voltages

OFFSETS = [-5, -4, -3]                                              # video frames around the best fixed offset (-4, -80 ms)
OUT_CSV = 'results/eod_ground_truth_3fish/conductivities/offset_all_fish_good.csv'
OTHERS = [1, 2]                                                     # fish 0 is in offset_sweep_all_good.csv
SAVE_EVERY = 200
N = 150; h = 1 / (N - 1); CELL_CM = (912 - 1) / (N - 1) / ex.PX_PER_CM; DELTA_H = 0.005 / CELL_CM
SIG_INT, SIG_SKIN = 100.0, 1 / (0.1 / 0.005 + 1 / 100.0)           # BEM σ (x water)

c = ex.use_better_list(ex.load_recording('3-fish')); EM = c['emitter']; P = c['P']; N_FRAMES = len(c['masks'])


def build_A_var(occ, N_, delta_h, gnode, gskin):
    """Vectorised build_A_shell with per-node values (water / interior σ in gnode, skin σ of interior nodes in gskin)."""
    tx, ty = theta_outer(occ); ins = occ > 0.5; idx = np.arange(N_ * N_).reshape(N_, N_); R, C, V = [], [], []
    for th, a, b in ((tx, np.s_[:-1, :], np.s_[1:, :]), (ty, np.s_[:, :-1], np.s_[:, 1:])):
        gA, gB = gnode[a], gnode[b]; gi_ = np.where(ins[a], gA, gB); gs_ = np.where(ins[a], gskin[a], gskin[b])
        t = np.nan_to_num(th)
        w = np.where(np.isnan(th), 2 / (1 / gA + 1 / gB), 1 / (t / gA + (1 - t) / gB + delta_h * (1 / gs_ - 1 / gi_)))
        R += [idx[a].ravel(), idx[b].ravel()]; C += [idx[b].ravel(), idx[a].ravel()]; V += [w.ravel()] * 2
    A = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(N_ * N_,) * 2).tocsr()
    return (A - sp.diags(np.asarray(A.sum(axis=1)).ravel())).astype(complex)




def u_of(k, f):
    """My model pattern (x -1, max |v| = 1) with fish f emitting, masks of video frame k (None if f has no midline)."""
    fm = c['masks'][k]; mid = ex.fish_midlines(fm)[f]
    if mid is None: return None
    occ = occupancy_field(fm, N, skip=f); ins = occ > 0.5
    A = build_A_var(occ, N, DELTA_H, np.where(ins, SIG_INT, 1.0), np.where(ins, SIG_SKIN, 1.0))
    v = -pair_voltages(unflatten_to_grid(solve_potential(A, build_multipole_b(*multipole_sources(mid, I0=1.0, sink_frac=0.0), h, N)), N).real,
                       c['pairs'], N)
    return v / np.abs(v).max()


def one_eod(i, offsets=OFFSETS):
    frame = int(c['df'].Frame[i]); r, U = ex.my_patterns(c, i)
    row = dict(eod_row=i, frame=frame)
    for f in OTHERS:
        row[f'd{f}_cached'] = np.linalg.norm(U[f] - r)
        for off in offsets:
            k = frame + off
            u = u_of(k, f) if 0 <= k < N_FRAMES else None
            row[f'd{f}_{off}'] = np.nan if u is None else np.linalg.norm(u - r)
            if u is not None: row.update({f'u{f}_{off}_p{q}': x for q, x in zip(c['pair_ids'], u)})
    return row


if __name__ == '__main__':
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    todo = list(c['assigned'].index[c['assigned']['good frames'].to_numpy()])
    if '--check' in sys.argv:                                       # offset 0 of fish 1, 2 against the cache, then stop
        n = int(sys.argv[sys.argv.index('--check') + 1]); rows = [one_eod(i, [0]) for i in todo[:n]]
        T = pd.DataFrame(rows)
        for f in OTHERS:
            ok = T[f'd{f}_0'].notna(); print(f'fish {f}: max |d at 0 - cached| = {np.abs(T[f"d{f}_0"] - T[f"d{f}_cached"])[ok].max():.1e} '
                                             f'over {ok.sum()} of {n} EODs; no midline: {(~ok).sum()}', flush=True)
        sys.exit()
    done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    rows = done.to_dict('records'); seen = set(done.eod_row) if len(done) else set()
    todo = [i for i in todo if i not in seen]
    print(f'{len(seen)} EODs already done, {len(todo)} to do, fish {OTHERS}, offsets {OFFSETS}, {n_workers} workers', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(n_workers) as pool:
        for n, row in enumerate(pool.imap_unordered(one_eod, todo, chunksize=4), 1):
            rows.append(row)
            if n % SAVE_EVERY == 0 or n == len(todo):
                pd.DataFrame(rows).sort_values('eod_row').to_csv(OUT_CSV, index=False)
                el = time.time() - t0
                print(f'{n}/{len(todo)} done, {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left', flush=True)
    print('finished', flush=True)
