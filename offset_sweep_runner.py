"""Video offset sweep, 3-fish recording: for every good-frame EOD (good_frames_better_list.csv), my model of the emitting
fish only (fish 0; BEM σ, sink at the tail tip, the other fish as bodies) rebuilt from the masks and midlines of video frame
Frame + offset, for offsets -10..+10 frames, and its L2 distance d to the same recorded EOD (patterns x -1, max |v| = 1).
The predicted pattern at every offset (u_<offset>_p<pair>) and the recorded pattern (r_p<pair>) are saved with d.
Offset 0 is solved too and checked against the cached model. Partial results are saved every SAVE_EVERY EODs and a rerun
resumes from them.

    OMP_NUM_THREADS=1 python offset_sweep_runner.py [n_workers]
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

OFFSETS = list(range(-10, 11))                                      # video frames (+-200 ms at 50 fps)
OUT_CSV = 'results/eod_ground_truth_3fish/conductivities/offset_sweep_all_good.csv'
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


def u_em(k):
    """My model pattern (x -1, max |v| = 1) with fish EM emitting, masks of video frame k; also its midline (None if none)."""
    fm = c['masks'][k]; mid = ex.fish_midlines(fm)[EM]
    if mid is None: return None, None
    occ = occupancy_field(fm, N, skip=EM); ins = occ > 0.5
    A = build_A_var(occ, N, DELTA_H, np.where(ins, SIG_INT, 1.0), np.where(ins, SIG_SKIN, 1.0))
    v = -pair_voltages(unflatten_to_grid(solve_potential(A, build_multipole_b(*multipole_sources(mid, I0=1.0, sink_frac=0.0), h, N)), N).real,
                       c['pairs'], N)
    return v / np.abs(v).max(), mid


def one_eod(i):
    frame = int(c['df'].Frame[i]); r, U = ex.my_patterns(c, i)
    row = dict(eod_row=i, frame=frame, gap_cm=c['assigned'].loc[i, 'gap_cm'], d_cached=np.linalg.norm(U[EM] - r))
    row.update({f'r_p{q}': x for q, x in zip(c['pair_ids'], r)})
    for off in OFFSETS:
        k = frame + off
        u, mid = u_em(k) if 0 <= k < N_FRAMES else (None, None)
        row[f'd_{off}'] = np.nan if u is None else np.linalg.norm(u - r)
        if u is not None: row.update({f'u_{off}_p{q}': x for q, x in zip(c['pair_ids'], u)})
        if mid is not None:
            t = (mid[-1] - mid[0]) * [-1, 1]; row[f'heading_{off}'] = np.degrees(np.arctan2(t[0], t[1]))
            row[f'head_x_{off}'], row[f'head_y_{off}'] = mid[-1][1] * (P - 1), mid[-1][0] * (P - 1)
    return row


if __name__ == '__main__':
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    todo = list(c['assigned'].index[c['assigned']['good frames'].to_numpy()])
    done = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else pd.DataFrame()
    rows = done.to_dict('records'); seen = set(done.eod_row) if len(done) else set()
    todo = [i for i in todo if i not in seen]
    print(f'{len(seen)} EODs already done, {len(todo)} to do, offsets {OFFSETS[0]}..{OFFSETS[-1]}, {n_workers} workers', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(n_workers) as pool:
        for n, row in enumerate(pool.imap_unordered(one_eod, todo, chunksize=4), 1):
            rows.append(row)
            if n % SAVE_EVERY == 0 or n == len(todo):
                pd.DataFrame(rows).sort_values('eod_row').to_csv(OUT_CSV, index=False)
                el = time.time() - t0
                print(f'{n}/{len(todo)} done, {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left', flush=True)
    T = pd.DataFrame(rows)
    bad = np.abs(T.d_0 - T.d_cached) > 1e-6
    print(f'offset 0 vs cached model: {bad.sum()} of {len(T)} EODs differ by > 1e-6', flush=True)
