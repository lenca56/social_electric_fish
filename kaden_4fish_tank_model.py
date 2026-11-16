"""My model on the 4-fish Kaden recordings in the tank geometry inferred from the camera, for the voltage rule.

Model (as eod_assignment_control_groups.py, --geometry tank --interior-sigmas): 2D resistive field on an N x N grid over the video
frame (compute_electric_field), non-conductive tank walls along data/electrodes/tank_outline_updated_4-fish.csv
(tank_outline_electrodes.py --rec 4-fish: wall lines fitted to the median background of video 12_56_22; the outline of 18_56_25
and 21_16_26 differs by <= 2 px), electrodes at electrode_locations_updated_4-fish.csv, model scale 14.2 px/cm; four thin
non-conductive plastic L-pieces at the corners of a tilted inner square (data/electrodes/tank_barriers_4-fish.csv, read off the
median background; their grid nodes are removed like the walls, --no-barriers leaves them out); point currents
along the mask midline of the emitting fish (multipole_sources, unit current, sink at the tail tip); every other fish a body with
the skin (as resistive as SKIN_LAMBDA_CM of water) and an interior of INTERIOR x water (default 4). Each fish in turn is the
emitter. Masks: Kaden's visible masks of the EOD's own frame (no offset); midlines: <t>/midlines_by_frame.npz.

Output: results/eod_assign_4fish_kaden/<t>/model_N<N>_tank_barriers_int<s>_by_frame.pkl (without --no-barriers), {frame: (4 fish, 10 pairs) unit-current pair
voltages V(+) - V(-) in the raw model sign (NaN where a fish has no midline)}; resumable (checkpoint every CHECKPOINT frames).

    OMP_NUM_THREADS=1 python kaden_4fish_tank_model.py --rec 05_52_55 [--frames frames.csv] [--N 150] [--interior 4] [--workers 2]
--frames: a csv with columns video, Frame: only those frames of this recording (e.g. the 1000-per-fish clean set).
--bench n: time n frames and exit.
"""
import argparse, os, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from io_utils import load_masks
from kaden_4fish_assign import kaden_pkl, EOD
from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, build_A_shell, solve_potential,
                                    unflatten_to_grid, materials, tank_inside, confine_sources, SKIN_CM, SKIN_LAMBDA_CM, PX_PER_CM)
from eod_assign import pair_voltages, load_electrode_pairs

BASE, P, SINK, CHECKPOINT = 'results/eod_assign_4fish_kaden', 912, 0.0, 2000
OUTLINE = 'data/electrodes/tank_outline_updated_4-fish.csv'
ELECTRODES = 'data/electrodes/electrode_locations_updated_4-fish.csv'
BARRIERS = 'data/electrodes/tank_barriers_4-fish.csv'      # 4 plastic L-pieces at the corners of an inner square
C = None                                                            # run context, set before the pool forks


def barrier_nodes(N, path=BARRIERS):
    '''(N, N) bool: grid nodes on the plastic L-pieces (each arm a segment in video px, x = column, y = row). Every arm is sampled
       densely and each sample marks its nearest node, so the removed nodes form an 8-connected line that no grid edge crosses.'''
    out = np.zeros((N, N), bool); s = (N - 1) / (P - 1)
    for _, r in pd.read_csv(path).iterrows():
        n = int(np.ceil(np.hypot(r.x2 - r.x1, r.y2 - r.y1) * s * 4)) + 2
        x = np.linspace(r.x1, r.x2, n) * s; y = np.linspace(r.y1, r.y2, n) * s
        out[np.clip(np.rint(y).astype(int), 0, N - 1), np.clip(np.rint(x).astype(int), 0, N - 1)] = True
    return out


def tank_mask(N, barriers=True):
    inside = tank_inside(pd.read_csv(OUTLINE)[['x', 'y']].to_numpy(), N, P)
    return inside & ~barrier_nodes(N) if barriers else inside


def context(t, N, interior, barriers=True):
    z = np.load(f'{BASE}/{t}/midlines_by_frame.npz')
    mat = materials(N, P, PX_PER_CM)
    mat = dict(mat, g_interior=interior, g_skin=1 / (SKIN_LAMBDA_CM / SKIN_CM + 1 / interior))
    return dict(masks=load_masks(kaden_pkl(t)), mids=dict(zip(z['frames'].tolist(), z['mids'])), N=N, mat=mat,
                inside=tank_mask(N, barriers), pairs=load_electrode_pairs(ELECTRODES, P))


def solve_frame(k):
    N, inside = C['N'], C['inside']; fm = C['masks'][k]; M = C['mids'][int(k)]; u = np.full((4, len(C['pairs'])), np.nan)
    for f in range(4):
        if not np.isfinite(M[f]).all(): continue
        b = confine_sources(build_multipole_b(*multipole_sources(M[f], I0=1.0, sink_frac=SINK), 1.0 / (N - 1), N), inside)
        A = build_A_shell(occupancy_field(fm, N, skip=f), N, **C['mat'], inside=inside)
        u[f] = pair_voltages(unflatten_to_grid(solve_potential(A, b), N).real, C['pairs'], N)
    return int(k), u


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True); ap.add_argument('--frames', default=None)
    ap.add_argument('--N', type=int, default=150); ap.add_argument('--interior', type=float, default=4.0)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--bench', type=int, default=0)
    ap.add_argument('--no-barriers', action='store_true')
    a = ap.parse_args(); t = a.rec; t0 = time.time()
    C = context(t, a.N, a.interior, not a.no_barriers)
    if a.frames:
        F = pd.read_csv(a.frames); frames = np.unique(F[F.video == t].Frame.to_numpy())
    else:
        frames = np.unique(pd.read_csv(EOD.format(t=t), usecols=['Frame']).Frame.to_numpy())
    frames = [int(k) for k in frames if int(k) in C['mids']]
    if a.bench:
        t1 = time.time(); [solve_frame(k) for k in frames[:a.bench]]
        print(f'{t}: N {a.N}, {(time.time() - t1) / a.bench:.2f} s per frame (4 fish), context {t1 - t0:.0f} s', flush=True); raise SystemExit
    path = f'{BASE}/{t}/model_N{a.N}_tank{"" if a.no_barriers else "_barriers"}_int{a.interior:g}_by_frame.pkl'
    done = pd.read_pickle(path) if os.path.exists(path) else {}
    todo = [k for k in frames if k not in done]
    print(f'{t}: {len(frames)} frames, {len(done)} cached, {len(todo)} to solve, N {a.N}, interior {a.interior:g}', flush=True)
    with mp.get_context('fork').Pool(a.workers) as pool:
        for j, (k, u) in enumerate(pool.imap_unordered(solve_frame, todo, chunksize=8), 1):
            done[k] = u
            if j % CHECKPOINT == 0 or j == len(todo):
                pd.to_pickle(done, path + '.tmp'); os.replace(path + '.tmp', path)
                el = time.time() - t0
                print(f'  {t}: {j}/{len(todo)} frames, {el / 60:.1f} min, ~{el / j * (len(todo) - j) / 60:.0f} min left', flush=True)
    if not todo: pd.to_pickle(done, path)
    print(f'{t} done: {len(done)} frames, {(time.time() - t0) / 60:.1f} min', flush=True)
