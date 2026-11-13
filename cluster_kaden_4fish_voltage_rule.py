"""Cluster job: my numerical model on the 4-fish recordings of 2024-02-20 with Kaden's masks, and the voltage rule for EOD
assignment. One SLURM array task = one (video, chunk) pair; see the job table below.

For every EOD of the task's frames and every fish f (each fish in turn the emitter, the other fish as bodies):
  u_f,p      raw model: unit-current pair voltage V(+) - V(-) of pair p (2D resistive field on an N x N grid, tank walls and
             the four plastic L-pieces non-conductive, the other fish with skin + an interior of INTERIOR_SIGMA x water,
             point currents along f's mask midline, sink at the tail tip)
  factor_f,p normalising factor to volts: a_f * e_k, a_f = median fish factor, e_k = median factor of signed electrode k,
             k = the electrode of pair p (+ or -) nearer to f's mask midline in this frame (FACTORS_FILE)
  pred_f,p   = factor_f,p * MODEL_SIGN * u_f,p, predicted voltage (V)
  r_p        recorded main-phase pattern: each pair's 25-sample trace demeaned, all 10 pairs at the sample of the largest |V|
  l2_volt_f  = ||r - pred_f||           (V)
  l2_rel_f   = l2_volt_f / max_p |r_p|  (unitless)
  l2_shape_f = || r / max|r| - MODEL_SIGN u_f / max|u_f| ||   (shape only, no factors)
pick_voltage = the fish with the smallest l2_volt; pick_shape = the fish with the smallest l2_shape.

Files the job needs (paths relative to PROJECT_DIR, copy them to the cluster):
  code     compute_electric_field.py, eod_assign.py, io_utils.py, midline_methods.py (this script imports them)
  EODs     data/4 fish/Data_4fish_2024-02-20T<video>.csv
  masks    data/track_Kaden/results_<video>_full/video2024-02-20T<video>/outputs/video2024-02-20T<video>.pkl (or the
           2024-02-20T<video>_R2-*.pkl name of 5 recordings)
  tank     data/electrodes/tank_outline_updated_4-fish.csv, electrode_locations_updated_4-fish.csv, tank_barriers_4-fish.csv
  factors  FACTORS_FILE
  optional results/eod_assign_4fish_kaden/<video>/midlines_by_frame.npz (mask midlines; fitted on the masks when missing)
           results/eod_assign_4fish_kaden/<video>/mask_qc_good_frames.csv (good-mask flag; all False when missing)

Output: OUT_DIR/<video>/chunk<c>_of<N_CHUNKS>.npz with eod_rows, frames, recorded (n, 10), u_raw (n, 4, 10), factor (n, 4, 10),
pred_volt (n, 4, 10), nearer_pole (n, 4, 10; 0 = +, 1 = -), l2_volt, l2_rel, l2_shape (n, 4), pick_voltage, pick_shape,
good_masks (n,), hyperparameters (json); and the same per EOD as OUT_DIR/<video>/chunk<c>_of<N_CHUNKS>.csv.

Local test without SLURM:  python cluster_kaden_4fish_voltage_rule.py <task index>   (python ... --list prints the job table)
"""
import os, sys, json, glob, time
import multiprocessing as mp
import numpy as np
import pandas as pd

from io_utils import load_masks
from midline_methods import fish_midline_with_head
from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, build_A_shell, solve_potential,
                                    unflatten_to_grid, materials, tank_inside, confine_sources, SKIN_CM)
from eod_assign import pair_voltages, load_electrode_pairs, recorded_pair_traces_all

# ================================================================================================================ hyperparameters
PROJECT_DIR = os.environ.get('PROJECT_DIR', '.')       # folder that holds data/ and results/
OUT_DIR = os.environ.get('OUT_DIR', 'results/eod_assign_4fish_kaden/cluster_voltage_rule')

# model
N_GRID = 150                  # grid nodes per side over the 912 x 912 px video frame
INTERIOR_SIGMA = 4.0          # interior conductivity of the other fish's bodies, x water
SKIN_LAMBDA_CM = 0.1          # skin resistance: as resistive as this many cm of water (skin thickness SKIN_CM = 0.005 cm)
SINK_FRAC = 0.0               # position of the current sink along the midline (0 = tail tip)
PX_PER_CM = 14.2              # model scale: the 60 x 60 cm tank spans ~852 px inside its walls
MODEL_SIGN = -1               # constant sign of the unit-current model against the recorded main phase
N_MIDLINE = 101               # midline points (tail -> head) carrying the point currents
P = 912                       # video frame size, px

# geometry (camera-derived tank, 2026-11-11)
TANK_OUTLINE = 'data/electrodes/tank_outline_updated_4-fish.csv'          # wall polygon, video px (x = column, y = row)
ELECTRODES = 'data/electrodes/electrode_locations_updated_4-fish.csv'     # electrodes moved inside the walls
BARRIERS = 'data/electrodes/tank_barriers_4-fish.csv'                     # 4 plastic L-pieces, one row per arm (segment)
USE_BARRIERS = True

# voltage rule: median factors (fish factor = median of the fish's ratios |V_rec| / |V_pred| on the 2 loudest pairs of the
# 1000 best clean isolated EODs per fish; electrode factor = median of ratio / fish factor per signed electrode; rows
# 'fish A'..'fish D', 'electrode 1+'..'electrode 10-', column 'value')
FACTORS_FILE = 'results/eod_assign_4fish_kaden/clean_emitter_set_71/gain_mine_tank_int4/voltage_rule_factors_median.csv'

# data
EOD_TABLE = 'data/4 fish/Data_4fish_2024-02-20T{video}.csv'
KADEN_PKL = 'data/track_Kaden/results_{video}_full/video2024-02-20T{video}/outputs/video2024-02-20T{video}.pkl'
MIDLINES = 'results/eod_assign_4fish_kaden/{video}/midlines_by_frame.npz'
MASK_QC = 'results/eod_assign_4fish_kaden/{video}/mask_qc_good_frames.csv'
N_SAMPLES = 25                # samples per EOD trace in the EOD table
LETTERS = 'ABCD'

# jobs
N_CHUNKS = 4                  # each video's EOD frames split into this many array tasks (~10k frames each)
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 0))  # tests only: solve just the first MAX_FRAMES frames of the chunk (0 = all)

os.chdir(PROJECT_DIR)
VIDEOS = sorted(os.path.basename(p)[len('Data_4fish_2024-02-20T'):-len('.csv')]
                for p in glob.glob(EOD_TABLE.format(video='*')) if not os.path.basename(p).startswith('._'))
df = pd.DataFrame(columns=['video', 'chunk'])          # in total len(VIDEOS) * N_CHUNKS tasks (71 x 4 = 284: --array=0-283)
z = 0
for video in VIDEOS:
    for chunk in range(N_CHUNKS):
        df.loc[z, 'video'] = video
        df.loc[z, 'chunk'] = chunk
        z += 1
if len(sys.argv) > 1 and sys.argv[1] == '--list':
    print(df.to_string()); sys.exit()

idx = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else int(sys.argv[1])
video = df.loc[idx, 'video']
chunk = int(df.loc[idx, 'chunk'])
ORDER = [f'{p}{s}' for p in range(1, 11) for s in ('+', '-')]
HYPER = dict(N_GRID=N_GRID, INTERIOR_SIGMA=INTERIOR_SIGMA, SKIN_LAMBDA_CM=SKIN_LAMBDA_CM, SINK_FRAC=SINK_FRAC, PX_PER_CM=PX_PER_CM,
             MODEL_SIGN=MODEL_SIGN, TANK_OUTLINE=TANK_OUTLINE, ELECTRODES=ELECTRODES, BARRIERS=BARRIERS if USE_BARRIERS else None,
             FACTORS_FILE=FACTORS_FILE, N_CHUNKS=N_CHUNKS, video=video, chunk=chunk)


# ===================================================================================================================== helpers
def kaden_pkl(video):
    p = KADEN_PKL.format(video=video)
    if os.path.exists(p): return p
    alt = sorted(glob.glob(os.path.join(os.path.dirname(p), f'2024-02-20T{video}_*.pkl')))
    if len(alt) != 1: raise FileNotFoundError(f'{video}: no unique Kaden pickle in {os.path.dirname(p)}')
    return alt[0]


def barrier_nodes(N):
    '''(N, N) bool: grid nodes on the plastic L-pieces; each arm is sampled densely and each sample marks its nearest node,
       so the removed nodes form an 8-connected line that no grid edge crosses.'''
    out = np.zeros((N, N), bool); s = (N - 1) / (P - 1)
    for _, r in pd.read_csv(BARRIERS).iterrows():
        n = int(np.ceil(np.hypot(r.x2 - r.x1, r.y2 - r.y1) * s * 4)) + 2
        x, y = np.linspace(r.x1, r.x2, n) * s, np.linspace(r.y1, r.y2, n) * s
        out[np.clip(np.rint(y).astype(int), 0, N - 1), np.clip(np.rint(x).astype(int), 0, N - 1)] = True
    return out


def fit_midlines(fm):
    '''(4, N_MIDLINE, 2) mask midlines (row, col) / (P - 1), tail -> head, NaN where a fish has no mask or the fit fails.'''
    out = np.full((len(fm), N_MIDLINE, 2), np.nan)
    for f, m in enumerate(fm):
        try:
            mid = fish_midline_with_head(m, n_ctrl=N_MIDLINE)['midline'] if m.any() else None
            if mid is not None and np.isfinite(mid).all(): out[f] = mid
        except Exception:
            pass
    return out


def solve_frame(k):
    '''Unit-current pair voltages (4 fish, 10 pairs) of video frame k, each fish in turn emitting, and the frame's midlines.'''
    fm = MASKS[k]; M = MIDS.get(int(k))
    if M is None: M = fit_midlines(fm)
    u = np.full((4, len(PAIRS)), np.nan)
    for f in range(4):
        if not np.isfinite(M[f]).all(): continue
        b = confine_sources(build_multipole_b(*multipole_sources(M[f], I0=1.0, sink_frac=SINK_FRAC), 1.0 / (N_GRID - 1), N_GRID), INSIDE)
        A = build_A_shell(occupancy_field(fm, N_GRID, skip=f), N_GRID, **MAT, inside=INSIDE)
        u[f] = pair_voltages(unflatten_to_grid(solve_potential(A, b), N_GRID).real, PAIRS, N_GRID)
    return int(k), u, M


def nearer_pole(M):
    '''(4, 10): 0 where pair p's + electrode is nearer to fish f's midline, 1 where its - electrode is (0 for a missing fish).'''
    out = np.zeros((len(M), 10), int)
    for f in range(len(M)):
        if not np.isfinite(M[f]).all(): continue
        xy = M[f][:, ::-1] * (P - 1)                                                          # (N_MIDLINE, x/y) px
        dp = np.linalg.norm(xy[:, None] - POLES_PLUS[None], axis=-1).min(0)
        dm = np.linalg.norm(xy[:, None] - POLES_MINUS[None], axis=-1).min(0)
        out[f] = (dm < dp).astype(int)
    return out


# ===================================================================================================================== setup
t0 = time.time()
print(f'task {idx}: video {video}, chunk {chunk + 1}/{N_CHUNKS}, {N_WORKERS} worker(s)', flush=True)
MASKS = load_masks(kaden_pkl(video))
MIDS = {}
if os.path.exists(MIDLINES.format(video=video)):
    zm = np.load(MIDLINES.format(video=video)); MIDS = dict(zip(zm['frames'].tolist(), zm['mids']))
PAIRS = load_electrode_pairs(ELECTRODES, P); PIDS = [p for p, _, _ in PAIRS]
EL = pd.read_csv(ELECTRODES).sort_values('pair')
POLES_PLUS = np.array([(r.x1, r.y1) if str(r.pol1).strip() == '+' else (r.x2, r.y2) for _, r in EL.iterrows()], float)
POLES_MINUS = np.array([(r.x2, r.y2) if str(r.pol1).strip() == '+' else (r.x1, r.y1) for _, r in EL.iterrows()], float)
INSIDE = tank_inside(pd.read_csv(TANK_OUTLINE)[['x', 'y']].to_numpy(), N_GRID, P)
if USE_BARRIERS: INSIDE = INSIDE & ~barrier_nodes(N_GRID)
MAT = materials(N_GRID, P, PX_PER_CM)
MAT = dict(MAT, g_interior=INTERIOR_SIGMA, g_skin=1 / (SKIN_LAMBDA_CM / SKIN_CM + 1 / INTERIOR_SIGMA))
FAC = pd.read_csv(FACTORS_FILE)
FAC['factor'] = FAC['factor'].str.replace('−', '-'); FAC = FAC.set_index('factor').value
A_FISH = np.array([FAC[f'fish {f}'] for f in LETTERS])
E_ELEC = np.array([FAC.get(f'electrode {e}', 1.0) for e in ORDER]).reshape(10, 2)              # (pair, + / -)

eods = pd.read_csv(EOD_TABLE.format(video=video), usecols=['Frame'] + [f'electrode_{p}_time_{i}' for p in PIDS for i in range(1, N_SAMPLES + 1)])
frames_all = np.unique(eods.Frame.to_numpy())
my_frames = np.array_split(frames_all, N_CHUNKS)[chunk]
if MAX_FRAMES: my_frames = my_frames[:MAX_FRAMES]
sel = np.flatnonzero(np.isin(eods.Frame.to_numpy(), my_frames))
frames = eods.Frame.to_numpy()[sel]
V = recorded_pair_traces_all(eods.iloc[sel], PIDS, N_SAMPLES); del eods
R = V[np.arange(len(V)), :, np.abs(V).max(1).argmax(1)]; del V                                 # (n, 10) recorded main phase, V
print(f'  {len(sel)} EODs on {len(my_frames)} frames; setup {time.time() - t0:.0f} s', flush=True)

# ===================================================================================================================== solve
U, POLE = {}, {}
work = [int(k) for k in my_frames]
if N_WORKERS > 1:
    with mp.get_context('fork').Pool(N_WORKERS) as pool:
        results = pool.imap_unordered(solve_frame, work, chunksize=8)
        for j, (k, u, M) in enumerate(results, 1):
            U[k] = u; POLE[k] = nearer_pole(M)
            if j % 1000 == 0: print(f'  {j}/{len(work)} frames, {(time.time() - t0) / 60:.1f} min', flush=True)
else:
    for j, k in enumerate(work, 1):
        _, u, M = solve_frame(k); U[k] = u; POLE[k] = nearer_pole(M)
        if j % 1000 == 0: print(f'  {j}/{len(work)} frames, {(time.time() - t0) / 60:.1f} min', flush=True)

# ===================================================================================================================== voltage rule
u_raw = np.stack([U[int(k)] for k in frames])                                                   # (n, 4, 10), raw model sign
pole = np.stack([POLE[int(k)] for k in frames])                                                 # (n, 4, 10)
factor = A_FISH[None, :, None] * E_ELEC[np.arange(10)[None, None, :], pole]                     # (n, 4, 10)
pred = factor * MODEL_SIGN * u_raw
l2_volt = np.linalg.norm(R[:, None] - pred, axis=2)
r_max = np.abs(R).max(1)
l2_rel = l2_volt / r_max[:, None]
unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
l2_shape = np.linalg.norm(unit(R)[:, None] - unit(MODEL_SIGN * u_raw), axis=2)

def pick(d):
    d = np.where(np.isfinite(d), d, np.inf)
    return np.where(np.isfinite(d.min(1)), np.array(list(LETTERS))[d.argmin(1)], '')

good = (np.isin(frames, pd.read_csv(MASK_QC.format(video=video)).frame.to_numpy())
        if os.path.exists(MASK_QC.format(video=video)) else np.zeros(len(frames), bool))
os.makedirs(f'{OUT_DIR}/{video}', exist_ok=True)
base = f'{OUT_DIR}/{video}/chunk{chunk}_of{N_CHUNKS}'
np.savez_compressed(base + '.npz', eod_rows=sel, frames=frames, recorded=R, u_raw=u_raw, factor=factor, pred_volt=pred,
                    nearer_pole=pole, l2_volt=l2_volt, l2_rel=l2_rel, l2_shape=l2_shape, pick_voltage=pick(l2_volt),
                    pick_shape=pick(l2_shape), good_masks=good, letters=np.array(list(LETTERS)), hyperparameters=json.dumps(HYPER))
out = pd.DataFrame({'eod_row': sel, 'Frame': frames, 'good_masks': good, 'r_max': r_max,
                    **{f'l2_volt_{f}': l2_volt[:, j] for j, f in enumerate(LETTERS)},
                    **{f'l2_rel_{f}': l2_rel[:, j] for j, f in enumerate(LETTERS)},
                    **{f'l2_shape_{f}': l2_shape[:, j] for j, f in enumerate(LETTERS)},
                    'pick_voltage': pick(l2_volt), 'pick_shape': pick(l2_shape)})
out.to_csv(base + '.csv', index=False)
print(f'done: {base}.npz / .csv, {len(out)} EODs, {(time.time() - t0) / 60:.1f} min', flush=True)
