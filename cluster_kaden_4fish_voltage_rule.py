"""Cluster job: my numerical model on the 71 4-fish recordings of 2024-02-20 (Kaden's masks), and the voltage rule for EOD
assignment. One SLURM array task = one (video, chunk) pair of the job table below: 71 videos x N_CHUNKS = 4 -> tasks 0-283.

WHAT IT COMPUTES, for every EOD of the task's frames and every fish f (each fish in turn the emitter, the other three as bodies)
  model      my 2D resistive field model (compute_electric_field.py): an N_GRID x N_GRID grid over the 912 x 912 px video
             frame; non-conductive tank walls fitted to the camera background (TANK_OUTLINE); the four plastic L-pieces of the
             inner square as non-conductive barriers (BARRIERS); electrodes moved just inside the walls (ELECTRODES); the
             other fish as bodies with a skin (as resistive as SKIN_LAMBDA_CM of water) and an interior of INTERIOR_SIGMA x
             water, from Kaden's masks of the EOD's own frame (no offset); the emitter as point currents along its mask
             midline (tail -> head, N_MIDLINE points, unit current, sink at the tail tip). Each fish costs one sparse solve.
  u_raw      unit-current pair voltages V(+) - V(-) of the 10 electrode pairs, raw model sign (n, 4 fish, 10 pairs)
  factor     a_f * e_k: median fish factor a_f times the median factor e_k of the signed electrode of pair p (+ or -) nearer to
             fish f's midline in this frame (FACTORS_FILE)
  pred_volt  factor * MODEL_SIGN * u_raw, predicted pair voltages in V
  recorded   recorded main-phase pattern: each pair's 25-sample trace demeaned, all 10 pairs at the sample of the largest |V| (V)
  l2_volt    ||recorded - pred_volt_f|| (V);  l2_rel = l2_volt / max_p |recorded_p|;
  l2_shape   ||recorded / max|recorded| - MODEL_SIGN u_f / max|u_f||| (shape only, no factors)
  picks      pick_voltage = fish with the smallest l2_volt (the voltage rule); pick_shape = fish with the smallest l2_shape

WHERE IT SAVES (folder created by the task): OUT_DIR/<video>/chunk<c>_of4.npz and .csv, OUT_DIR = results/eod_assign_4fish_kaden/
cluster_voltage_rule/N150_tank_barriers_int4 (the hyperparameters are in the folder name and in every file).
  .npz  eod_rows (row of the EOD in the EOD table), frames, recorded (n, 10), u_raw (n, 4, 10), factor (n, 4, 10),
        pred_volt (n, 4, 10), nearer_pole (n, 4, 10; 0 = +, 1 = -), l2_volt, l2_rel, l2_shape (n, 4), pick_voltage, pick_shape,
        good_masks (n,: the frame passes the mask QC), midline_fitted_here (n,), letters, hyperparameters (json string)
  .csv  one row per EOD: eod_row, Frame, good_masks, r_max, l2_volt_<f>, l2_rel_<f>, l2_shape_<f>, pick_voltage, pick_shape
By default a task recomputes and replaces its files; OVERWRITE=0 skips tasks whose two files already exist (to resume an
interrupted array). Files are written to .tmp and renamed when complete, so a half-written file never looks finished.

FILES IT READS (relative to the repository root, the folder of this script)
  code     compute_electric_field.py, eod_assign.py, io_utils.py, midline_methods.py (needs numpy >= 2, pandas, scipy, matplotlib)
  EODs     data/4 fish/Data_4fish_2024-02-20T<video>.csv                                         (rsync from the SSD)
  masks    data/track_Kaden/results_<video>_full/video2024-02-20T<video>/outputs/video2024-02-20T<video>.pkl, or the
           2024-02-20T<video>_R2-*.pkl name of 5 recordings in the same folder                     (rsync from the SSD)
  tank     data/electrodes/tank_outline_updated_4-fish.csv, electrode_locations_updated_4-fish.csv, tank_barriers_4-fish.csv (git)
  factors  results/eod_assign_4fish_kaden/clean_emitter_set_71/gain_mine_tank_int4/voltage_rule_factors_median.csv  (git)
  mask QC  results/eod_assign_4fish_kaden/<video>/mask_qc_good_frames.csv                                 (git; else all False)
  midlines results/eod_assign_4fish_kaden/<video>/midlines_by_frame.npz (rsync; optional: missing frames are fitted on the masks)

USAGE
  python cluster_kaden_4fish_voltage_rule.py --check     every input present and readable for all 71 videos? (run before sbatch)
  python cluster_kaden_4fish_voltage_rule.py --list      the job table (task index -> video, chunk)
  python cluster_kaden_4fish_voltage_rule.py 17          run task 17 here (without SLURM)
  MAX_FRAMES=10 OUT_DIR=/tmp/test python cluster_kaden_4fish_voltage_rule.py 0     quick test on 10 frames
  sbatch cluster_kaden_4fish_voltage_rule.slurm           all 284 tasks (task index from SLURM_ARRAY_TASK_ID)
"""
import os, sys, json, glob, time
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.environ.get('PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))   # repository root: data/ and results/
os.chdir(ROOT); sys.path.insert(0, ROOT)

# ================================================================================================================ hyperparameters
# model
N_GRID = 150                  # grid nodes per side over the 912 x 912 px video frame (~6.1 px = 0.43 cm per cell)
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
# 'fish A'..'fish D', 'electrode 1+'..'electrode 10-', column 'value'; an electrode missing from the file gets 1)
FACTORS_FILE = 'results/eod_assign_4fish_kaden/clean_emitter_set_71/gain_mine_tank_int4/voltage_rule_factors_median.csv'

# data
EOD_TABLE = 'data/4 fish/Data_4fish_2024-02-20T{video}.csv'
KADEN_PKL = 'data/track_Kaden/results_{video}_full/video2024-02-20T{video}/outputs/video2024-02-20T{video}.pkl'
MIDLINES = 'results/eod_assign_4fish_kaden/{video}/midlines_by_frame.npz'
MASK_QC = 'results/eod_assign_4fish_kaden/{video}/mask_qc_good_frames.csv'
N_SAMPLES = 25                # samples per EOD trace in the EOD table
LETTERS = 'ABCD'

# output
TAG = f'N{N_GRID}_tank{"_barriers" if USE_BARRIERS else ""}_int{INTERIOR_SIGMA:g}'
OUT_DIR = os.environ.get('OUT_DIR', f'results/eod_assign_4fish_kaden/cluster_voltage_rule/{TAG}')

# jobs
N_CHUNKS = 4                  # each video's EOD frames split into this many array tasks (~8-12k frames each)
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 0))   # tests only: solve just the first MAX_FRAMES frames of the chunk (0 = all)
OVERWRITE = os.environ.get('OVERWRITE', '1') == '1'   # 1 (default): recompute and replace existing outputs; 0: skip finished tasks

# the 71 recordings, fixed so a task index always means the same video (whatever is on disk)
VIDEOS = ['00_12_52', '00_32_53', '00_52_53', '01_12_53', '01_32_53', '01_52_53', '02_12_53', '02_32_53', '02_52_54',
          '03_12_54', '03_32_54', '03_52_54', '04_12_54', '04_32_54', '04_52_54', '05_12_55', '05_32_55', '05_52_55',
          '06_12_55', '06_32_55', '06_52_55', '07_12_56', '07_32_56', '07_52_56', '08_12_56', '08_32_56', '08_52_56',
          '09_12_56', '09_32_57', '09_52_57', '10_12_57', '10_32_57', '10_52_57', '11_12_57', '11_32_57', '11_52_58',
          '12_12_58', '12_32_58', '12_56_22', '13_16_22', '13_36_22', '13_56_22', '14_16_23', '14_36_23', '14_56_23',
          '15_16_23', '15_36_23', '15_56_23', '16_16_23', '16_36_24', '16_56_24', '17_16_24', '17_36_24', '17_56_24',
          '18_16_24', '18_36_24', '18_56_25', '19_16_25', '19_36_25', '19_56_25', '20_16_25', '20_36_25', '21_16_26',
          '21_36_26', '21_56_26', '22_16_26', '22_36_26', '22_56_26', '23_16_27', '23_36_27', '23_56_27']
df = pd.DataFrame(columns=['video', 'chunk'])          # in total len(VIDEOS) * N_CHUNKS = 284 tasks: --array=0-283
z = 0
for video in VIDEOS:
    for chunk in range(N_CHUNKS):
        df.loc[z, 'video'] = video
        df.loc[z, 'chunk'] = chunk
        z += 1
ORDER = [f'{p}{s}' for p in range(1, 11) for s in ('+', '-')]


# ===================================================================================================================== helpers
def kaden_pkl(video):
    p = KADEN_PKL.format(video=video)
    if os.path.exists(p): return p
    alt = sorted(glob.glob(os.path.join(os.path.dirname(p), f'2024-02-20T{video}_*.pkl')))
    if len(alt) != 1: raise FileNotFoundError(f'{video}: no Kaden pickle {p} (nor a unique 2024-02-20T{video}_*.pkl beside it)')
    return alt[0]


def out_base(video, chunk):
    return f'{OUT_DIR}/{video}/chunk{chunk}_of{N_CHUNKS}'


def check_inputs():
    '''Every input of every task present? Prints a table and the missing files; exit code 1 if anything required is missing.'''
    bad = []
    if int(np.__version__.split('.')[0]) < 2: bad.append(f'numpy {np.__version__} (Kaden pickles need numpy >= 2)')
    for f in (TANK_OUTLINE, ELECTRODES, FACTORS_FILE) + ((BARRIERS,) if USE_BARRIERS else ()):
        if not os.path.exists(f): bad.append(f)
    rows = []
    for v in VIDEOS:
        try: pk = kaden_pkl(v)
        except FileNotFoundError: pk = None
        rows.append(dict(video=v, eod_table=os.path.exists(EOD_TABLE.format(video=v)), kaden_pkl=pk is not None,
                         midlines=os.path.exists(MIDLINES.format(video=v)), mask_qc=os.path.exists(MASK_QC.format(video=v)),
                         chunks_done=sum(os.path.exists(out_base(v, c) + '.npz') and os.path.exists(out_base(v, c) + '.csv') for c in range(N_CHUNKS))))
        if not rows[-1]['eod_table']: bad.append(EOD_TABLE.format(video=v))
        if pk is None: bad.append(KADEN_PKL.format(video=v))
    T = pd.DataFrame(rows)
    print(f'repository root: {ROOT}\noutput folder:   {OUT_DIR}\nnumpy {np.__version__}, pandas {pd.__version__}')
    print(T.to_string(index=False))
    print(f'\n{T.eod_table.sum()}/71 EOD tables, {T.kaden_pkl.sum()}/71 Kaden pickles, {T.midlines.sum()}/71 midline files (optional), '
          f'{T.mask_qc.sum()}/71 mask-QC lists (optional); tasks done {T.chunks_done.sum()}/{len(df)}')
    try:
        import compute_electric_field, eod_assign, io_utils, midline_methods, scipy, matplotlib   # noqa: F401
    except Exception as e:
        bad.append(f'import failed: {e!r}')
    if bad:
        print('\nMISSING / PROBLEMS:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print('\nall required inputs present')


if len(sys.argv) > 1 and sys.argv[1] == '--list':
    print(df.to_string()); sys.exit()
if len(sys.argv) > 1 and sys.argv[1] == '--check':
    check_inputs(); sys.exit()

from io_utils import load_masks
from midline_methods import fish_midline_with_head
from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, build_A_shell, solve_potential,
                                    unflatten_to_grid, materials, tank_inside, confine_sources, SKIN_CM)
from eod_assign import pair_voltages, load_electrode_pairs, recorded_pair_traces_all

idx = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else int(sys.argv[1])
video = df.loc[idx, 'video']
chunk = int(df.loc[idx, 'chunk'])
base = out_base(video, chunk)
HYPER = dict(N_GRID=N_GRID, INTERIOR_SIGMA=INTERIOR_SIGMA, SKIN_LAMBDA_CM=SKIN_LAMBDA_CM, SINK_FRAC=SINK_FRAC, PX_PER_CM=PX_PER_CM,
             MODEL_SIGN=MODEL_SIGN, N_MIDLINE=N_MIDLINE, TANK_OUTLINE=TANK_OUTLINE, ELECTRODES=ELECTRODES,
             BARRIERS=BARRIERS if USE_BARRIERS else None, FACTORS_FILE=FACTORS_FILE, N_CHUNKS=N_CHUNKS, MAX_FRAMES=MAX_FRAMES,
             video=video, chunk=chunk, task=idx)


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
    '''Unit-current pair voltages (4 fish, 10 pairs) of video frame k, each fish in turn emitting; the frame's midlines; and
       whether the midlines had to be fitted here (not in the midline file).'''
    fm = np.asarray(MASKS[k]); M = MIDS.get(int(k)); fitted = M is None
    if fitted: M = fit_midlines(fm)
    u = np.full((4, len(PAIRS)), np.nan)
    for f in range(4):
        if not np.isfinite(M[f]).all(): continue
        b = confine_sources(build_multipole_b(*multipole_sources(M[f], I0=1.0, sink_frac=SINK_FRAC), 1.0 / (N_GRID - 1), N_GRID), INSIDE)
        A = build_A_shell(occupancy_field(fm, N_GRID, skip=f), N_GRID, **MAT, inside=INSIDE)
        u[f] = pair_voltages(unflatten_to_grid(solve_potential(A, b), N_GRID).real, PAIRS, N_GRID)
    return int(k), u, M, fitted


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


def save_atomic(path, writer):
    tmp = path + '.tmp'; writer(tmp); os.replace(tmp, path)


# ===================================================================================================================== setup
t0 = time.time()
print(f'task {idx}: video {video}, chunk {chunk + 1}/{N_CHUNKS}, {N_WORKERS} worker(s); root {ROOT}; output {base}.npz/.csv', flush=True)
print('hyperparameters: ' + json.dumps(HYPER), flush=True)
if not OVERWRITE and os.path.exists(base + '.npz') and os.path.exists(base + '.csv'):
    print('  already done, skipped (OVERWRITE=0)'); sys.exit(0)
assert int(np.__version__.split('.')[0]) >= 2, f'numpy {np.__version__}: Kaden pickles need numpy >= 2'
for f in (EOD_TABLE.format(video=video), TANK_OUTLINE, ELECTRODES, FACTORS_FILE) + ((BARRIERS,) if USE_BARRIERS else ()):
    if not os.path.exists(f): raise FileNotFoundError(f'missing input {os.path.join(ROOT, f)} (python {sys.argv[0]} --check lists all)')

MASKS = load_masks(kaden_pkl(video))
MIDS = {}
if os.path.exists(MIDLINES.format(video=video)):
    zm = np.load(MIDLINES.format(video=video)); MIDS = dict(zip(zm['frames'].tolist(), zm['mids'])); del zm
else:
    print(f'  no midline file {MIDLINES.format(video=video)}: midlines are fitted on the masks (slower)', flush=True)
PAIRS = load_electrode_pairs(ELECTRODES, P); PIDS = [p for p, _, _ in PAIRS]
assert PIDS == list(range(1, 11)), f'electrode pairs out of order in {ELECTRODES}: {PIDS}'
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
print(f'  factors: fish {dict(zip(LETTERS, A_FISH.round(4)))}; electrodes {E_ELEC.min():.2f}-{E_ELEC.max():.2f}', flush=True)

eods = pd.read_csv(EOD_TABLE.format(video=video), usecols=['Frame'] + [f'electrode_{p}_time_{i}' for p in PIDS for i in range(1, N_SAMPLES + 1)])
frames_all = np.unique(eods.Frame.to_numpy())
my_frames = np.array_split(frames_all, N_CHUNKS)[chunk]
if MAX_FRAMES: my_frames = my_frames[:MAX_FRAMES]
sel = np.flatnonzero(np.isin(eods.Frame.to_numpy(), my_frames))
frames = eods.Frame.to_numpy()[sel]
V = recorded_pair_traces_all(eods.iloc[sel], PIDS, N_SAMPLES); del eods
R = V[np.arange(len(V)), :, np.abs(V).max(1).argmax(1)]; del V                                 # (n, 10) recorded main phase, V
n_missing_mid = sum(int(k) not in MIDS for k in my_frames)
print(f'  {len(sel)} EODs on {len(my_frames)} frames ({n_missing_mid} without a stored midline); setup {time.time() - t0:.0f} s', flush=True)

# ===================================================================================================================== solve
U, POLE, FITTED = {}, {}, {}
work = [int(k) for k in my_frames]
def collect(j, res):
    k, u, M, fitted = res; U[k] = u; POLE[k] = nearer_pole(M); FITTED[k] = fitted
    if j % 1000 == 0 or j == len(work):
        el = time.time() - t0; print(f'  {j}/{len(work)} frames, {el / 60:.1f} min, ~{el / j * (len(work) - j) / 60:.0f} min left', flush=True)
if N_WORKERS > 1:
    with mp.get_context('fork').Pool(N_WORKERS) as pool:
        for j, res in enumerate(pool.imap_unordered(solve_frame, work, chunksize=8), 1): collect(j, res)
else:
    for j, k in enumerate(work, 1): collect(j, solve_frame(k))

# ===================================================================================================================== voltage rule
u_raw = np.stack([U[int(k)] for k in frames])                                                   # (n, 4, 10), raw model sign
pole = np.stack([POLE[int(k)] for k in frames])                                                 # (n, 4, 10)
fitted_here = np.array([FITTED[int(k)] for k in frames])
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
os.makedirs(os.path.dirname(base), exist_ok=True)
def write_npz(p):
    with open(p, 'wb') as fh:                                     # a file handle: numpy adds no '.npz' to the .tmp name
        np.savez_compressed(fh, eod_rows=sel, frames=frames, recorded=R, u_raw=u_raw, factor=factor, pred_volt=pred,
                            nearer_pole=pole, l2_volt=l2_volt, l2_rel=l2_rel, l2_shape=l2_shape, pick_voltage=pick(l2_volt),
                            pick_shape=pick(l2_shape), good_masks=good, midline_fitted_here=fitted_here,
                            letters=np.array(list(LETTERS)), hyperparameters=json.dumps(HYPER))
save_atomic(base + '.npz', write_npz)
out = pd.DataFrame({'eod_row': sel, 'Frame': frames, 'good_masks': good, 'r_max': r_max,
                    **{f'l2_volt_{f}': l2_volt[:, j] for j, f in enumerate(LETTERS)},
                    **{f'l2_rel_{f}': l2_rel[:, j] for j, f in enumerate(LETTERS)},
                    **{f'l2_shape_{f}': l2_shape[:, j] for j, f in enumerate(LETTERS)},
                    'pick_voltage': pick(l2_volt), 'pick_shape': pick(l2_shape)})
save_atomic(base + '.csv', lambda p: out.to_csv(p, index=False))
ok = out.pick_voltage != ''
print(f'done: {base}.npz / .csv, {len(out)} EODs ({ok.sum()} with a voltage pick; shares ' +
      ' '.join(f'{f} {(out.pick_voltage[ok] == f).mean() * 100:.1f}%' for f in LETTERS) +
      f'), {(time.time() - t0) / 60:.1f} min', flush=True)
