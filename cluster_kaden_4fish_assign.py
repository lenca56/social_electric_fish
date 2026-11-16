"""Cluster job: EOD assignment on the 71 4-fish recordings of 2024-02-20 with the fitted mask midlines, with either my numerical
model or Denis' analytic pipeline (MODEL), and the voltage rule. One SLURM array task = one (video, chunk) pair of the job table
below: 71 videos x N_CHUNKS = 4 -> tasks 0-283.

Run cluster_kaden_4fish_midlines.py first: this job reads its midlines (MIDLINES) and predicts nothing for a fish without one.

MODELS (MODEL = 'mine' or 'denis'; every other model setting is in the hyperparameters below)
  mine   my 2D resistive field model (compute_electric_field.py): N_GRID x N_GRID grid over the 912 x 912 px frame;
         non-conductive tank walls (TANK_OUTLINE) and plastic L-barriers (BARRIERS); the other three fish as bodies with a skin
         (as resistive as SKIN_LAMBDA_CM of water) and an interior of INTERIOR_SIGMA x water, from Kaden's masks of the EOD's own
         frame; the emitter as unit point currents along its mask midline, sink at the tail tip. One sparse solve per fish.
         model_raw = unit-current pair voltages V(+) - V(-) in the model's raw sign (MODEL_SIGN = -1 against the recording).
         recorded  = main phase: each pair's 25-sample trace demeaned, all 10 pairs at the sample of the largest |V|.
  denis  Denis' assign_eods (EODAssignment repo, DENIS_REPO; needs numba) through silent_fish.run_pipeline with the mask
         midlines as his 101 point currents (identity = Kaden's letters, no refractory step), electrodes DENIS_ELECTRODES.
         It needs every fish's midline, so an EOD where one fish has none gets no prediction.
         model_raw = his predicted_PTP (signed peak-to-peak, his units; MODEL_SIGN = +1); recorded = his recorded_PTP.
         Also saved: his similarity per fish and his own pick (highest similarity).

VOLTAGE RULE (both models): pred_volt = a_f * e_k * MODEL_SIGN * model_raw, with a_f the fish factor and e_k the factor of the
signed electrode k of pair p (+ or -) nearer to fish f's midline (FACTORS: medians fitted on clean isolated EODs, one file per
model); the EOD goes to the fish with the smallest ||recorded - pred_volt|| (pick_voltage). The shape rule scales recorded and
MODEL_SIGN * model_raw to max |v| = 1 and takes the smallest L2 (pick_shape).

WHERE IT SAVES (folder created by the task): OUT_DIR/<video>/chunk<c>_of4.npz and .csv,
OUT_DIR = results/eod_assign_4fish_kaden/cluster_assign/<TAG>, TAG = mine_N150_tank_barriers_int4 or denis_analytic
  .npz  eod_rows (row of the EOD in the EOD table), frames, recorded (n, 10), model_raw (n, 4, 10), model_sign, factor (n, 4, 10),
        nearer_pole (n, 4, 10; 0 = +, 1 = -), pred_volt (n, 4, 10), l2_volt, l2_rel (= l2_volt / max|recorded|), l2_shape (n, 4),
        pick_voltage, pick_shape (n,), has_midline (n, 4), good_masks (n,: the frame passes the mask QC), letters,
        hyperparameters (json); denis only: similarity (n, 4), pick_similarity (n,)
  .csv  one row per EOD: eod_row, Frame, good_masks, r_max, l2_volt_<f>, l2_rel_<f>, l2_shape_<f>, pick_voltage, pick_shape
        (+ pick_similarity for denis)
By default a task recomputes and replaces its files; OVERWRITE=0 skips tasks whose two files exist (to resume an interrupted
array). Files are written to .tmp and renamed when complete.

FILES IT READS (relative to the repository root, the folder of this script)
  code      mine: compute_electric_field.py, eod_assign.py, io_utils.py (numpy >= 2.4, pandas, scipy)
            denis: DENIS_REPO/src/*.py and DENIS_REPO/analysis/notebooks/silent_fish.py (numba, pandas, scipy, matplotlib)
  EODs      data/4 fish/Data_4fish_2024-02-20T<video>.csv
  midlines  results/eod_assign_4fish_kaden/cluster_midlines/<video>/midlines_all_frames.npz (cluster_kaden_4fish_midlines.py)
  masks     mine only: data/track_Kaden/results_<video>_full/video2024-02-20T<video>/outputs/video2024-02-20T<video>.pkl (or the
            2024-02-20T<video>_R2-*.pkl name of 5 recordings)
  tank      data/electrodes/*_4-fish.csv (git); factors (git); mask QC results/eod_assign_4fish_kaden/<video>/mask_qc_good_frames.csv

USAGE (MODEL=denis in front of any of these for Denis' pipeline)
  python cluster_kaden_4fish_assign.py --check     every input present and readable for all 71 videos?
  python cluster_kaden_4fish_assign.py --list      the job table (task index -> video, chunk)
  python cluster_kaden_4fish_assign.py 17          run task 17 here (without SLURM)
  MAX_FRAMES=10 OUT_DIR=results/test python cluster_kaden_4fish_assign.py 0     quick test on 10 frames
  MODEL=mine sbatch cluster_kaden_4fish_assign.slurm                           all 284 tasks
"""
import os, sys, re, json, glob, time, shutil
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = os.environ.get('PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))   # repository root: data/ and results/
os.chdir(ROOT); sys.path.insert(0, ROOT)

# ================================================================================================================ hyperparameters
MODEL = os.environ.get('MODEL', 'mine')     # 'mine': my numerical model; 'denis': Denis' analytic pipeline
assert MODEL in ('mine', 'denis'), f"MODEL = {MODEL!r}: must be 'mine' or 'denis'"

# my numerical model (MODEL = 'mine')
N_GRID = 150                  # grid nodes per side over the 912 x 912 px video frame (~6.1 px = 0.43 cm per cell)
INTERIOR_SIGMA = 4.0          # interior conductivity of the other fish's bodies, x water
SKIN_LAMBDA_CM = 0.1          # skin resistance: as resistive as this many cm of water (skin thickness SKIN_CM = 0.005 cm)
SINK_FRAC = 0.0               # position of the current sink along the midline (0 = tail tip)
PX_PER_CM = 14.2              # model scale: the 60 x 60 cm tank spans ~852 px inside its walls
TANK_OUTLINE = 'data/electrodes/tank_outline_updated_4-fish.csv'          # wall polygon, video px (x = column, y = row)
ELECTRODES = 'data/electrodes/electrode_locations_updated_4-fish.csv'     # electrodes moved just inside the walls
BARRIERS = 'data/electrodes/tank_barriers_4-fish.csv'                     # 4 plastic L-pieces, one row per arm (segment)
USE_BARRIERS = True

# Denis' analytic pipeline (MODEL = 'denis')
DENIS_REPO = os.environ.get('DENIS_REPO', os.path.join(os.path.dirname(ROOT), 'EODAssignment'))   # default: beside this repo
DENIS_ELECTRODES = 'data/electrodes/electrodes_signed_px_coord_4-fish.csv'   # the electrode layout his pipeline uses

# per model: sign of model_raw against the recording, the electrodes that define the nearer pole, and the voltage-rule factors
# (rows 'fish A'..'fish D', 'electrode 1+'..'electrode 10-', column 'value'; an electrode missing from the file gets 1)
MODEL_SIGN = {'mine': -1, 'denis': +1}[MODEL]
POLE_ELECTRODES = {'mine': ELECTRODES, 'denis': DENIS_ELECTRODES}[MODEL]
FACTORS_FILE = {'mine': 'results/eod_assign_4fish_kaden/clean_emitter_set_71/gain_mine_tank_int4/voltage_rule_factors_median.csv',
                'denis': 'results/eod_assign_4fish_kaden/l2_distributions/denis_71/denis_voltage_rule_factors_median.csv'}[MODEL]

# common
N_MIDLINE = 101               # midline points (tail -> head) = point currents (both models)
P = 912                       # video frame size, px
N_SAMPLES = 25                # samples per EOD trace in the EOD table
LETTERS = 'ABCD'

# data
EOD_TABLE = 'data/4 fish/Data_4fish_2024-02-20T{video}.csv'
KADEN_PKL = 'data/track_Kaden/results_{video}_full/video2024-02-20T{video}/outputs/video2024-02-20T{video}.pkl'
MIDLINES = os.environ.get('MIDLINES', 'results/eod_assign_4fish_kaden/cluster_midlines/{video}/midlines_all_frames.npz')
MASK_QC = 'results/eod_assign_4fish_kaden/{video}/mask_qc_good_frames.csv'

# output
TAG = {'mine': f'mine_N{N_GRID}_tank{"_barriers" if USE_BARRIERS else ""}_int{INTERIOR_SIGMA:g}', 'denis': 'denis_analytic'}[MODEL]
OUT_DIR = os.environ.get('OUT_DIR', f'results/eod_assign_4fish_kaden/cluster_assign/{TAG}')

# jobs
N_CHUNKS = 4                  # each video's EOD frames split into this many array tasks (~8-12k EODs each)
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))   # mine: worker processes; denis: numba threads
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 0))   # tests only: just the first MAX_FRAMES frames of the chunk (0 = all)
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
JOBS = pd.DataFrame([(v, c) for v in VIDEOS for c in range(N_CHUNKS)], columns=['video', 'chunk'])   # 284 tasks: --array=0-283
ORDER = [f'{p}{s}' for p in range(1, 11) for s in ('+', '-')]
NUMPY_VERSION = tuple(int(x) for x in np.__version__.split('.')[:2])


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
    if MODEL == 'mine' and NUMPY_VERSION < (2, 4): bad.append(f'numpy {np.__version__} (Kaden pickles need numpy >= 2.4, from conda-forge)')
    need = [POLE_ELECTRODES, FACTORS_FILE] + ([TANK_OUTLINE, ELECTRODES] + ([BARRIERS] if USE_BARRIERS else []) if MODEL == 'mine' else [])
    bad += [f for f in need if not os.path.exists(f)]
    if MODEL == 'denis':
        for f in ('src/assign_eods.py', 'src/field_model.py', 'analysis/notebooks/silent_fish.py'):
            if not os.path.exists(os.path.join(DENIS_REPO, f)): bad.append(f'Denis repo file {os.path.join(DENIS_REPO, f)} (set DENIS_REPO)')
        try:
            import numba   # noqa: F401
        except Exception as e:
            bad.append(f'numba (Denis pipeline): {e!r}')
    rows = []
    for v in VIDEOS:
        try: pk = kaden_pkl(v)
        except FileNotFoundError: pk = None
        rows.append(dict(video=v, eod_table=os.path.exists(EOD_TABLE.format(video=v)), midlines=os.path.exists(MIDLINES.format(video=v)),
                         kaden_pkl=pk is not None, mask_qc=os.path.exists(MASK_QC.format(video=v)),
                         chunks_done=sum(os.path.exists(out_base(v, c) + '.npz') and os.path.exists(out_base(v, c) + '.csv') for c in range(N_CHUNKS))))
        if not rows[-1]['eod_table']: bad.append(EOD_TABLE.format(video=v))
        if not rows[-1]['midlines']: bad.append(MIDLINES.format(video=v) + '  (run cluster_kaden_4fish_midlines.py first)')
        if MODEL == 'mine' and pk is None: bad.append(KADEN_PKL.format(video=v))
    T = pd.DataFrame(rows)
    print(f'model:           {MODEL}\nrepository root: {ROOT}\noutput folder:   {OUT_DIR}\nmidlines:        {MIDLINES}'
          + (f'\nDenis repo:      {DENIS_REPO}' if MODEL == 'denis' else '') + f'\nnumpy {np.__version__}, pandas {pd.__version__}')
    print(T.to_string(index=False))
    print(f'\n{T.eod_table.sum()}/71 EOD tables, {T.midlines.sum()}/71 midline files, {T.kaden_pkl.sum()}/71 Kaden pickles '
          f'({"needed" if MODEL == "mine" else "not needed"}), {T.mask_qc.sum()}/71 mask-QC lists (optional); '
          f'tasks done {T.chunks_done.sum()}/{len(JOBS)}')
    if MODEL == 'mine':
        try:
            import compute_electric_field, eod_assign, io_utils, scipy   # noqa: F401
        except Exception as e:
            bad.append(f'import failed: {e!r}')
    if bad:
        print('\nMISSING / PROBLEMS:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print('\nall required inputs present')


if len(sys.argv) > 1 and sys.argv[1] == '--list':
    print(JOBS.to_string()); sys.exit()
if len(sys.argv) > 1 and sys.argv[1] == '--check':
    check_inputs(); sys.exit()

idx = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else int(sys.argv[1])
video, chunk = JOBS.loc[idx, 'video'], int(JOBS.loc[idx, 'chunk'])
base = out_base(video, chunk)
HYPER = dict(MODEL=MODEL, MODEL_SIGN=MODEL_SIGN, POLE_ELECTRODES=POLE_ELECTRODES, FACTORS_FILE=FACTORS_FILE, MIDLINES=MIDLINES,
             N_MIDLINE=N_MIDLINE, N_CHUNKS=N_CHUNKS, MAX_FRAMES=MAX_FRAMES, video=video, chunk=chunk, task=idx,
             **(dict(N_GRID=N_GRID, INTERIOR_SIGMA=INTERIOR_SIGMA, SKIN_LAMBDA_CM=SKIN_LAMBDA_CM, SINK_FRAC=SINK_FRAC, PX_PER_CM=PX_PER_CM,
                     TANK_OUTLINE=TANK_OUTLINE, ELECTRODES=ELECTRODES, BARRIERS=BARRIERS if USE_BARRIERS else None)
                if MODEL == 'mine' else dict(DENIS_REPO=DENIS_REPO, DENIS_ELECTRODES=DENIS_ELECTRODES)))


def poles(path):
    '''(10, 2) + and (10, 2) - electrode positions (x, y) video px of pairs 1-10.'''
    EL = pd.read_csv(path).sort_values('pair')
    assert EL.pair.tolist() == list(range(1, 11)), f'electrode pairs in {path}: {EL.pair.tolist()}'
    plus = np.array([(r.x1, r.y1) if str(r.pol1).strip() == '+' else (r.x2, r.y2) for _, r in EL.iterrows()], float)
    minus = np.array([(r.x2, r.y2) if str(r.pol1).strip() == '+' else (r.x1, r.y1) for _, r in EL.iterrows()], float)
    return plus, minus


def nearer_pole(mids, plus, minus):
    '''mids (n, 4, N_MIDLINE, 2) (row, col) / (P - 1) -> (n, 4, 10): 0 where pair p's + electrode is nearer to fish f's midline,
       1 where its - electrode is (0 for a fish without a midline).'''
    out = np.zeros(mids.shape[:2] + (10,), int)
    for s in range(0, len(mids), 1000):
        xy = mids[s:s + 1000, :, :, ::-1] * (P - 1)                                           # (n, 4, N_MIDLINE, x/y) px
        dp = np.linalg.norm(xy[:, :, :, None] - plus[None, None, None], axis=-1).min(2)
        dm = np.linalg.norm(xy[:, :, :, None] - minus[None, None, None], axis=-1).min(2)
        out[s:s + 1000] = np.where(np.isfinite(dp) & np.isfinite(dm), dm < dp, 0)
    return out


def save_atomic(path, writer):
    tmp = path + '.tmp'; writer(tmp); os.replace(tmp, path)


# ===================================================================================================================== my model
def mine_context():
    from io_utils import load_masks
    from compute_electric_field import tank_inside, materials, SKIN_CM
    from eod_assign import load_electrode_pairs
    inside = tank_inside(pd.read_csv(TANK_OUTLINE)[['x', 'y']].to_numpy(), N_GRID, P)
    if USE_BARRIERS:                                  # plastic L-pieces: each arm sampled densely, every sample removes its nearest node
        s = (N_GRID - 1) / (P - 1)
        for _, r in pd.read_csv(BARRIERS).iterrows():
            n = int(np.ceil(np.hypot(r.x2 - r.x1, r.y2 - r.y1) * s * 4)) + 2
            x, y = np.linspace(r.x1, r.x2, n) * s, np.linspace(r.y1, r.y2, n) * s
            inside[np.clip(np.rint(y).astype(int), 0, N_GRID - 1), np.clip(np.rint(x).astype(int), 0, N_GRID - 1)] = False
    mat = dict(materials(N_GRID, P, PX_PER_CM), g_interior=INTERIOR_SIGMA, g_skin=1 / (SKIN_LAMBDA_CM / SKIN_CM + 1 / INTERIOR_SIGMA))
    pairs = load_electrode_pairs(ELECTRODES, P)
    assert [p for p, _, _ in pairs] == list(range(1, 11)), f'electrode pairs out of order in {ELECTRODES}'
    return dict(masks=load_masks(kaden_pkl(video)), inside=inside, mat=mat, pairs=pairs)


def mine_solve_frame(job):
    '''(frame, (4, 10) unit-current pair voltages, raw sign; NaN for a fish without a midline), each fish in turn emitting.'''
    from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, build_A_shell, solve_potential,
                                        unflatten_to_grid, confine_sources)
    from eod_assign import pair_voltages
    k, M = job; C = CTX; fm = np.asarray(C['masks'][k]); u = np.full((4, 10), np.nan)
    for f in range(4):
        if not np.isfinite(M[f]).all(): continue
        b = confine_sources(build_multipole_b(*multipole_sources(M[f], I0=1.0, sink_frac=SINK_FRAC), 1.0 / (N_GRID - 1), N_GRID), C['inside'])
        A = build_A_shell(occupancy_field(fm, N_GRID, skip=f), N_GRID, **C['mat'], inside=C['inside'])
        u[f] = pair_voltages(unflatten_to_grid(solve_potential(A, b), N_GRID).real, C['pairs'], N_GRID)
    return k, u


def run_mine(frames, mids_eod, sel):
    '''model_raw (n, 4, 10) and recorded main phase (n, 10, V) for the EODs `sel` of the EOD table (frames, midlines per EOD).'''
    global CTX
    from eod_assign import recorded_pair_traces_all
    eods = pd.read_csv(EOD_TABLE.format(video=video), usecols=['Frame'] + [f'electrode_{p}_time_{i}' for p in range(1, 11) for i in range(1, N_SAMPLES + 1)])
    V = recorded_pair_traces_all(eods.iloc[sel], list(range(1, 11)), N_SAMPLES); del eods
    recorded = V[np.arange(len(V)), :, np.abs(V).max(1).argmax(1)]; del V                      # (n, 10) main phase, V
    CTX = mine_context()
    uniq, first = np.unique(frames, return_index=True)                                         # one solve per frame
    work = [(int(k), mids_eod[i]) for k, i in zip(uniq, first)]
    print(f'  my model: {len(work)} frames to solve, {N_WORKERS} worker(s)', flush=True)
    U, t1 = {}, time.time()
    def collect(j, res):
        U[res[0]] = res[1]
        if j % 1000 == 0 or j == len(work):
            el = time.time() - t1; print(f'  {j}/{len(work)} frames, {el / 60:.1f} min, ~{el / j * (len(work) - j) / 60:.0f} min left', flush=True)
    if N_WORKERS > 1:
        with mp.get_context('fork').Pool(N_WORKERS) as pool:
            for j, res in enumerate(pool.imap_unordered(mine_solve_frame, work, chunksize=8), 1): collect(j, res)
    else:
        for j, w in enumerate(work, 1): collect(j, mine_solve_frame(w))
    return np.stack([U[int(k)] for k in frames]), recorded, {}


# ===================================================================================================================== Denis
def run_denis(frames, mids_eod, sel):
    '''model_raw = his predicted_PTP (n, 4, 10), recorded = his recorded_PTP (n, 10), and {similarity, pick_similarity}; NaN / ''
       for EODs where a fish has no midline (his pipeline needs all four).'''
    os.environ.setdefault('NUMBA_NUM_THREADS', str(N_WORKERS))
    sys.path[:0] = [os.path.join(DENIS_REPO, 'analysis', 'notebooks'), DENIS_REPO]
    import silent_fish as sf
    keep = lambda c: c in ('Frame', 'Time') or c.startswith('electrode_') or any(c.startswith(p + '_') for p in sf.TRACKED_POINTS)
    src = pd.read_csv(EOD_TABLE.format(video=video), usecols=keep).iloc[sel].reset_index(drop=True)
    rename = {f'{p}_score_{i}': f'score_{p}_{i}' for p in sf.TRACKED_POINTS for i in range(1, 5)}
    raw = pd.concat([src[['Frame', 'Time']], src[[f'{p}_{k}_{i}' for i in range(1, 5) for p in sf.TRACKED_POINTS for k in ('x', 'y', 'score')]]
                     .rename(columns=rename), src.filter(like='electrode_')], axis=1); del src
    has = np.isfinite(mids_eod).all(axis=(1, 2, 3))                                            # every fish has a midline
    cm = np.full(mids_eod.shape, np.nan)
    for i in np.flatnonzero(has):
        for f in range(4): cm[i, f] = sf.resample(mids_eod[i, f][:, ::-1] * (P - 1), N_MIDLINE) * sf.PX_TO_CM   # (x, y) cm, even arc length
    work = Path(f'{OUT_DIR}/{video}/denis_work_chunk{chunk}'); work.mkdir(parents=True, exist_ok=True)
    csv = work / f'Data_4fish_2024-02-20T{video}.csv'                    # his loader reads the date from the path: exactly one allowed
    assert len(re.findall(r'\d{4}-\d{2}-\d{2}', str(csv.resolve()))) == 1, f'the path {csv.resolve()} must contain exactly one date'
    sf.midline_input(raw[has], cm[has]).to_csv(csv, index=False)
    print(f'  Denis: {has.sum()} / {len(raw)} EODs with every fish midline, {os.environ["NUMBA_NUM_THREADS"]} numba thread(s)', flush=True)
    d = sf.run_pipeline(csv, Path(DENIS_ELECTRODES).resolve(), 4, midlines_cm=cm[has])
    assert (d.Frame.to_numpy() == raw.Frame.to_numpy()[has]).all(), 'Denis output rows do not match the input EODs'
    shutil.rmtree(work)
    n = len(raw); model_raw = np.full((n, 4, 10), np.nan); recorded = np.full((n, 10), np.nan); sim = np.full((n, 4), np.nan)
    if has.any():
        model_raw[has] = np.stack([np.stack(d[f'predicted_PTP_{f}'].to_numpy()) for f in LETTERS], axis=1)
        recorded[has] = np.stack(d['recorded_PTP'].to_numpy())
        sim[has] = np.stack([d[f'similarity_{f}'].to_numpy(float) for f in LETTERS], axis=1)
    pick_sim = np.full(n, '', object); pick_sim[has] = d.discharging_fish.to_numpy()
    return model_raw, recorded, dict(similarity=sim, pick_similarity=pick_sim.astype(str))


# ===================================================================================================================== run
t0 = time.time()
print(f'task {idx}: model {MODEL}, video {video}, chunk {chunk + 1}/{N_CHUNKS}, {N_WORKERS} cpu(s); root {ROOT}; output {base}.npz/.csv', flush=True)
print('hyperparameters: ' + json.dumps(HYPER), flush=True)
if not OVERWRITE and os.path.exists(base + '.npz') and os.path.exists(base + '.csv'):
    print('  already done, skipped (OVERWRITE=0)'); sys.exit(0)
if MODEL == 'mine':
    assert NUMPY_VERSION >= (2, 4), f'numpy {np.__version__}: Kaden pickles need numpy >= 2.4 (install it from conda-forge)'
for f in (EOD_TABLE.format(video=video), MIDLINES.format(video=video), POLE_ELECTRODES, FACTORS_FILE):
    if not os.path.exists(f): raise FileNotFoundError(f'missing input {os.path.join(ROOT, f)} (python {sys.argv[0]} --check lists all)')

# the chunk's EODs and their midlines
frames_eod = pd.read_csv(EOD_TABLE.format(video=video), usecols=['Frame']).Frame.to_numpy()
my_frames = np.array_split(np.unique(frames_eod), N_CHUNKS)[chunk]
if MAX_FRAMES: my_frames = my_frames[:MAX_FRAMES]
sel = np.flatnonzero(np.isin(frames_eod, my_frames)); frames = frames_eod[sel]
zm = np.load(MIDLINES.format(video=video)); mf = zm['frames']
j = np.clip(np.searchsorted(mf, frames), 0, len(mf) - 1); found = mf[j] == frames
mids_eod = np.full((len(frames), 4, N_MIDLINE, 2), np.nan); mids_eod[found] = zm['mids'][j[found]]; del zm
has_midline = np.isfinite(mids_eod).all(axis=(2, 3))
print(f'  {len(sel)} EODs on {len(my_frames)} frames; {(~found).sum()} EODs on frames missing from the midline file; midlines per fish '
      + ' '.join(f'{f} {has_midline[:, i].sum()}' for i, f in enumerate(LETTERS)) + f'; setup {time.time() - t0:.0f} s', flush=True)

model_raw, recorded, extra = (run_mine if MODEL == 'mine' else run_denis)(frames, mids_eod, sel)

# voltage rule and shape rule
FAC = pd.read_csv(FACTORS_FILE); FAC['factor'] = FAC['factor'].str.replace('−', '-'); FAC = FAC.set_index('factor').value
A_FISH = np.array([FAC[f'fish {f}'] for f in LETTERS])
E_ELEC = np.array([FAC.get(f'electrode {e}', 1.0) for e in ORDER]).reshape(10, 2)              # (pair, + / -)
print(f'  factors ({FACTORS_FILE}): fish {dict(zip(LETTERS, A_FISH.round(4)))}; electrodes {E_ELEC.min():.2f}-{E_ELEC.max():.2f}', flush=True)
pole = nearer_pole(mids_eod, *poles(POLE_ELECTRODES))                                          # (n, 4, 10)
factor = A_FISH[None, :, None] * E_ELEC[np.arange(10)[None, None, :], pole]                    # (n, 4, 10)
pred = factor * MODEL_SIGN * model_raw
l2_volt = np.linalg.norm(recorded[:, None] - pred, axis=2)
r_max = np.abs(recorded).max(1)
l2_rel = l2_volt / r_max[:, None]
unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
l2_shape = np.linalg.norm(unit(recorded)[:, None] - unit(MODEL_SIGN * model_raw), axis=2)

def pick(d):
    d = np.where(np.isfinite(d), d, np.inf)
    return np.where(np.isfinite(d.min(1)), np.array(list(LETTERS))[d.argmin(1)], '')

good = (np.isin(frames, pd.read_csv(MASK_QC.format(video=video)).frame.to_numpy())
        if os.path.exists(MASK_QC.format(video=video)) else np.zeros(len(frames), bool))
os.makedirs(os.path.dirname(base), exist_ok=True)
def write_npz(p):
    with open(p, 'wb') as fh:                                     # a file handle: numpy adds no '.npz' to the .tmp name
        np.savez_compressed(fh, eod_rows=sel, frames=frames, recorded=recorded, model_raw=model_raw, model_sign=MODEL_SIGN,
                            factor=factor, nearer_pole=pole, pred_volt=pred, l2_volt=l2_volt, l2_rel=l2_rel, l2_shape=l2_shape,
                            pick_voltage=pick(l2_volt), pick_shape=pick(l2_shape), has_midline=has_midline, good_masks=good,
                            letters=np.array(list(LETTERS)), hyperparameters=json.dumps(HYPER), **extra)
save_atomic(base + '.npz', write_npz)
out = pd.DataFrame({'eod_row': sel, 'Frame': frames, 'good_masks': good, 'r_max': r_max,
                    **{f'l2_volt_{f}': l2_volt[:, j] for j, f in enumerate(LETTERS)},
                    **{f'l2_rel_{f}': l2_rel[:, j] for j, f in enumerate(LETTERS)},
                    **{f'l2_shape_{f}': l2_shape[:, j] for j, f in enumerate(LETTERS)},
                    'pick_voltage': pick(l2_volt), 'pick_shape': pick(l2_shape),
                    **({'pick_similarity': extra['pick_similarity']} if 'pick_similarity' in extra else {})})
save_atomic(base + '.csv', lambda p: out.to_csv(p, index=False))
ok = out.pick_voltage != ''
print(f'done: {base}.npz / .csv, {len(out)} EODs ({ok.sum()} with a voltage pick; shares ' +
      ' '.join(f'{f} {(out.pick_voltage[ok] == f).mean() * 100:.1f}%' for f in LETTERS) +
      f'), {(time.time() - t0) / 60:.1f} min', flush=True)
