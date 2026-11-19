"""Cluster job, step 2 of 3 (see cluster_3fish_common.py): raw model predictions for every EOD of a 3-fish recording, with the
midlines (step 1) and masks of one efish4-track stage, for one model setting. No factors and no assignment here (step 3).
One SLURM array task = one (recording, stage, setting): 8 recordings x 10 stages x len(SETTINGS) = 400 tasks with the default
SETTINGS (0-399; python cluster_3fish_stage_predict.py --list).
MASK_STAGES=final: only the run's final masks, 8 x 5 = 40 tasks (recording r = tasks 5 r .. 5 r + 4); the same variable as step 1.

SETTINGS (default: my model at interior 1, 4, 16, 100 x water, and Denis' pipeline)
  mine   my 2D resistive field model (compute_electric_field.py): N_GRID x N_GRID grid over the 912 x 912 px frame, non-conductive
         tank walls (TANK_OUTLINE), electrodes ELECTRODES; the other two fish as bodies from that stage's masks of the EOD's own
         frame: interior sigma x water, skin as resistive as SKIN_LAMBDA_CM of water (conductivities relative to water = 1); the
         emitter as unit point currents along its midline, sink at the tail tip. One sparse solve per fish and frame.
         model_raw = unit-current pair voltages V(+) - V(-), raw sign (model_sign -1); recorded = main phase (each pair's 25-sample
         trace demeaned, all pairs at the sample of the largest |V|), V
  denis  Denis' assign_eods (EODAssignment repo, DENIS_REPO; numba) with the midlines as his point currents (identity = letters,
         no refractory step), his electrodes DENIS_ELECTRODES; needs all 3 midlines, else no prediction.
         model_raw = his predicted_PTP (model_sign +1); recorded = his recorded_PTP; also his similarity and own pick

WHERE IT SAVES  results/eod_assignment_3_fish_silent/mask_stages/predictions/<setting>/<rec>/<NN_stage>.npz,
<setting> = mine_int1, mine_int4, mine_int16, mine_int100 or denis:
  eod_rows (n,: every EOD row of the table), frames, recorded (n, 10), model_raw (n, 3, 10), model_sign, has_midline (n, 3),
  nearer_pole (n, 3, 10; 0 = + electrode of the pair nearer to the fish's midline, 1 = -), letters, hyperparameters (json);
  denis only: similarity (n, 3), pick_similarity (n,)

USAGE
  python cluster_3fish_stage_predict.py --check | --list | <task>
  MAX_FRAMES=100 OUT_ROOT=results/test python cluster_3fish_stage_predict.py 151      quick test (14_12_25, 01_segment, mine_int4)
  sbatch cluster_3fish_stage_predict.slurm
"""
import os, sys, re, json, time, shutil
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = os.environ.get('PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_3fish_common as C

# ================================================================================================================ hyperparameters
INTERIOR_SIGMAS = [1.0, 4.0, 16.0, 100.0]   # my model: one setting per interior conductivity of the other fish (x water)
INCLUDE_DENIS = True                        # also Denis' analytic pipeline as a setting
SETTINGS = [('mine', s) for s in INTERIOR_SIGMAS] + ([('denis', None)] if INCLUDE_DENIS else [])

# my model
N_GRID = 150                  # grid nodes per side over the 912 x 912 px video frame
SKIN_LAMBDA_CM = 0.1          # skin as resistive as this many cm of water (skin thickness SKIN_CM = 0.005 cm)
SINK_FRAC = 0.0               # current sink along the midline (0 = tail tip)
PX_PER_CM = 14.2              # model scale (compute_electric_field.PX_PER_CM)
MINE_SIGN = -1                # constant sign of the unit-current model against the recorded main phase
N_SAMPLES = 25                # samples per EOD trace in the EOD table

# Denis' pipeline
DENIS_REPO = os.environ.get('DENIS_REPO', os.path.join(os.path.dirname(ROOT), 'EODAssignment'))

# jobs
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))    # mine: worker processes; denis: numba threads
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 0))            # tests only: EODs on the first MAX_FRAMES frames (0 = all)
OVERWRITE = os.environ.get('OVERWRITE', '1') == '1'          # 0: skip tasks whose output exists
OUT_ROOT = os.environ.get('OUT_ROOT', C.OUT)

STAGES = C.stage_list()
JOBS = pd.DataFrame([(v, C.stage_label(s), s, m, sg) for v in C.VIDEOS for s in STAGES for m, sg in SETTINGS],
                    columns=['video', 'label', 'stage', 'model', 'sigma'])
JOBS['setting'] = [C.setting_tag(m, s) for m, s in zip(JOBS.model, JOBS.sigma)]
out_path = lambda r: C.PREDICTIONS.replace(C.OUT, OUT_ROOT).format(setting=r.setting, video=r.video, stage=r.label)
mid_path = lambda video, label: C.MIDLINES.replace(C.OUT, OUT_ROOT).format(video=video, stage=label)


def check():
    T = JOBS.assign(eod_table=[os.path.exists(C.EOD_TABLE.format(video=v)) for v in JOBS.video],
                    midlines=[os.path.exists(mid_path(v, l)) for v, l in zip(JOBS.video, JOBS.label)],
                    done=[os.path.exists(out_path(r)) for r in JOBS.itertuples()])
    print(f'settings {[t for t in dict.fromkeys(JOBS.setting)]}; output {OUT_ROOT}/predictions; numpy {np.__version__}')
    print(T.groupby('video')[['eod_table', 'midlines', 'done']].sum().to_string())
    runnable = T[T.eod_table]
    print(f'\n{len(runnable)} tasks with an EOD table, {runnable.midlines.sum()} with their midlines (step 1), {T.done.sum()} done')
    for f in (C.TANK_OUTLINE, C.ELECTRODES, C.DENIS_ELECTRODES):
        if not os.path.exists(f): print('MISSING', f)
    if INCLUDE_DENIS:
        if not os.path.exists(os.path.join(DENIS_REPO, 'analysis/notebooks/silent_fish.py')): print('MISSING Denis repo', DENIS_REPO)
        try: import numba  # noqa: F401
        except Exception as e: print('PROBLEM numba (Denis):', repr(e))


if len(sys.argv) > 1 and sys.argv[1] == '--list': print(JOBS.drop(columns='stage').to_string()); sys.exit()
if len(sys.argv) > 1 and sys.argv[1] == '--check': check(); sys.exit()

idx = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else int(sys.argv[1])
J = JOBS.loc[idx]; video, label, stage, MODEL, SIGMA = J.video, J.label, J.stage, J.model, J.sigma
path = out_path(J)
HYPER = dict(model=MODEL, setting=J.setting, video=video, stage=label, task=int(idx), MAX_FRAMES=MAX_FRAMES, N_MIDLINE=C.N_MIDLINE,
             **(dict(INTERIOR_SIGMA=SIGMA, N_GRID=N_GRID, SKIN_LAMBDA_CM=SKIN_LAMBDA_CM, SINK_FRAC=SINK_FRAC, PX_PER_CM=PX_PER_CM,
                     TANK_OUTLINE=C.TANK_OUTLINE, ELECTRODES=C.ELECTRODES, MODEL_SIGN=MINE_SIGN) if MODEL == 'mine'
                else dict(DENIS_REPO=DENIS_REPO, ELECTRODES=C.DENIS_ELECTRODES, MODEL_SIGN=1)))


# ===================================================================================================================== my model
def mine_solve_frame(job):
    from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, build_A_shell, solve_potential,
                                        unflatten_to_grid, confine_sources)
    from eod_assign import pair_voltages
    k, M = job; fm = np.asarray(CTX['masks'][k]); u = np.full((3, 10), np.nan)
    for f in range(3):
        if not np.isfinite(M[f]).all(): continue
        b = confine_sources(build_multipole_b(*multipole_sources(M[f], I0=1.0, sink_frac=SINK_FRAC), 1.0 / (N_GRID - 1), N_GRID), CTX['inside'])
        A = build_A_shell(occupancy_field(fm, N_GRID, skip=f), N_GRID, **CTX['mat'], inside=CTX['inside'])
        u[f] = pair_voltages(unflatten_to_grid(solve_potential(A, b), N_GRID).real, CTX['pairs'], N_GRID)
    return k, u


def run_mine(rows, frames, mids):
    global CTX
    from io_utils import load_stage_masks
    from compute_electric_field import tank_inside, materials, SKIN_CM
    from eod_assign import load_electrode_pairs, recorded_pair_traces_all
    eods = pd.read_csv(C.EOD_TABLE.format(video=video), usecols=[f'electrode_{p}_time_{i}' for p in range(1, 11) for i in range(1, N_SAMPLES + 1)])
    V = recorded_pair_traces_all(eods.iloc[rows], list(range(1, 11)), N_SAMPLES); del eods
    recorded = V[np.arange(len(V)), :, np.abs(V).max(1).argmax(1)]; del V                     # (n, 10) main phase, V
    pairs = load_electrode_pairs(C.ELECTRODES, C.P); assert [p for p, _, _ in pairs] == list(range(1, 11))
    CTX = dict(masks=load_stage_masks(C.MASK_RUN.format(video=video), stage), pairs=pairs,
               inside=tank_inside(pd.read_csv(C.TANK_OUTLINE)[['x', 'y']].to_numpy(), N_GRID, C.P),
               mat=dict(materials(N_GRID, C.P, PX_PER_CM), g_interior=SIGMA, g_skin=1 / (SKIN_LAMBDA_CM / SKIN_CM + 1 / SIGMA)))
    has = np.isfinite(mids).all(axis=(2, 3)).any(1)                                           # some fish has a midline
    uniq, first = np.unique(frames[has], return_index=True)
    work = [(int(k), mids[has][i]) for k, i in zip(uniq, first)]
    print(f'  my model, interior {SIGMA:g}: {len(work)} frames to solve, {N_WORKERS} worker(s)', flush=True)
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
    model_raw = np.stack([U.get(int(k), np.full((3, 10), np.nan)) for k in frames])
    return model_raw, recorded, MINE_SIGN, C.ELECTRODES, {}


# ===================================================================================================================== Denis
def run_denis(rows, frames, mids):
    os.environ.setdefault('NUMBA_NUM_THREADS', str(N_WORKERS))
    sys.path[:0] = [os.path.join(DENIS_REPO, 'analysis', 'notebooks'), DENIS_REPO]
    import silent_fish as sf
    keep = lambda c: c in ('Frame', 'Time') or c.startswith('electrode_') or any(c.startswith(p + '_') for p in sf.TRACKED_POINTS)
    src = pd.read_csv(C.EOD_TABLE.format(video=video), usecols=keep).iloc[rows].reset_index(drop=True)
    rename = {f'{p}_score_{i}': f'score_{p}_{i}' for p in sf.TRACKED_POINTS for i in range(1, 4)}
    raw = pd.concat([src[['Frame', 'Time']], src[[f'{p}_{k}_{i}' for i in range(1, 4) for p in sf.TRACKED_POINTS for k in ('x', 'y', 'score')]]
                     .rename(columns=rename), src.filter(like='electrode_')], axis=1); del src
    has = np.isfinite(mids).all(axis=(1, 2, 3))                                              # his pipeline needs all 3 fish
    cm = np.full(mids.shape, np.nan)
    for i in np.flatnonzero(has):
        for f in range(3): cm[i, f] = sf.resample(mids[i, f][:, ::-1] * (C.P - 1), C.N_MIDLINE) * sf.PX_TO_CM   # (x, y) cm
    work = Path(f'{OUT_ROOT}/predictions/denis_work/task{idx}'); work.mkdir(parents=True, exist_ok=True)
    csv = work / f'Data_3fish_{video}.csv'                         # his loader reads the date from the path: exactly one allowed
    assert len(re.findall(r'\d{4}-\d{2}-\d{2}', str(csv.resolve()))) == 1, f'the path {csv.resolve()} must contain exactly one date'
    sf.midline_input(raw[has], cm[has]).to_csv(csv, index=False)
    print(f'  Denis: {has.sum()} / {len(raw)} EODs with all 3 midlines, {os.environ["NUMBA_NUM_THREADS"]} numba thread(s)', flush=True)
    d = sf.run_pipeline(csv, Path(C.DENIS_ELECTRODES).resolve(), 3, midlines_cm=cm[has])
    assert (d.Frame.to_numpy() == raw.Frame.to_numpy()[has]).all(), 'Denis output rows do not match the input EODs'
    shutil.rmtree(work)
    n = len(raw); model_raw = np.full((n, 3, 10), np.nan); recorded = np.full((n, 10), np.nan); sim = np.full((n, 3), np.nan)
    if has.any():
        model_raw[has] = np.stack([np.stack(d[f'predicted_PTP_{f}'].to_numpy()) for f in C.LETTERS], axis=1)
        recorded[has] = np.stack(d['recorded_PTP'].to_numpy())
        sim[has] = np.stack([d[f'similarity_{f}'].to_numpy(float) for f in C.LETTERS], axis=1)
    pick_sim = np.full(n, '', object); pick_sim[has] = d.discharging_fish.to_numpy()
    return model_raw, recorded, 1, C.DENIS_ELECTRODES, dict(similarity=sim, pick_similarity=pick_sim.astype(str))


# ===================================================================================================================== run
t0 = time.time()
print(f'task {idx}: {video} {label}, setting {J.setting}, {N_WORKERS} cpu(s); output {path}', flush=True)
print('hyperparameters: ' + json.dumps(HYPER), flush=True)
if not OVERWRITE and os.path.exists(path): print('  already done, skipped (OVERWRITE=0)'); sys.exit(0)
frames_eod = C.eod_frames(video)
if frames_eod is None: print(f'  no EOD table {C.EOD_TABLE.format(video=video)}: skipped'); sys.exit(0)
if not os.path.exists(mid_path(video, label)): raise FileNotFoundError(f'{mid_path(video, label)}: run cluster_3fish_stage_midlines.py first')
rows = np.arange(len(frames_eod))
if MAX_FRAMES: rows = rows[np.isin(frames_eod, np.unique(frames_eod)[:MAX_FRAMES])]
frames = frames_eod[rows]
zm = np.load(mid_path(video, label)); mf = zm['frames']
j = np.clip(np.searchsorted(mf, frames), 0, len(mf) - 1); found = mf[j] == frames
mids = np.full((len(frames), 3, C.N_MIDLINE, 2), np.nan); mids[found] = zm['mids'][j[found]]; del zm
has_midline = np.isfinite(mids).all(axis=(2, 3))
print(f'  {len(rows)} EODs; midlines per fish ' + ' '.join(f'{f} {has_midline[:, i].sum()}' for i, f in enumerate(C.LETTERS)), flush=True)

model_raw, recorded, sign, el_file, extra = (run_mine if MODEL == 'mine' else run_denis)(rows, frames, mids)
pole = C.nearer_pole(mids, *C.poles(el_file))
C.save_npz(path, eod_rows=rows, frames=frames, recorded=recorded, model_raw=model_raw, model_sign=sign, has_midline=has_midline,
           nearer_pole=pole, letters=np.array(list(C.LETTERS)), hyperparameters=json.dumps(HYPER), **extra)
print(f'done: {path}, {len(rows)} EODs, {np.isfinite(model_raw).all(2).any(1).sum()} with a prediction; '
      f'{(time.time() - t0) / 60:.1f} min', flush=True)
