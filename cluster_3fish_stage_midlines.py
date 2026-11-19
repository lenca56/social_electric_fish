"""Cluster job, step 1 of 3 (see cluster_3fish_common.py): mask midline of every fish in every EOD frame of a 3-fish recording,
from the masks of one efish4-track stage. One SLURM array task = one (recording, stage): 8 recordings x 10 stages = 80 tasks
(0-79; tasks 10 r .. 10 r + 9 = recording r at stages 01_segment .. 10_disturbance).
MASK_STAGES=final: only the run's final masks, 8 tasks (recording r = task r); MASK_STAGES=segment,extent: those stages.

WHAT IT COMPUTES  per frame and fish: fish_midline_with_head (midline_methods.py) on that stage's visible mask, N_MIDLINE points,
tail -> head, (row, col) / (P - 1); NaN where the fish has no mask or the fit fails (the method of the 4-fish midlines).
FRAMES = 'eod' (default): only frames with an EOD (needs the recording's EOD table; without one the task is skipped);
'all': every video frame.

WHERE IT SAVES  results/eod_assignment_3_fish_silent/mask_stages/midlines/<rec>/<NN_stage>.npz:
  frames (n,), mids (n, 3, N_MIDLINE, 2), has_mask (n, 3), letters, hyperparameters (json)

USAGE
  python cluster_3fish_stage_midlines.py --check | --list | <task>
  MAX_FRAMES=200 OUT_ROOT=results/test python cluster_3fish_stage_midlines.py 30        quick test (14_12_25, 01_segment)
  sbatch cluster_3fish_stage_midlines.slurm
"""
import os, sys, json, time
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.environ.get('PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_3fish_common as C

# ================================================================================================================ hyperparameters
FRAMES = os.environ.get('FRAMES', 'eod')     # 'eod': frames with an EOD; 'all': every video frame
MASK_KIND = 'visible'                        # Kaden's 'visible' masks (as everywhere else) or 'amodal' (full body)
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 0))       # tests only: the first MAX_FRAMES frames (0 = all)
OVERWRITE = os.environ.get('OVERWRITE', '1') == '1'     # 0: skip tasks whose output exists
OUT_ROOT = os.environ.get('OUT_ROOT', C.OUT)            # output root (midlines/ below it)

STAGES = C.stage_list()
JOBS = pd.DataFrame([(v, C.stage_label(s), s) for v in C.VIDEOS for s in STAGES], columns=['video', 'label', 'stage'])
out_path = lambda video, label: C.MIDLINES.replace(C.OUT, OUT_ROOT).format(video=video, stage=label)


def check():
    rows = [dict(video=v, eod_table=os.path.exists(C.EOD_TABLE.format(video=v)),
                 stages_done=sum(os.path.exists(out_path(v, l)) for l in JOBS[JOBS.video == v].label)) for v in C.VIDEOS]
    print(f'frames {FRAMES}; output {OUT_ROOT}/midlines; numpy {np.__version__}\n' + pd.DataFrame(rows).to_string(index=False))
    from io_utils import stage_mask_file
    miss = [f'{v} {s}' for v in C.VIDEOS for s in STAGES if stage_mask_file(C.MASK_RUN.format(video=v), s) is None]
    print(f'\nmask stages missing: {len(miss)}' + ''.join(f'\n  {m}' for m in miss))
    if tuple(int(x) for x in np.__version__.split('.')[:2]) < (2, 4): print('PROBLEM: numpy < 2.4 cannot read the mask pickles')


if len(sys.argv) > 1 and sys.argv[1] == '--list': print(JOBS.to_string()); sys.exit()
if len(sys.argv) > 1 and sys.argv[1] == '--check': check(); sys.exit()

from io_utils import load_stage_masks
from midline_methods import fish_midline_with_head

idx = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else int(sys.argv[1])
video, label, stage = JOBS.loc[idx, ['video', 'label', 'stage']]
path = out_path(video, label)
HYPER = dict(FRAMES=FRAMES, MASK_KIND=MASK_KIND, N_MIDLINE=C.N_MIDLINE, MAX_FRAMES=MAX_FRAMES, video=video, stage=label, task=idx,
             method='midline_methods.fish_midline_with_head', units='(row, col) / (P - 1), tail -> head')


def fit_frame(k):
    fm = MASKS[k]; out = np.full((len(fm), C.N_MIDLINE, 2), np.nan); has = fm.reshape(len(fm), -1).any(1)
    for f in np.flatnonzero(has):
        try:
            mid = fish_midline_with_head(fm[f], n_ctrl=C.N_MIDLINE)['midline']
        except Exception:
            continue
        if mid is not None and np.isfinite(mid).all(): out[f] = mid
    return out, has


t0 = time.time()
print(f'task {idx}: {video} {label}, frames {FRAMES}, {N_WORKERS} worker(s); output {path}', flush=True)
if not OVERWRITE and os.path.exists(path): print('  already done, skipped (OVERWRITE=0)'); sys.exit(0)
frames_eod = C.eod_frames(video)
if FRAMES == 'eod' and frames_eod is None: print(f'  no EOD table {C.EOD_TABLE.format(video=video)}: skipped'); sys.exit(0)
MASKS = load_stage_masks(C.MASK_RUN.format(video=video), stage, kind=MASK_KIND)
if MASKS is None: raise FileNotFoundError(f'{video}: no {stage} masks in {C.MASK_RUN.format(video=video)}')
assert ''.join(MASKS.names) == C.LETTERS, f'{video} {stage}: fish {MASKS.names}, expected {tuple(C.LETTERS)}'
frames = np.arange(len(MASKS)) if FRAMES == 'all' else np.unique(frames_eod[(frames_eod >= 0) & (frames_eod < len(MASKS))])
if MAX_FRAMES: frames = frames[:MAX_FRAMES]
mids = np.full((len(frames), 3, C.N_MIDLINE, 2), np.nan); has_mask = np.zeros((len(frames), 3), bool)
if N_WORKERS > 1:
    with mp.get_context('fork').Pool(N_WORKERS) as pool:
        for j, (m, h) in enumerate(pool.imap(fit_frame, frames.tolist(), chunksize=64)): mids[j], has_mask[j] = m, h
else:
    for j, k in enumerate(frames.tolist()): mids[j], has_mask[j] = fit_frame(k)
C.save_npz(path, frames=frames, mids=mids, has_mask=has_mask, letters=np.array(list(C.LETTERS)), hyperparameters=json.dumps(HYPER))
ok = np.isfinite(mids).all(axis=(2, 3))
print(f'done: {path}, {len(frames)} frames; midline / mask per fish ' +
      ' '.join(f'{f} {ok[:, i].sum()}/{has_mask[:, i].sum()}' for i, f in enumerate(C.LETTERS)) +
      f'; all 3 fish in {ok.all(1).sum()} frames; {(time.time() - t0) / 60:.1f} min', flush=True)
