"""Cluster job: mask midline of every fish in every frame of the 71 4-fish recordings of 2024-02-20 (Kaden's masks).
One SLURM array task = one video: tasks 0-70 (the VIDEOS list below).

WHAT IT COMPUTES, for every frame (FRAMES = 'all': every video frame; 'eod': only frames with an EOD) and every fish A-D
  mids       the mask midline, tail -> head, N_MIDLINE points (row, col) / (P - 1), from fish_midline_with_head
             (midline_methods.py) on Kaden's visible mask of that fish in that frame; NaN where the fish has no mask or the
             fit fails (the same method as the midlines_by_frame.npz files of kaden_4fish_assign.py --stage midlines)
  has_mask   (n, 4) bool: the fish has a mask in that frame (so NaN midline + has_mask = a failed fit)

WHERE IT SAVES (folder created by the task): OUT_DIR/<video>/midlines_<FRAMES>_frames.npz, OUT_DIR =
results/eod_assign_4fish_kaden/cluster_midlines
  .npz  frames (n,), mids (n, 4, N_MIDLINE, 2), has_mask (n, 4), letters, hyperparameters (json string)
        frames and mids are named as in midlines_by_frame.npz, so readers of that file can read this one
Written to .tmp and renamed when complete. By default a task recomputes and replaces its file; OVERWRITE=0 skips videos
whose file exists (to resume an interrupted array).

FILES IT READS (relative to the repository root, the folder of this script)
  code     io_utils.py, midline_methods.py (needs numpy >= 2.4, pandas, scipy)
  masks    data/track_Kaden/results_<video>_full/video2024-02-20T<video>/outputs/video2024-02-20T<video>.pkl, or the
           2024-02-20T<video>_R2-*.pkl name of 5 recordings in the same folder
  EODs     data/4 fish/Data_4fish_2024-02-20T<video>.csv (only with FRAMES = 'eod': the Frame column)

USAGE
  python cluster_kaden_4fish_midlines.py --check     every input present and readable for all 71 videos?
  python cluster_kaden_4fish_midlines.py --list      the job table (task index -> video)
  python cluster_kaden_4fish_midlines.py 17          run task 17 here (without SLURM)
  MAX_FRAMES=200 OUT_DIR=results/test_midlines python cluster_kaden_4fish_midlines.py 0     quick test on 200 frames
  sbatch cluster_kaden_4fish_midlines.slurm           all 71 tasks (task index from SLURM_ARRAY_TASK_ID)
"""
import os, sys, json, glob, time
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.environ.get('PROJECT_DIR', os.path.dirname(os.path.abspath(__file__)))   # repository root: data/ and results/
os.chdir(ROOT); sys.path.insert(0, ROOT)

# ================================================================================================================ hyperparameters
# midline
N_MIDLINE = 101               # midline points (tail -> head)
MASK_KIND = 'visible'         # Kaden's 'visible' masks (as everywhere else) or 'amodal' (full body)
P = 912                       # video frame size, px
FRAMES = os.environ.get('FRAMES', 'all')    # 'all': every video frame; 'eod': only the frames with an EOD

# data
EOD_TABLE = 'data/4 fish/Data_4fish_2024-02-20T{video}.csv'
KADEN_PKL = 'data/track_Kaden/results_{video}_full/video2024-02-20T{video}/outputs/video2024-02-20T{video}.pkl'
LETTERS = 'ABCD'

# output
OUT_DIR = os.environ.get('OUT_DIR', 'results/eod_assign_4fish_kaden/cluster_midlines')

# jobs
N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', 0))     # tests only: just the first MAX_FRAMES frames (0 = all)
OVERWRITE = os.environ.get('OVERWRITE', '1') == '1'   # 1 (default): recompute and replace existing outputs; 0: skip finished videos

# the 71 recordings, fixed so a task index always means the same video (whatever is on disk)
VIDEOS = ['00_12_52', '00_32_53', '00_52_53', '01_12_53', '01_32_53', '01_52_53', '02_12_53', '02_32_53', '02_52_54',
          '03_12_54', '03_32_54', '03_52_54', '04_12_54', '04_32_54', '04_52_54', '05_12_55', '05_32_55', '05_52_55',
          '06_12_55', '06_32_55', '06_52_55', '07_12_56', '07_32_56', '07_52_56', '08_12_56', '08_32_56', '08_52_56',
          '09_12_56', '09_32_57', '09_52_57', '10_12_57', '10_32_57', '10_52_57', '11_12_57', '11_32_57', '11_52_58',
          '12_12_58', '12_32_58', '12_56_22', '13_16_22', '13_36_22', '13_56_22', '14_16_23', '14_36_23', '14_56_23',
          '15_16_23', '15_36_23', '15_56_23', '16_16_23', '16_36_24', '16_56_24', '17_16_24', '17_36_24', '17_56_24',
          '18_16_24', '18_36_24', '18_56_25', '19_16_25', '19_36_25', '19_56_25', '20_16_25', '20_36_25', '21_16_26',
          '21_36_26', '21_56_26', '22_16_26', '22_36_26', '22_56_26', '23_16_27', '23_36_27', '23_56_27']
NUMPY_VERSION = tuple(int(x) for x in np.__version__.split('.')[:2])


# ===================================================================================================================== helpers
def kaden_pkl(video):
    p = KADEN_PKL.format(video=video)
    if os.path.exists(p): return p
    alt = sorted(glob.glob(os.path.join(os.path.dirname(p), f'2024-02-20T{video}_*.pkl')))
    if len(alt) != 1: raise FileNotFoundError(f'{video}: no Kaden pickle {p} (nor a unique 2024-02-20T{video}_*.pkl beside it)')
    return alt[0]


def out_path(video):
    return f'{OUT_DIR}/{video}/midlines_{FRAMES}_frames.npz'


def check_inputs():
    '''Every input of every task present? Prints a table and the missing files; exit code 1 if anything required is missing.'''
    bad = []
    if NUMPY_VERSION < (2, 4): bad.append(f'numpy {np.__version__} (Kaden pickles need numpy >= 2.4, from conda-forge)')
    if FRAMES not in ('all', 'eod'): bad.append(f"FRAMES = {FRAMES!r}: must be 'all' or 'eod'")
    rows = []
    for v in VIDEOS:
        try: pk = kaden_pkl(v)
        except FileNotFoundError: pk = None
        rows.append(dict(video=v, kaden_pkl=pk is not None, eod_table=os.path.exists(EOD_TABLE.format(video=v)),
                         done=os.path.exists(out_path(v))))
        if pk is None: bad.append(KADEN_PKL.format(video=v))
        if FRAMES == 'eod' and not rows[-1]['eod_table']: bad.append(EOD_TABLE.format(video=v))
    T = pd.DataFrame(rows)
    print(f'repository root: {ROOT}\noutput folder:   {OUT_DIR}\nframes:          {FRAMES}\nnumpy {np.__version__}, pandas {pd.__version__}')
    print(T.to_string(index=False))
    print(f'\n{T.kaden_pkl.sum()}/71 Kaden pickles, {T.eod_table.sum()}/71 EOD tables; videos done {T.done.sum()}/{len(VIDEOS)}')
    try:
        import io_utils, midline_methods, scipy   # noqa: F401
    except Exception as e:
        bad.append(f'import failed: {e!r}')
    if bad:
        print('\nMISSING / PROBLEMS:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print('\nall required inputs present')


if len(sys.argv) > 1 and sys.argv[1] == '--list':
    print(pd.DataFrame({'video': VIDEOS}).to_string()); sys.exit()
if len(sys.argv) > 1 and sys.argv[1] == '--check':
    check_inputs(); sys.exit()

from io_utils import load_masks
from midline_methods import fish_midline_with_head

idx = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ else int(sys.argv[1])
video = VIDEOS[idx]
path = out_path(video)
HYPER = dict(N_MIDLINE=N_MIDLINE, MASK_KIND=MASK_KIND, P=P, FRAMES=FRAMES, MAX_FRAMES=MAX_FRAMES, video=video, task=idx,
             method='midline_methods.fish_midline_with_head', units='(row, col) / (P - 1), tail -> head')


def fit_frame(k):
    '''(4, N_MIDLINE, 2) midlines of frame k (NaN where no mask or the fit fails) and (4,) has_mask.'''
    fm = MASKS[k]; out = np.full((len(fm), N_MIDLINE, 2), np.nan); has = fm.reshape(len(fm), -1).any(1)
    for f in np.flatnonzero(has):
        try:
            mid = fish_midline_with_head(fm[f], n_ctrl=N_MIDLINE)['midline']
        except Exception:
            continue
        if mid is not None and np.isfinite(mid).all(): out[f] = mid
    return out, has


# ===================================================================================================================== run
t0 = time.time()
print(f'task {idx}: video {video}, frames {FRAMES}, {N_WORKERS} worker(s); root {ROOT}; output {path}', flush=True)
print('hyperparameters: ' + json.dumps(HYPER), flush=True)
if not OVERWRITE and os.path.exists(path):
    print('  already done, skipped (OVERWRITE=0)'); sys.exit(0)
assert NUMPY_VERSION >= (2, 4), f'numpy {np.__version__}: Kaden pickles need numpy >= 2.4 (install it from conda-forge)'
assert FRAMES in ('all', 'eod'), f"FRAMES = {FRAMES!r}: must be 'all' or 'eod'"

MASKS = load_masks(kaden_pkl(video), kind=MASK_KIND)
assert MASKS.n_fish == 4 and ''.join(MASKS.names) == LETTERS, f'{video}: fish {MASKS.names}, expected {tuple(LETTERS)}'
if FRAMES == 'all':
    frames = np.arange(len(MASKS))
else:
    frames = np.unique(pd.read_csv(EOD_TABLE.format(video=video), usecols=['Frame']).Frame.to_numpy())
    frames = frames[(frames >= 0) & (frames < len(MASKS))]
if MAX_FRAMES: frames = frames[:MAX_FRAMES]
print(f'  {len(frames)} frames (video has {len(MASKS)}); setup {time.time() - t0:.0f} s', flush=True)

mids = np.full((len(frames), 4, N_MIDLINE, 2), np.nan); has_mask = np.zeros((len(frames), 4), bool)
work = [int(k) for k in frames]
def results():
    if N_WORKERS > 1:
        with mp.get_context('fork').Pool(N_WORKERS) as pool:
            yield from pool.imap(fit_frame, work, chunksize=64)               # in frame order
    else:
        yield from map(fit_frame, work)
for j, (m, h) in enumerate(results()):
    mids[j], has_mask[j] = m, h
    if (j + 1) % 5000 == 0 or j + 1 == len(work):
        el = time.time() - t0
        print(f'  {j + 1}/{len(work)} frames, {el / 60:.1f} min, ~{el / (j + 1) * (len(work) - j - 1) / 60:.0f} min left', flush=True)

os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path + '.tmp', 'wb') as fh:                                         # a file handle: numpy adds no '.npz' to the .tmp name
    np.savez_compressed(fh, frames=frames, mids=mids, has_mask=has_mask, letters=np.array(list(LETTERS)),
                        hyperparameters=json.dumps(HYPER))
os.replace(path + '.tmp', path)
ok = np.isfinite(mids).all(axis=(2, 3))
print(f'done: {path}, {len(frames)} frames; midline / mask per fish ' +
      ' '.join(f'{f} {ok[:, i].sum()}/{has_mask[:, i].sum()}' for i, f in enumerate(LETTERS)) +
      f'; all 4 fish in {ok.all(1).sum()} frames; {(time.time() - t0) / 60:.1f} min', flush=True)
