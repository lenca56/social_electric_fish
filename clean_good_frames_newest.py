"""Fully clean good frames for the *_synch_fix_newest.csv EOD tables, judged on the video frame each EOD has in the newest table
(that frame's masks are the ones the pipelines use):

1. mask QC: the frame is in good_frames.csv (EOD_assignment_<n>_fish_silent.ipynb; computed on every video frame);
2. every fish's mask principal-axis length >= 0.8 x that fish's median (EOD_QC_length_limit_<n>fish.ipynb).
clean_good = mask QC and length ok.

From 2026-11-10 SLEAP no longer counts: both pipelines (mine and Denis') run on the mask midlines, so the SLEAP-good rule
(silent_fish.sleap_good: mean keypoint score >= 0.8 and no missing keypoint on that frame) was dropped from clean_good. It is still
written as the sleap_good column, and clean_good_with_sleap (all three) is the previous definition; the files of that definition
were kept as *_with_sleap.csv. The same code is first run on the old EOD table and, with the SLEAP rule, must give
good_frames_better_list.csv exactly.

    OMP_NUM_THREADS=1 python clean_good_frames_newest.py [--workers 6] [--fresh]   # --fresh: recompute the mask lengths

Output per recording, in <res>/: mask_length_ratio_all_frames.csv (min length / median over the fish, per frame),
good_frames_better_list_synch_fix_newest.csv (frames) and eod_clean_good_synch_fix_newest.csv (one row per EOD of the newest
table: Frame, mask_qc, sleap_good, length_ok, clean_good, clean_good_with_sleap).
"""
import argparse, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import mask_metrics as mm
from io_utils import load_masks

DENIS_NB = Path('/Volumes/Lenca_SSD/github/EODAssignment/analysis/notebooks')
sys.path[:0] = [str(DENIS_NB), str(DENIS_NB.parents[1])]
TRACKED_POINTS = ['mouth', 'head', 'middle', 'tail']               # as silent_fish.TRACKED_POINTS
LENGTH_MIN = 0.8

REC = {
    '2-fish': dict(fish=(0, 1), n=2, res='results/eod_ground_truth',
                   track='data/2_fish_silent/elephantnose_2024-02-13T14-33-21_Shiv_tracking_final.pkl',
                   old='data/2_fish_silent/Data_1fish_2024-02-13T14_33_21.csv', old_layout='keypoints',
                   newest='data/2_fish_silent/Data_2fish_2024-02-13T14_33_21_synch_fix_newest.csv'),
    '3-fish': dict(fish=(0, 1, 2), n=3, res='results/eod_ground_truth_3fish',
                   track='data/3_fish_silent/elephantnose_2024-10-15T14-12-25_Shiv_tracking_final.pkl',
                   old='data/3_fish_silent/Assigned_Data_3fish_2024-10-15T14_12_25.csv', old_layout='letters',
                   newest='data/3_fish_silent/Data_3fish_2024-10-15T14_12_25_synch_fix_newest.csv'),
}
MASKS = None                                                        # ZEUS masks, set in the parent before the pool forks


def sleap_good(raw, num_fish, min_score=0.8):
    """silent_fish.sleap_good: per EOD, its frame has mean keypoint score >= min_score and no missing keypoint."""
    coords = [f'{p}_{a}_{i}' for p in TRACKED_POINTS for a in 'xy' for i in range(1, num_fish + 1)]
    scores = [f'score_{p}_{i}' for p in TRACKED_POINTS for i in range(1, num_fish + 1)]
    missing = raw[coords].isna().any(axis=1) | (raw[scores].fillna(0) == 0).any(axis=1)
    by_frame = raw['Frame']
    ok = (raw[scores].mean(axis=1).groupby(by_frame).transform('mean') >= min_score) & ~missing.groupby(by_frame).transform('any')
    return ok.to_numpy()


def keypoint_table(c, which):
    """Frame + keypoints/scores in Denis' input layout (<point>_<x|y>_<i>, score_<point>_<i>) for the old or newest table."""
    if which == 'newest':
        cols = ['Frame'] + [f'{p}_{k}_{i}' for i in range(1, c['n'] + 1) for p in TRACKED_POINTS for k in ('x', 'y', 'score')]
        d = pd.read_csv(c['newest'], usecols=cols)
        return d.rename(columns={f'{p}_score_{i}': f'score_{p}_{i}' for i in range(1, c['n'] + 1) for p in TRACKED_POINTS})
    if c['old_layout'] == 'keypoints':
        return pd.read_csv(c['old'], usecols=lambda col: col == 'Frame' or col.split('_')[0] in TRACKED_POINTS + ['score'])
    letters = [chr(65 + i) for i in range(c['n'])]
    rename = {f'{p}_{k}_{f}': (f'score_{p}_{i}' if k == 'score' else f'{p}_{k}_{i}')
              for i, f in enumerate(letters, 1) for p in TRACKED_POINTS for k in ('x', 'y', 'score')}
    return pd.read_csv(c['old'], usecols=['Frame'] + list(rename)).rename(columns=rename)


def majors(keys):
    return [(k, MASKS.fish_order[f], mm.shape_metrics(crop, box)['major']) for k in keys for f, (box, crop) in MASKS.crops(k).items()]


def length_ratio(c, workers, fresh=False):
    """Per frame: min over the fish of (principal-axis length / that fish's median over all frames), as in the length-limit QC."""
    path = Path(c['res']) / 'mask_length_ratio_all_frames.csv'
    if path.exists() and not fresh: return pd.read_csv(path, index_col='frame')['min_length_ratio']
    global MASKS
    MASKS = load_masks(c['track'], 'shiv', fish_order=c['fish']); keys = list(range(len(MASKS)))
    with mp.get_context('fork').Pool(workers) as pool:
        parts = pool.map(majors, np.array_split(keys, 8 * workers))
    q = pd.DataFrame([r for p in parts for r in p], columns=['frame', 'fish', 'major'])
    q['length_ratio'] = q.major / q.groupby('fish').major.transform('median')
    L = q.pivot_table(index='frame', columns='fish', values='length_ratio', aggfunc='min')
    s = L.reindex(columns=list(c['fish'])).min(axis=1, skipna=False).rename('min_length_ratio')   # NaN if a fish has no mask
    s.rename_axis('frame').to_csv(path); MASKS = None
    return s


def clean_flags(c, tab, L):
    good_mask = set(pd.read_csv(Path(c['res']) / 'good_frames.csv').frame)
    f = tab.Frame.to_numpy()
    return pd.DataFrame({'Frame': f, 'mask_qc': np.isin(f, list(good_mask)), 'sleap_good': sleap_good(tab, c['n']),
                         'length_ok': (L.reindex(f).to_numpy() >= LENGTH_MIN - 1e-9)}).assign(
        clean_good=lambda d: d.mask_qc & d.length_ok, clean_good_with_sleap=lambda d: d.mask_qc & d.sleap_good & d.length_ok)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--fresh', action='store_true'); a = ap.parse_args()
    for name, c in REC.items():
        t0 = time.time(); L = length_ratio(c, a.workers, a.fresh)
        old = clean_flags(c, keypoint_table(c, 'old'), L)
        better = set(pd.read_csv(Path(c['res']) / 'good_frames_better_list.csv').frame); got = set(old.Frame[old.clean_good_with_sleap])
        print(f'{name}: old table -> {len(got)} clean frames; good_frames_better_list.csv has {len(better)}; '
              f'only mine {len(got - better)}, only the list {len(better - got)}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
        new = clean_flags(c, keypoint_table(c, 'newest'), L)
        new.rename_axis('eod_row').to_csv(Path(c['res']) / 'eod_clean_good_synch_fix_newest.csv')
        frames = np.sort(new.Frame[new.clean_good].unique())
        pd.Series(frames, name='frame').to_csv(Path(c['res']) / 'good_frames_better_list_synch_fix_newest.csv', index=False)
        print(f'{name}: newest table {len(new)} EODs -> mask QC {new.mask_qc.sum()}, length ok {new.length_ok.sum()}, clean good '
              f'(both) {new.clean_good.sum()} EODs on {len(frames)} frames; with the old SLEAP rule {new.clean_good_with_sleap.sum()}',
              flush=True)
