"""Mask QC (as in EOD_assignment_<n>_fish_silent.ipynb, with the 0.8 length limit of the better list) on Kaden's visible masks
of the 4-fish recordings, for every EOD frame and its neighbours.

Per fish and frame, a mask passes when:
- the fish has a mask (Kaden's tracker gives each letter at most one mask, so no duplicates or remnants are possible);
- its area is within 20% of that fish's median area;
- its principal-axis length is 0.8-1.2 x that fish's median (lower limit 0.8 as in good_frames_better_list.csv);
- >= 99% of the mask is one connected piece;
- its centroid moves <= 15 px and its area changes <= 15% to the previous AND the next frame;
- it is not within 5 px of any other fish's mask.
A frame is good when all four fish pass. Medians are per video, over the frames computed here (EOD frames +- 1).

    OMP_NUM_THREADS=1 python kaden_mask_qc.py --rec 05_52_55 [--workers 7]

Output: results/eod_assign_4fish_kaden/<rec>/mask_qc_per_fish.csv (frame, fish, metrics, checks) and mask_qc_good_frames.csv.
"""
import argparse, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import mask_metrics as mm
from io_utils import load_masks

KADEN = 'data/track_Kaden/results_{t}_full/video2024-02-20T{t}/outputs/video2024-02-20T{t}.pkl'
EOD = 'data/4 fish/Data_4fish_2024-02-20T{t}.csv'
OUT = Path('results/eod_assign_4fish_kaden')
LET = 'ABCD'
AREA_TOL, LENGTH_RANGE, MIN_LARGEST_FRAC, MAX_STEP_PX, MAX_DAREA, CONTACT_GAP = 0.20, (0.8, 1.2), 0.99, 15, 0.15, 5
MASKS = None                                                        # Kaden's visible masks, set before the pool forks


def in_contact(a, b, gap=CONTACT_GAP):
    '''True if two (box, crop) masks come within `gap` px (8-neighbour steps) of each other (as in the 3-fish notebook).'''
    (ba, ca), (bb, cb) = a, b
    if max(ba[0] - bb[1], bb[0] - ba[1], ba[2] - bb[3], bb[2] - ba[3]) > gap: return False
    y0, x0 = min(ba[0], bb[0]) - gap, min(ba[2], bb[2]) - gap
    win = np.zeros((2, max(ba[1], bb[1]) + gap - y0, max(ba[3], bb[3]) + gap - x0), bool)
    for w, ((r0, r1, c0, c1), c) in zip(win, (a, b)): w[r0 - y0:r1 - y0, c0 - x0:c1 - x0] = c
    return bool((binary_dilation(win[0], np.ones((3, 3)), iterations=gap) & win[1]).any())


def frame_rows(keys):
    rows = []
    for k in keys:
        crops = MASKS.crops(k)
        for f, (box, crop) in crops.items():
            rows.append(dict(frame=int(k), fish=LET[f], contact=any(in_contact(crops[f], crops[o]) for o in crops if o != f),
                             **mm.shape_metrics(crop, box)))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True); ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args(); t = a.rec; t0 = time.time()
    from kaden_4fish_assign import kaden_pkl                         # also finds the 2024-02-20T<t>_R2-*.pkl name of 5 recordings
    MASKS = load_masks(kaden_pkl(t)); assert ''.join(MASKS.names) == LET; n = len(MASKS)
    ef = np.unique(pd.read_csv(EOD.format(t=t), usecols=['Frame']).Frame.to_numpy())
    keys = np.unique(np.r_[ef - 1, ef, ef + 1]); keys = keys[(keys >= 0) & (keys < n)]
    with mp.get_context('fork').Pool(a.workers) as pool:
        parts = pool.map(frame_rows, np.array_split(keys, 16 * a.workers))
    q = pd.DataFrame([r for p in parts for r in p])

    # every computed frame x fish, NaN where a fish has no mask
    full = pd.MultiIndex.from_product([keys, list(LET)], names=['frame', 'fish'])
    q = q.set_index(['frame', 'fish']).reindex(full).reset_index()
    q['present'] = q.px.fillna(0) > 0
    for f in LET:                                                   # worst of the step to the previous and to the next frame
        s = q[q.fish == f].set_index('frame').reindex(np.arange(keys.min() - 1, keys.max() + 2))
        step = np.hypot(s.cx.diff(), s.cy.diff()); darea = (s.px / s.px.shift(1) - 1).abs()
        m = q.fish == f
        q.loc[m, 'step'] = np.fmax(step, step.shift(-1)).fillna(np.inf).loc[q.frame[m]].to_numpy()
        q.loc[m, 'darea'] = np.fmax(darea, darea.shift(-1)).fillna(np.inf).loc[q.frame[m]].to_numpy()
    med = q[q.present].groupby('fish')[['px', 'major']].median()
    q['area_ratio'] = q.px / q.fish.map(med.px); q['length_ratio'] = q.major / q.fish.map(med.major)
    checks = {
        'has a mask': q.present,
        f'area within {AREA_TOL:.0%} of median': (q.area_ratio - 1).abs() <= AREA_TOL,
        f'length {LENGTH_RANGE[0]}-{LENGTH_RANGE[1]} x median': q.length_ratio.between(*LENGTH_RANGE),
        f'>= {MIN_LARGEST_FRAC:.0%} in one piece': q.largest_frac >= MIN_LARGEST_FRAC,
        f'step <= {MAX_STEP_PX} px': q.step <= MAX_STEP_PX,
        f'area change <= {MAX_DAREA:.0%}': q.darea <= MAX_DAREA,
        'not touching other fish': ~q.contact.astype('boolean').fillna(True).to_numpy(bool),
    }
    for c, v in checks.items(): q[c] = np.asarray(v)
    q['all'] = pd.DataFrame(checks).all(axis=1).to_numpy()
    q.to_csv(OUT / t / 'mask_qc_per_fish.csv', index=False)
    good = q.groupby('frame')['all'].all()
    good = good.index[good.to_numpy()].to_numpy()
    good = good[np.isin(good, ef)]
    pd.Series(good, name='frame').to_csv(OUT / t / 'mask_qc_good_frames.csv', index=False)
    rates = (q[q.frame.isin(ef)].groupby('fish')[list(checks) + ['all']].mean().T * 100).round(1)
    print(f'{t}: {len(ef)} EOD frames, mask QC good (all four fish) {len(good)} ({len(good) / len(ef) * 100:.1f}%), '
          f'{(time.time() - t0) / 60:.1f} min\n{rates.to_string()}', flush=True)
