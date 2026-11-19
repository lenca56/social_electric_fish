"""Mask QC of Kaden's efish4-track masks (visible masks) for the 2- and 3-fish silent controls: the rules of kaden_mask_qc.py
(4-fish), which are those of EOD_assignment_<n>_fish_silent.ipynb with the 0.8 length limit of the clean good list.

Per fish and frame, a mask passes when:
- the fish has a mask (one mask per letter, no remnants possible);
- its area is within 20% of that fish's median area;
- its principal-axis length is 0.8-1.2 x that fish's median;
- >= 99% of the mask is one connected piece;
- its centroid moves <= 15 px and its area changes <= 15% to the previous AND the next frame;
- it is not within 5 px of any other fish's mask.
A frame is good when every fish passes. Medians are per video, over the frames computed here (EOD frames +- 1). No SLEAP rule.

    OMP_NUM_THREADS=1 python kaden_mask_qc_controls.py [--workers 7] [--masks kaden|latest]
--masks latest: Kaden's newer run (*_latest_run folders), output in with_latest_run_tracking/ instead.

Output in results/eod_assignment_<n>_fish_silent/with_Kaden_tracking/: mask_qc_per_fish.csv (frame, letter, fish = ZEUS number,
metrics, checks), mask_qc_good_frames.csv, and eod_mask_qc.csv (one row per EOD of the newest table: eod_row, Frame, Time,
in_video = Time >= 0, mask_qc_good).
"""
import argparse, time
import multiprocessing as mp
import numpy as np
import pandas as pd
import mask_metrics as mm
from io_utils import load_masks
import kaden_mask_qc as kq                                          # in_contact and the thresholds of the 4-fish QC
import eod_assignment_control_groups as g

MASKS, LET = None, None                                             # set before the pool forks


def frame_rows(keys):
    rows = []
    for k in keys:
        crops = MASKS.crops(k)
        for f, (box, crop) in crops.items():
            rows.append(dict(frame=int(k), letter=LET[f], contact=any(kq.in_contact(crops[f], crops[o]) for o in crops if o != f),
                             **mm.shape_metrics(crop, box)))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=7); ap.add_argument('--masks', choices=['kaden', 'latest'], default='kaden')
    a = ap.parse_args()
    for rec in ('2-fish', '3-fish'):
        c = g.REC[rec]; t0 = time.time()
        MASKS = load_masks(g.mask_file(rec, a.masks)); OUT = g.out_dir(rec, a.masks); LET = ''.join(MASKS.names); n = len(MASKS)
        to_fish = {LET[j]: f for f, j in enumerate(c['kaden_order'])}
        eod = pd.read_csv(c['newest'], usecols=['Frame', 'Time']); ef = np.unique(eod.Frame.to_numpy())
        keys = np.unique(np.r_[ef - 1, ef, ef + 1]); keys = keys[(keys >= 0) & (keys < n)]
        with mp.get_context('fork').Pool(a.workers) as pool:
            parts = pool.map(frame_rows, np.array_split(keys, 16 * a.workers))
        q = pd.DataFrame([r for p in parts for r in p])
        full = pd.MultiIndex.from_product([keys, list(LET)], names=['frame', 'letter'])
        q = q.set_index(['frame', 'letter']).reindex(full).reset_index()
        q['present'] = q.px.fillna(0) > 0
        for f in LET:
            s = q[q.letter == f].set_index('frame').reindex(np.arange(keys.min() - 1, keys.max() + 2))
            step = np.hypot(s.cx.diff(), s.cy.diff()); darea = (s.px / s.px.shift(1) - 1).abs(); m = q.letter == f
            q.loc[m, 'step'] = np.fmax(step, step.shift(-1)).fillna(np.inf).loc[q.frame[m]].to_numpy()
            q.loc[m, 'darea'] = np.fmax(darea, darea.shift(-1)).fillna(np.inf).loc[q.frame[m]].to_numpy()
        med = q[q.present].groupby('letter')[['px', 'major']].median()
        q['area_ratio'] = q.px / q.letter.map(med.px); q['length_ratio'] = q.major / q.letter.map(med.major)
        checks = {
            'has a mask': q.present,
            f'area within {kq.AREA_TOL:.0%} of median': (q.area_ratio - 1).abs() <= kq.AREA_TOL,
            f'length {kq.LENGTH_RANGE[0]}-{kq.LENGTH_RANGE[1]} x median': q.length_ratio.between(*kq.LENGTH_RANGE),
            f'>= {kq.MIN_LARGEST_FRAC:.0%} in one piece': q.largest_frac >= kq.MIN_LARGEST_FRAC,
            f'step <= {kq.MAX_STEP_PX} px': q.step <= kq.MAX_STEP_PX,
            f'area change <= {kq.MAX_DAREA:.0%}': q.darea <= kq.MAX_DAREA,
            'not touching other fish': ~q.contact.astype('boolean').fillna(True).to_numpy(bool),
        }
        for k_, v in checks.items(): q[k_] = np.asarray(v)
        q['all'] = pd.DataFrame(checks).all(axis=1).to_numpy(); q['fish'] = q.letter.map(to_fish)
        q.to_csv(f"{OUT}/mask_qc_per_fish.csv", index=False)
        good = q.groupby('frame')['all'].all(); good = good.index[good.to_numpy()].to_numpy(); good = good[np.isin(good, ef)]
        pd.Series(good, name='frame').to_csv(f"{OUT}/mask_qc_good_frames.csv", index=False)
        e = pd.DataFrame({'eod_row': np.arange(len(eod)), 'Frame': eod.Frame, 'Time': eod.Time, 'in_video': eod.Time >= 0})
        e['mask_qc_good'] = e.in_video & e.Frame.isin(good)
        e.to_csv(f"{OUT}/eod_mask_qc.csv", index=False)
        rates = (q[q.frame.isin(ef)].groupby('letter')[list(checks) + ['all']].mean().T * 100).round(1)
        print(f"{rec}: {len(ef)} EOD frames, mask QC good (every fish) {len(good)} ({len(good) / len(ef) * 100:.1f}%); EODs: "
              f"{e.mask_qc_good.sum()} of {e.in_video.sum()} in the video; {(time.time() - t0) / 60:.1f} min\n"
              f"(letters -> fish {to_fish})\n{rates.to_string()}", flush=True)
