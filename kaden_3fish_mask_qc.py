"""Mask QC of the 8 3-fish recordings (Kaden's final masks, visible): the frames in which the tracking is good, with the rules of the
4-fish recordings (kaden_mask_qc.py, thresholds imported from there), for the 'good' EOD set of kaden_3fish_assign_rules.py.

Per fish and frame, a mask passes when:
- the fish has a mask (one mask per letter, no remnants possible);
- its area is within 20% of that fish's median area;
- its principal-axis length is 0.8-1.2 x that fish's median;
- >= 99% of the mask is one connected piece;
- its centroid moves <= 15 px and its area changes <= 15% to the previous AND the next frame;
- it is not within 5 px of any other fish's mask.
A frame is good when all three fish pass. Medians are per recording, over the frames computed here (EOD frames +- 1).

    OMP_NUM_THREADS=1 python kaden_3fish_mask_qc.py [--workers 7] [--stage final]

Output in results/eod_assignment_3_fish_silent/mask_stages/mask_qc/:
  mask_qc_good_frames.csv        video, frame: the good EOD frames of every recording (read by kaden_3fish_assign_rules.py)
  mask_qc_rates.csv              video, fish, % of EOD frames passing each check and all of them
  <rec>/mask_qc_per_fish.csv     frame, fish, metrics and checks
"""
import os, sys, argparse, time
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import mask_metrics as mm
import kaden_mask_qc as kq                                          # in_contact and the thresholds of the 4-fish QC
import cluster_kaden_3fish_common as C
from io_utils import load_stage_masks

OUT = f'{C.OUT}/mask_qc'
MASKS = None                                                        # set before the pool forks


def frame_rows(keys):
    rows = []
    for k in keys:
        crops = MASKS.crops(k)
        for f, (box, crop) in crops.items():
            rows.append(dict(frame=int(k), fish=C.LETTERS[f], contact=any(kq.in_contact(crops[f], crops[o]) for o in crops if o != f),
                             **mm.shape_metrics(crop, box)))
    return rows


def qc(video, stage, workers):
    global MASKS
    t0 = time.time()
    MASKS = load_stage_masks(C.MASK_RUN.format(video=video), stage); assert ''.join(MASKS.names) == C.LETTERS; n = len(MASKS)
    ef = np.unique(C.eod_frames(video)); ef = ef[ef >= 0]
    keys = np.unique(np.r_[ef - 1, ef, ef + 1]); keys = keys[(keys >= 0) & (keys < n)]
    with mp.get_context('fork').Pool(workers) as pool:
        parts = pool.map(frame_rows, np.array_split(keys, 16 * workers))
    q = pd.DataFrame([r for p in parts for r in p])
    full = pd.MultiIndex.from_product([keys, list(C.LETTERS)], names=['frame', 'fish'])
    q = q.set_index(['frame', 'fish']).reindex(full).reset_index()
    q['present'] = q.px.fillna(0) > 0
    for f in C.LETTERS:                                             # worst of the step to the previous and to the next frame
        s = q[q.fish == f].set_index('frame').reindex(np.arange(keys.min() - 1, keys.max() + 2))
        step = np.hypot(s.cx.diff(), s.cy.diff()); darea = (s.px / s.px.shift(1) - 1).abs(); m = q.fish == f
        q.loc[m, 'step'] = np.fmax(step, step.shift(-1)).fillna(np.inf).loc[q.frame[m]].to_numpy()
        q.loc[m, 'darea'] = np.fmax(darea, darea.shift(-1)).fillna(np.inf).loc[q.frame[m]].to_numpy()
    med = q[q.present].groupby('fish')[['px', 'major']].median()
    q['area_ratio'] = q.px / q.fish.map(med.px); q['length_ratio'] = q.major / q.fish.map(med.major)
    checks = {
        'has a mask': q.present,
        f'area within {kq.AREA_TOL:.0%} of median': (q.area_ratio - 1).abs() <= kq.AREA_TOL,
        f'length {kq.LENGTH_RANGE[0]}-{kq.LENGTH_RANGE[1]} x median': q.length_ratio.between(*kq.LENGTH_RANGE),
        f'>= {kq.MIN_LARGEST_FRAC:.0%} in one piece': q.largest_frac >= kq.MIN_LARGEST_FRAC,
        f'step <= {kq.MAX_STEP_PX} px': q.step <= kq.MAX_STEP_PX,
        f'area change <= {kq.MAX_DAREA:.0%}': q.darea <= kq.MAX_DAREA,
        'not touching other fish': ~q.contact.astype('boolean').fillna(True).to_numpy(bool),
    }
    for c, v in checks.items(): q[c] = np.asarray(v)
    q['all'] = pd.DataFrame(checks).all(axis=1).to_numpy()
    os.makedirs(f'{OUT}/{video}', exist_ok=True); q.to_csv(f'{OUT}/{video}/mask_qc_per_fish.csv', index=False)
    good = q.groupby('frame')['all'].all(); good = good.index[good.to_numpy()].to_numpy(); good = good[np.isin(good, ef)]
    rates = (q[q.frame.isin(ef)].groupby('fish')[list(checks) + ['all']].mean() * 100).round(1).reset_index().assign(video=video)
    print(f'{video}: {len(ef)} EOD frames, good (all three fish) {len(good)} ({len(good) / len(ef) * 100:.1f}%), '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return pd.DataFrame({'video': video, 'frame': good}), rates


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=7); ap.add_argument('--stage', default='final')
    a = ap.parse_args()
    res = [qc(v, a.stage, a.workers) for v in C.VIDEOS]
    G = pd.concat([r[0] for r in res], ignore_index=True); R = pd.concat([r[1] for r in res], ignore_index=True)
    G.to_csv(f'{OUT}/mask_qc_good_frames.csv', index=False); R.to_csv(f'{OUT}/mask_qc_rates.csv', index=False)
    print(R.set_index(['video', 'fish']).to_string())
