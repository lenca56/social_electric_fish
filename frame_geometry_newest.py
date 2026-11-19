"""Fish geometry at each EOD of the *_synch_fix_newest.csv tables (2- and 3-fish silent controls), from the ZEUS masks
(*_Shiv_tracking_final.pkl) of the EOD's video frame: per fish, mask centroid (cm, x = column, y = row from the top-left of the
video) and area; the emitter's distance to the nearest other fish (cm, mask edge to edge: closest pair of mask pixels) and the
distance between every pair of fish. A fish with no mask in the frame has NaN.

    OMP_NUM_THREADS=1 python frame_geometry_newest.py [--workers 6]

Output: <res>/eod_geometry_synch_fix_newest.csv, one row per EOD of the newest table (eod_row, Frame, clean_good from
eod_clean_good_synch_fix_newest.csv, x_<f>, y_<f>, area_<f>, gap_<f>_<g>, gap_emitter_cm).
"""
import argparse, itertools, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from io_utils import load_masks, load_eods

PX_PER_CM = 912 / 60
REC = {
    '2-fish': dict(fish=(0, 1), emitter=1, res='results/eod_ground_truth',
                   track='data/2_fish_silent/elephantnose_2024-02-13T14-33-21_Shiv_tracking_final.pkl',
                   newest='data/2_fish_silent/Data_2fish_2024-02-13T14_33_21_synch_fix_newest.csv'),
    '3-fish': dict(fish=(0, 1, 2), emitter=0, res='results/eod_ground_truth_3fish',
                   track='data/3_fish_silent/elephantnose_2024-10-15T14-12-25_Shiv_tracking_final.pkl',
                   newest='data/3_fish_silent/Data_3fish_2024-10-15T14_12_25_synch_fix_newest.csv'),
}
MASKS, FISH = None, None                                            # set in the parent before the pool forks


def frame_geometry(keys):
    out = []
    for k in keys:
        pts = {FISH[f]: np.argwhere(crop) + (y0, x0) for f, ((y0, y1, x0, x1), crop) in MASKS.crops(k).items()}
        row = dict(Frame=int(k))
        for f in FISH:
            yx = pts.get(f)
            row[f'x_{f}'], row[f'y_{f}'] = (yx[:, 1].mean() / PX_PER_CM, yx[:, 0].mean() / PX_PER_CM) if yx is not None and len(yx) else (np.nan, np.nan)
            row[f'area_{f}'] = 0 if yx is None else len(yx) / PX_PER_CM ** 2
        for f, g in itertools.combinations(FISH, 2):
            ok = f in pts and g in pts and len(pts[f]) and len(pts[g])
            row[f'gap_{f}_{g}'] = cKDTree(pts[f]).query(pts[g])[0].min() / PX_PER_CM if ok else np.nan
        out.append(row)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6); a = ap.parse_args()
    for name, c in REC.items():
        t0 = time.time()
        eod = load_eods(c['newest'], columns=['Frame'], traces=False)[0]
        flags = pd.read_csv(f"{c['res']}/eod_clean_good_synch_fix_newest.csv")
        assert (flags.Frame.to_numpy() == eod.Frame.to_numpy()).all()
        FISH = c['fish']; MASKS = load_masks(c['track'], 'shiv', fish_order=FISH)
        uk = np.unique(eod.Frame)
        with mp.get_context('fork').Pool(a.workers) as pool:
            parts = pool.map(frame_geometry, np.array_split(uk, 16 * a.workers))
        G = pd.DataFrame([r for p in parts for r in p]).set_index('Frame')
        out = flags[['eod_row', 'Frame', 'clean_good']].join(G, on='Frame')
        e = c['emitter']
        out['gap_emitter_cm'] = out[[f'gap_{min(e, g)}_{max(e, g)}' for g in c['fish'] if g != e]].min(axis=1, skipna=False)
        out.to_csv(f"{c['res']}/eod_geometry_synch_fix_newest.csv", index=False)
        print(f"{name}: {len(out)} EODs on {len(uk)} frames; emitter gap NaN (a fish without mask) in "
              f"{out.gap_emitter_cm.isna().sum()} EODs; {(time.time() - t0) / 60:.1f} min", flush=True)
        MASKS = None
