"""Denis' EOD assignment with my mask midline on the 4-fish recordings tracked by Kaden (efish4-track), at a video-frame offset.

For every EOD of data/Denis Assignment/with Kaden Tracking/Data_4fish_2024-02-20T<t>_no_ref_per.csv: the mask midline of each
fish (fish_midline_with_head on Kaden's visible mask, 101 points evenly spaced, tail -> head) is taken from video frame
Frame + offset (Frame indexes Kaden's frames: the CSV keypoints are his keypoints at frame Frame), and Denis' assign_eods
(EODAssignment repo, analysis/notebooks/silent_fish.run_pipeline, unchanged: no refractory step, highest cosine similarity
wins, his 1/r field with mirror images) is run on them, as in denis_offset_runner_all.py. Identity = Kaden's letters A-D.
EODs where a fish has no mask midline at that frame are not assigned (empty discharging_fish, NaN similarities).

    python denis_mask_midline_kaden_4fish.py --rec 12_32_58 --offsets 0 -5 [--workers 7] [--rows 500]

Output, results/eod_assign_4fish/denis_mask_midline_kaden/:
  midlines/midlines_<t>.npz                      mask midline of every fish per video frame needed (cm, x y), cached
  offset<k>/                                     Denis' input and Assigned_* files
  denis_mask_midline_<t>_offset<k>.csv           one row per EOD, in the order of the input CSV:
      eod_row, Frame, RelativeTime, mask_frame, discharging_fish (A-D, empty if not assigned), similarity_A-D,
      recorded_PTP_pair1-10 (his signed peak-to-peak), predicted_PTP_<X>_pair1-10 (his unit-current prediction for fish X),
      midline_<X> (101 points 'x y;x y;...' in cm, tail -> head, empty where none)
Electrodes: data/electrodes/electrodes_signed_px_coord_4-fish.csv (February layout).
"""
import argparse, pickle, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

DENIS_NB = Path('/Volumes/Lenca_SSD/github/EODAssignment/analysis/notebooks')
import sys
sys.path[:0] = [str(DENIS_NB), str(DENIS_NB.parents[1])]
import silent_fish as sf                                            # noqa: E402  Denis' notebook helpers
import mask_metrics as mm                                           # noqa: E402  (sets the Agg backend)
from midline_methods import fish_midline_with_head                  # noqa: E402

RECS = ('12_32_58', '12_56_22', '13_16_22', '14_56_23', '15_16_23', '15_36_23', '15_56_23', '16_16_23', '16_36_24',
        '16_56_24', '17_16_24', '17_36_24', '17_56_24', '18_36_24')
KADEN = 'data/track_Kaden/results_{t}_full/video2024-02-20T{t}/outputs/video2024-02-20T{t}.pkl'
EOD_CSV = 'data/Denis Assignment/with Kaden Tracking/Data_4fish_2024-02-20T{t}_no_ref_per.csv'
ELECTRODES = 'data/electrodes/electrodes_signed_px_coord_4-fish.csv'
OUT = Path('results/eod_assign_4fish/denis_mask_midline_kaden')
LETTERS, N_POINTS, MIN_PIXELS = 'ABCD', 101, 50
MASKS = None                                                        # Kaden's visible masks, set before the pool forks


def frame_midlines(k):
    '''(4, N_POINTS, 2) mask midlines (cm, x y, tail -> head) of video frame k; NaN for a fish without one.'''
    out = np.full((4, N_POINTS, 2), np.nan)
    for i, rle in enumerate(MASKS[k]):
        if rle is None: continue
        box, crop = mm.rle_crop(rle)
        if box is None or crop.sum() < MIN_PIXELS: continue
        y0, _, x0, _ = box
        try:
            m = fish_midline_with_head(crop, n_ctrl=N_POINTS)['midline']
        except Exception:
            continue
        if m is None or not np.isfinite(m).all(): continue
        m = m * (np.array(crop.shape) - 1) + (y0, x0)
        out[i] = sf.resample(m[:, ::-1], N_POINTS) * sf.PX_TO_CM     # (row, col) px -> (x, y) cm
    return out


def midlines_for(t, frames, workers):
    '''{video frame: (4, N_POINTS, 2)} for the frames asked, computed once per recording and cached.'''
    global MASKS
    path = OUT / 'midlines' / f'midlines_{t}.npz'
    cache = {}
    if path.exists():
        z = np.load(path); cache = dict(zip(z['frames'].tolist(), z['midlines']))
    todo = sorted(set(int(k) for k in frames) - set(cache))
    if todo:
        with open(KADEN.format(t=t), 'rb') as f:
            K = pickle.load(f)
        assert tuple(K['image_size']) == (912, 912) and K['letters'] == LETTERS
        MASKS = K['mask_visible']
        todo = [k for k in todo if 0 <= k < K['n_frames']]
        t0 = time.time()
        with mp.get_context('fork').Pool(workers) as pool:
            for k, m in zip(todo, pool.imap(frame_midlines, todo, chunksize=64)):
                cache[k] = m
        print(f'  {t}: {len(todo)} frames of mask midlines in {(time.time() - t0) / 60:.1f} min', flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        ks = np.array(sorted(cache)); np.savez(path, frames=ks, midlines=np.stack([cache[k] for k in ks]))
    return cache


def denis_input(src):
    '''Kaden CSV (keypoints <point>_<x|y|score>_<letter>) -> Denis' input layout (<point>_<x|y>_<i>, score_<point>_<i>).'''
    rename = {f'{p}_{k}_{f}': (f'score_{p}_{i}' if k == 'score' else f'{p}_{k}_{i}')
              for i, f in enumerate(LETTERS, 1) for p in sf.TRACKED_POINTS for k in ('x', 'y', 'score')}
    return pd.concat([src[['Frame']], src['RelativeTime'].rename('Time'), src[list(rename)].rename(columns=rename),
                      src.filter(like='electrode_')], axis=1)


def midline_text(m):
    return '' if not np.isfinite(m).all() else ';'.join(f'{x:.3f} {y:.3f}' for x, y in m)


def run(t, offsets, workers, rows=None):
    src = pd.read_csv(EOD_CSV.format(t=t), nrows=rows)
    raw = denis_input(src)
    frames = raw.Frame.to_numpy().astype(int)
    by_frame = midlines_for(t, np.concatenate([frames + o for o in offsets]), workers)
    nan = np.full((4, N_POINTS, 2), np.nan)
    for off in offsets:
        t0 = time.time()
        k = frames + off
        mids = np.stack([by_frame.get(int(f), nan) for f in k])                        # (n, 4, N_POINTS, 2)
        ok = np.isfinite(mids).all(axis=(1, 2, 3))
        run_dir = (OUT / f'offset{off:+d}' if rows is None else OUT / f'test_{rows}' / f'offset{off:+d}').resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        csv = run_dir / f'Data_4fish_2024-02-20T{t}.csv'
        sf.midline_input(raw[ok], mids[ok]).to_csv(csv, index=False)
        d = sf.run_pipeline(csv, Path(ELECTRODES).resolve(), 4, midlines_cm=mids[ok])
        assert (d['Frame'].to_numpy() == frames[ok]).all()

        res = pd.DataFrame({'eod_row': np.arange(len(raw)), 'Frame': frames, 'RelativeTime': src.RelativeTime.to_numpy(),
                            'mask_frame': k})
        pick = np.full(len(raw), '', dtype=object); pick[ok] = d['discharging_fish'].astype(str).to_numpy()
        res['discharging_fish'] = pick
        cols = {}
        for f in LETTERS:
            s = np.full(len(raw), np.nan); s[ok] = d[f'similarity_{f}'].to_numpy(); cols[f'similarity_{f}'] = s
        rec = np.full((len(raw), 10), np.nan); rec[ok] = np.stack(d['recorded_PTP'])
        cols.update({f'recorded_PTP_pair{p + 1}': rec[:, p] for p in range(10)})
        for f in LETTERS:
            pr = np.full((len(raw), 10), np.nan); pr[ok] = np.stack(d[f'predicted_PTP_{f}'])
            cols.update({f'predicted_PTP_{f}_pair{p + 1}': pr[:, p] for p in range(10)})
        for i, f in enumerate(LETTERS):
            cols[f'midline_{f}'] = [midline_text(m) for m in mids[:, i]]
        res = pd.concat([res, pd.DataFrame(cols)], axis=1)
        name = f'denis_mask_midline_{t}_offset{off:+d}.csv'
        res.to_csv((OUT if rows is None else OUT / f'test_{rows}') / name, index=False)
        counts = res.discharging_fish[ok].value_counts().reindex(list(LETTERS), fill_value=0)
        print(f'{t}, offset {off:+d}: {ok.sum()} / {len(raw)} EODs assigned (all 4 fish with a mask midline), '
              f'per fish {dict(counts)}, {(time.time() - t0) / 60:.1f} min -> {name}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec', nargs='+', default=list(RECS)); ap.add_argument('--offsets', nargs='+', type=int, default=[0, -5])
    ap.add_argument('--workers', type=int, default=7); ap.add_argument('--rows', type=int, default=None)
    a = ap.parse_args()
    for t in a.rec:
        run(t, a.offsets, a.workers, a.rows)
