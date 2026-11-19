"""Denis' analytical model on the 71 Kaden 4-fish recordings, written in the same format as the cluster run of my model
(cluster_kaden_4fish_assign.py), so the two can be read and compared with the same code. Nothing is recomputed in his model and his
files are not touched: this reads his saved run of 2026-11-06 (Denis' pipeline with my mask midlines, no refractory step) and adds
the voltage rule.

READS, per recording <video> (results/eod_assign_4fish_kaden/):
  <video>/denis_predicted_voltages.npz   his predicted_PTP per fish (n, 4, 10) and the recorded signed PTP (n, 10); NaN where some
                                         fish has no midline (his pipeline needs all four)
  <video>/denis_assignments.csv          his similarity per fish and his own pick (highest similarity)
  <video>/midlines_by_frame.npz          the mask midlines (which pole of each pair is nearer to each fish)
  <video>/mask_qc_good_frames.csv        mask QC
  FACTORS (median fish x signed-electrode factors of his model, fitted on the same 1000 clean isolated EODs per fish as mine)

WRITES results/eod_assign_4fish_kaden/cluster_assign/denis_analytic/:
  Voltage_prediction_4fish_2024-02-20T<video>.npz  the keys of my model's files: eod_rows, frames, recorded (his signed PTP),
      model_raw (his predicted PTP), model_sign (+1), factor, nearer_pole, pred_volt (= factor x model_sign x model_raw), l2_volt,
      l2_rel, l2_shape, pick_voltage, pick_shape, has_midline, good_masks, letters, hyperparameters; plus similarity (n, 4) and
      pick_similarity (his own pick)
  Assigned_Data_4fish_2024-02-20T<video>.csv       one row per EOD: eod_row, Frame, good_masks, r_max, l2_volt_<f>, l2_rel_<f>,
      l2_shape_<f>, pick_voltage, pick_shape, pick_similarity

    python denis_analytic_to_cluster_format.py [video ...]      (default: all 71)
"""
import os, sys, json, time
import numpy as np
import pandas as pd

BASE = 'results/eod_assign_4fish_kaden'
OUT = f'{BASE}/cluster_assign/denis_analytic'
FACTORS = f'{BASE}/l2_distributions/denis_71/denis_voltage_rule_factors_median_best1000.csv'
ELECTRODES = 'data/electrodes/electrodes_signed_px_coord_4-fish.csv'      # the electrode layout of his pipeline
NAME = {'npz': 'Voltage_prediction_4fish_2024-02-20T{video}.npz', 'csv': 'Assigned_Data_4fish_2024-02-20T{video}.csv'}
LETTERS, P, MODEL_SIGN = 'ABCD', 912, +1                                  # his signed PTP and predicted PTP have the same sign
ORDER = [f'{p}{s}' for p in range(1, 11) for s in ('+', '-')]

EL = pd.read_csv(ELECTRODES).sort_values('pair')
PLUS = np.array([(r.x1, r.y1) if str(r.pol1).strip() == '+' else (r.x2, r.y2) for _, r in EL.iterrows()], float)
MINUS = np.array([(r.x2, r.y2) if str(r.pol1).strip() == '+' else (r.x1, r.y1) for _, r in EL.iterrows()], float)
FAC = pd.read_csv(FACTORS); FAC['factor'] = FAC['factor'].str.replace('−', '-'); FAC = FAC.set_index('factor').value
A_FISH = np.array([FAC[f'fish {f}'] for f in LETTERS])
E_ELEC = np.array([FAC.get(f'electrode {e}', 1.0) for e in ORDER]).reshape(10, 2)        # (pair, + / -)


def nearer_pole(mids, chunk=1500):
    '''mids (n, 4, 101, 2) (row, col) / (P - 1) -> (n, 4, 10): 0 = + pole nearer, 1 = - pole (0 for a fish without a midline)'''
    out = np.zeros(mids.shape[:2] + (10,), int)
    for s in range(0, len(mids), chunk):
        xy = mids[s:s + chunk, :, :, ::-1] * (P - 1)
        dp = np.linalg.norm(xy[:, :, :, None] - PLUS[None, None, None], axis=-1).min(2)
        dm = np.linalg.norm(xy[:, :, :, None] - MINUS[None, None, None], axis=-1).min(2)
        out[s:s + chunk] = np.where(np.isfinite(dp) & np.isfinite(dm), dm < dp, 0)
    return out


def pick(d):
    d = np.where(np.isfinite(d), d, np.inf)
    return np.where(np.isfinite(d.min(1)), np.array(list(LETTERS))[d.argmin(1)], '')


def save_atomic(path, writer):
    tmp = path + '.tmp'; writer(tmp); os.replace(tmp, path)


def convert(video):
    t0 = time.time()
    z = np.load(f'{BASE}/{video}/denis_predicted_voltages.npz', allow_pickle=True)
    frames, recorded, model_raw = z['frame'], z['recorded_ptp'], z['predicted_ptp']
    d = pd.read_csv(f'{BASE}/{video}/denis_assignments.csv', keep_default_na=False)
    assert len(d) == len(frames) and (d.Frame.to_numpy() == frames).all(), video
    mz = np.load(f'{BASE}/{video}/midlines_by_frame.npz'); gi = {int(k): i for i, k in enumerate(mz['frames'])}
    idx = np.array([gi.get(int(k), -1) for k in frames])
    mids = np.where(idx[:, None, None, None] >= 0, mz['mids'][np.clip(idx, 0, None)], np.nan); del mz
    has_midline = np.isfinite(mids).all(axis=(2, 3))
    pole = nearer_pole(mids); del mids
    factor = A_FISH[None, :, None] * E_ELEC[np.arange(10)[None, None, :], pole]
    pred = factor * MODEL_SIGN * model_raw
    l2_volt = np.linalg.norm(recorded[:, None] - pred, axis=2)
    r_max = np.abs(recorded).max(1)
    l2_rel = l2_volt / r_max[:, None]
    unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
    l2_shape = np.linalg.norm(unit(recorded)[:, None] - unit(MODEL_SIGN * model_raw), axis=2)
    sim = d[[f'similarity {f}' for f in LETTERS]].replace('', np.nan).astype(float).to_numpy()
    pick_sim = d['fish | Denis, similarity'].astype(str).to_numpy()
    qc = f'{BASE}/{video}/mask_qc_good_frames.csv'
    good = np.isin(frames, pd.read_csv(qc).frame.to_numpy()) if os.path.exists(qc) else np.zeros(len(frames), bool)
    hyper = dict(MODEL='denis', MODEL_SIGN=MODEL_SIGN, source=f'{BASE}/{video}/denis_predicted_voltages.npz (run of 2026-11-06)',
                 FACTORS_FILE=FACTORS, POLE_ELECTRODES=ELECTRODES, MIDLINES=f'{BASE}/{video}/midlines_by_frame.npz', video=video,
                 recorded='signed peak-to-peak (Denis)')
    os.makedirs(OUT, exist_ok=True)
    npz, csv = f'{OUT}/{NAME["npz"].format(video=video)}', f'{OUT}/{NAME["csv"].format(video=video)}'
    def write_npz(p):
        with open(p, 'wb') as fh:
            np.savez_compressed(fh, eod_rows=np.arange(len(frames)), frames=frames, recorded=recorded, model_raw=model_raw,
                                model_sign=MODEL_SIGN, factor=factor, nearer_pole=pole, pred_volt=pred, l2_volt=l2_volt, l2_rel=l2_rel,
                                l2_shape=l2_shape, pick_voltage=pick(l2_volt), pick_shape=pick(l2_shape), has_midline=has_midline,
                                good_masks=good, letters=np.array(list(LETTERS)), hyperparameters=json.dumps(hyper),
                                similarity=sim, pick_similarity=pick_sim)
    save_atomic(npz, write_npz)
    out = pd.DataFrame({'eod_row': np.arange(len(frames)), 'Frame': frames, 'good_masks': good, 'r_max': r_max,
                        **{f'l2_volt_{f}': l2_volt[:, j] for j, f in enumerate(LETTERS)},
                        **{f'l2_rel_{f}': l2_rel[:, j] for j, f in enumerate(LETTERS)},
                        **{f'l2_shape_{f}': l2_shape[:, j] for j, f in enumerate(LETTERS)},
                        'pick_voltage': pick(l2_volt), 'pick_shape': pick(l2_shape), 'pick_similarity': pick_sim})
    save_atomic(csv, lambda p: out.to_csv(p, index=False))
    print(f'{video}: {len(out)} EODs, {(out.pick_voltage != "").sum()} with a pick, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    videos = sys.argv[1:] or sorted(v[-12:-4] for v in os.listdir(f'{BASE}/cluster_assign/mine_N150_tank_barriers_int4')
                                    if v.startswith('Voltage_prediction_4fish_2024-02-20T') and v.endswith('.npz'))
    for v in videos: convert(v)
