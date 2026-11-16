"""Voltage rule on a 4-fish Kaden recording with my model in the camera tank (kaden_4fish_tank_model.py: L-barriers, interior
4 x water, N 150), using median fish and signed-electrode factors.

For every EOD of recording <t> and every fish f with a model:
    predicted pair voltage p = a_f * e_k(f, p) * MODEL_SIGN * u_f,p
- u: my unit-current model of the EOD's frame (results/eod_assign_4fish_kaden/<t>/model_N150_tank_barriers_int4_by_frame.pkl);
- a_f: the median fish factor, e_k: the median factor of signed electrode k, where k is the electrode of pair p (+ or -) nearer
  to fish f's mask midline in that frame (FACTORS, from EOD_gain_fish_x_signed_electrode_71_mine_tank_4fish_kaden.ipynb: ratios
  |V_rec| / |V_pred| on the 2 loudest pairs of the 1000 best clean EODs per fish; an electrode without ratios has e = 1);
- recorded r: the main-phase pattern (10 pairs at the sample of the largest |V|, each pair's trace demeaned), volts.
The EOD goes to the fish with the smallest ||r - p|| (L2 in volts). For comparison the shape rule (each pattern scaled to
max |v| = 1, smallest L2) and Denis' pick (denis_assignments.csv, L2 with max |v| = 1) are saved too.

Output: results/eod_assign_4fish_kaden/<t>/voltage_rule_N150_tank_barriers_int4.csv, one row per EOD: row, Frame, good_masks
(mask QC), r_max (V), d_volt_<f> and d_shape_<f> per fish (NaN without a model), pick_voltage, pick_shape, pick_denis.

    OMP_NUM_THREADS=1 python kaden_4fish_voltage_rule.py --rec 05_52_55
"""
import argparse, time
import numpy as np
import pandas as pd
from eod_assign import load_electrode_pairs, recorded_pair_traces_all, MODEL_SIGN
from kaden_4fish_assign import EOD

BASE, P, LET = 'results/eod_assign_4fish_kaden', 912, 'ABCD'
MODEL = 'model_N150_tank_barriers_int4_by_frame.pkl'
FACTORS = f'{BASE}/clean_emitter_set_71/gain_mine_tank_int4/voltage_rule_factors_median.csv'
EL_FILE = 'data/electrodes/electrode_locations_updated_4-fish.csv'
ORDER = [f'{p}{s}' for p in range(1, 11) for s in ('+', '−')]


def nearer_pole(mids, chunk=2000):
    '''(n_frames, 4 fish, 10 pairs) index 0 (+ pole nearer) or 1 (- pole nearer) from the mask midlines ((row, col) / (P - 1)).'''
    EL = pd.read_csv(EL_FILE).sort_values('pair')
    plus = np.array([(r.x1, r.y1) if str(r.pol1).strip() == '+' else (r.x2, r.y2) for _, r in EL.iterrows()], float)
    minus = np.array([(r.x2, r.y2) if str(r.pol1).strip() == '+' else (r.x1, r.y1) for _, r in EL.iterrows()], float)
    out = np.zeros(mids.shape[:2] + (10,), int)
    for s in range(0, len(mids), chunk):
        xy = mids[s:s + chunk, :, :, ::-1] * (P - 1)                               # (n, 4, 101, x/y) px
        dp = np.linalg.norm(xy[:, :, :, None] - plus[None, None, None], axis=-1).min(2)
        dm = np.linalg.norm(xy[:, :, :, None] - minus[None, None, None], axis=-1).min(2)
        out[s:s + chunk] = (dm < dp).astype(int)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True); a = ap.parse_args(); t = a.rec; t0 = time.time()
    F = pd.read_csv(FACTORS).set_index('factor').value
    A = np.array([F[f'fish {f}'] for f in LET]); E = np.array([F[f'electrode {e}'] for e in ORDER]).reshape(10, 2)   # (pair, +/-)
    U = pd.read_pickle(f'{BASE}/{t}/{MODEL}')
    PIDS = [p for p, _, _ in load_electrode_pairs(EL_FILE, P)]
    df = pd.read_csv(EOD.format(t=t), usecols=['Frame'] + [f'electrode_{p}_time_{i}' for p in PIDS for i in range(1, 26)])
    frames = df.Frame.to_numpy(); V = recorded_pair_traces_all(df, PIDS, 25); del df
    R = V[np.arange(len(V)), :, np.abs(V).max(1).argmax(1)]; del V                   # (n, 10) main phase, V
    z = np.load(f'{BASE}/{t}/midlines_by_frame.npz'); gi = {int(k): i for i, k in enumerate(z['frames'])}
    pole = nearer_pole(z['mids'])
    n = len(R); Uall = np.full((n, 4, 10), np.nan); Pole = np.zeros((n, 4, 10), int)
    for i, k in enumerate(frames):
        if int(k) in U: Uall[i] = MODEL_SIGN * U[int(k)]
        if int(k) in gi: Pole[i] = pole[gi[int(k)]]
    G = A[None, :, None] * E[np.arange(10)[None, None, :], Pole]                    # (n, 4, 10) a_f * e_k
    pred = G * Uall
    d_volt = np.linalg.norm(R[:, None] - pred, axis=2)
    unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
    d_shape = np.linalg.norm(unit(R)[:, None] - unit(Uall), axis=2)
    def pick(d):
        d = np.where(np.isfinite(d), d, np.inf); return np.where(np.isfinite(d.min(1)), np.array(list(LET))[d.argmin(1)], '')
    good = set(pd.read_csv(f'{BASE}/{t}/mask_qc_good_frames.csv').frame)
    den = pd.read_csv(f'{BASE}/{t}/denis_assignments.csv', usecols=['Frame', 'fish | Denis, L2 (max |v| = 1)'])
    assert (den.Frame.to_numpy() == frames).all()
    out = pd.DataFrame({'row': np.arange(n), 'Frame': frames, 'good_masks': np.isin(frames, list(good)), 'r_max': np.abs(R).max(1),
                        **{f'd_volt_{f}': d_volt[:, j] for j, f in enumerate(LET)}, **{f'd_shape_{f}': d_shape[:, j] for j, f in enumerate(LET)},
                        'pick_voltage': pick(d_volt), 'pick_shape': pick(d_shape), 'pick_denis': den['fish | Denis, L2 (max |v| = 1)'].fillna('').to_numpy()})
    path = f'{BASE}/{t}/voltage_rule_N150_tank_barriers_int4.csv'; out.to_csv(path, index=False)
    ok = out.pick_voltage != ''
    print(f'{t}: {n} EODs, {ok.sum()} with a voltage pick; agreement voltage vs Denis {(out.pick_voltage == out.pick_denis)[ok].mean() * 100:.1f}%, '
          f'voltage vs shape {(out.pick_voltage == out.pick_shape)[ok].mean() * 100:.1f}%; shares ' +
          ' '.join(f'{f} {(out.pick_voltage[ok] == f).mean() * 100:.1f}%' for f in LET) + f'; {(time.time() - t0) / 60:.1f} min', flush=True)
