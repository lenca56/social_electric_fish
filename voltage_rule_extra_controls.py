"""Additions to voltage_rule_controls.py (2- and 3-fish silent controls), written to their own file so the two scripts never write
the same output:
  1. my N200 tank model WITHOUT the other fish's bodies (--bodies no, N200_tank_nobodies_none) under every rule of
     voltage_rule_controls.py: shape (no gains), voltage L2 / voltage cosine with the fish factor from the length law and 20
     signed-electrode gains fitted by plain least squares in volts (in-sample, and 2-fold by time);
  2. the cosine rule applied to the SHAPE (no gains, no fish factor: pick_cosine, largest cosine of MODEL_SIGN u and r) for every
     model of mine in voltage_rule_controls.MODELS and for the no-bodies model.
Everything else (fish lengths, length law, nearer-pole electrode per fish and pair, gain fit, picks) is taken from
voltage_rule_controls.py itself: its constants and functions (everything above its main loop) are executed here, so both use the
same code.

    python voltage_rule_extra_controls.py
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/voltage_rule/: accuracy_extra.csv (same columns as accuracy.csv),
gains_extra.csv, predicted_voltages_<model>.npz (N200_tank_no_fish_bodies, N150_square_no_fish_bodies), and accuracy_combined.csv (accuracy.csv + accuracy_extra.csv).

'all EODs' (from 2026-11-10) leaves out the few EODs with negative Time in the newest table (before the video starts; they
carry Frame 1 and are never in a clean good frame).
"""
import os
import numpy as np
import pandas as pd

src = open('voltage_rule_controls.py').read()
exec(src[:src.index('\nfor rec, r in REC.items():')])                 # constants and functions of voltage_rule_controls.py

EXTRA = {'N200 tank, no fish bodies': ('mine', '{out}/N200_tank_nobodies_none/eod_assignment_shape.csv', 'electrode_locations_updated'),
         'N150 square, no fish bodies': ('mine', '{out}/N150_image_nobodies_none/eod_assignment_shape.csv', 'electrodes_signed_px_coord')}

for rec, r in REC.items():
    c = g.REC[rec]; em, nf = r['emitter'], r['n']; out = f"{c['out']}/voltage_rule"
    q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv"); clean = q.clean_good.to_numpy(bool); frames = q.Frame.to_numpy()
    valid = load_eods(c['newest'], columns=['Time'], traces=False)[0].Time.to_numpy() >= 0   # 'all EODs': no EODs before the video
    n = len(q); half = np.arange(n) < n // 2
    z = np.load(c['midlines']); M = dict(zip(z['frames'].tolist(), z['midlines_px']))
    L = pd.read_csv(f'{out}/fish_lengths.csv').iloc[:, 1].to_numpy()          # the lengths voltage_rule_controls.py used
    rows, gains_rows = [], []
    # 1. shape cosine for every model of mine
    for name, (kind, path, el) in {**MODELS, **EXTRA}.items():
        if kind != 'mine': continue
        d = load_assignment(path.format(res=r['res'], out=c['out'])).table
        assert (d.Frame.to_numpy() == frames).all(); ok = d.pick_cosine.to_numpy() == em
        rows.append(dict(model=name, rule='shape cosine (no gains)', **{'all EODs': 100 * ok[valid].mean(), 'clean good': 100 * ok[clean].mean()}))
    # 2. the no-bodies model under every rule
    for name, (kind, path, el) in EXTRA.items():
        pairs = load_electrodes(f'data/electrodes/{el}_{rec}.csv', P); pair_ids = [p for p, _, _ in pairs]
        pairs_px = [(pp[::-1] * (P - 1), pm[::-1] * (P - 1)) for _, pp, pm in pairs]
        a = load_assignment(path.format(res=r['res'], out=c['out']))
        assert (a.table.Frame.to_numpy() == frames).all() and a.pair_ids == pair_ids and a.fish == list(range(nf))
        R, Mm = a.recorded, MODEL_SIGN * a.predicted
        A = LAW[kind][0] * L + LAW[kind][1]
        K = nearer_pole(M, frames, pairs_px, nf)
        Mem, Kem = A[em] * Mm[:, em], K[:, em]
        res = {'shape (no gains)': a.pick == em}
        G, cnt = fit_gains(R, Mem, Kem, clean)
        V = A[None, :, None] * np.take(G, np.where(K >= 0, K, 0)) * Mm; V[K < 0] = np.nan
        pl2, pcos = picks(R, V); res['voltage L2, in-sample'] = pl2 == em; res['voltage cosine, in-sample'] = pcos == em
        np.savez_compressed(f"{out}/predicted_voltages_{name.replace(',', '').replace(' ', '_')}.npz", predicted_volts=V, recorded=R, gains=G,
                            nearer_pole=K, fish_factor=A, fish_length_cm=L)
        cv_l2, cv_cos = np.zeros(n, bool), np.zeros(n, bool)
        for train in (half, ~half):
            Gt, _ = fit_gains(R, Mem, Kem, clean & train)
            Vt = A[None, :, None] * np.take(Gt, np.where(K >= 0, K, 0)) * Mm; Vt[K < 0] = np.nan
            a2, ac = picks(R[~train], Vt[~train]); cv_l2[~train] = a2 == em; cv_cos[~train] = ac == em
        res['voltage L2, 2-fold by time'] = cv_l2; res['voltage cosine, 2-fold by time'] = cv_cos
        cv_l2c = np.zeros(n, bool)                                             # control as in voltage_rule_controls.py: no length law
        for train in (half, ~half):
            Gt, _ = fit_gains(R, Mm[:, em], Kem, clean & train)
            Vt = np.take(Gt, np.where(K >= 0, K, 0)) * Mm; Vt[K < 0] = np.nan
            cv_l2c[~train] = picks(R[~train], Vt[~train])[0] == em
        res['voltage L2, 2-fold, no length law'] = cv_l2c
        for rule, ok in res.items():
            rows.append(dict(model=name, rule=rule, **{'all EODs': 100 * ok[valid].mean(), 'clean good': 100 * ok[clean].mean()}))
        gains_rows += [dict(model=name, electrode=f'{(k // 2) + 1}{"+−"[k % 2]}', gain=G[k], n_terms=int(cnt[k])) for k in range(20)]
    T = pd.DataFrame(rows); T.to_csv(f'{out}/accuracy_extra.csv', index=False); pd.DataFrame(gains_rows).to_csv(f'{out}/gains_extra.csv', index=False)
    B = pd.concat([pd.read_csv(f'{out}/accuracy.csv'), T], ignore_index=True); B.to_csv(f'{out}/accuracy_combined.csv', index=False)
    order = list(dict.fromkeys(B.model)); rules = list(dict.fromkeys(B.rule))
    with pd.option_context('display.width', 260, 'display.max_columns', 20):
        for col in ('clean good', 'all EODs'):
            print(f'\n{rec}, % correct, {col}:'); print(B.pivot_table(index='model', columns='rule', values=col).reindex(index=order, columns=rules).round(2).to_string())
