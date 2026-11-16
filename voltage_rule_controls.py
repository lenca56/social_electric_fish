"""EOD assignment by predicted VOLTAGE (not shape), 2- and 3-fish silent controls, for my numerical model (several geometries) and
Denis' analytic model, without solving any field again.

Predicted voltage of fish f on pair p (volts):  v_fp = a(L_f) * g_k(f,p) * m_fp
  m_fp     the model's prediction for fish f: mine = MODEL_SIGN x unit-current pair voltage (main phase), from each run's
           eod_assignment_shape.csv; Denis = his predicted peak-to-peak (predicted_PTP_<X>, his units) from Assigned_*.pkl;
  a(L)     fish factor from the linear law of fish_factor_vs_length.png (4-fish Kaden, fish x signed-electrode fit,
           results/eod_assign_4fish_kaden/clean_emitter_set/gain_fish_x_signed_electrode/fish_factors_and_length_fits.csv):
           mine a = 0.01066 L + 0.3139, Denis a = 0.0002045 L + 0.002990; L = that fish's length here, measured as for the law
           (90th percentile over the EOD frames of the mask-midline arc length, 15.2 px/cm);
  g_k      gain of signed electrode k (20: pair 1-10, + or - pole), k(f, p) = the pole of pair p nearer to fish f's midline (that
           model's electrode positions). Fitted per recording and model by plain least squares in volts on the true emitter of
           clean-good EODs: min sum_i sum_p (r_ip - g_k a(L_em) m_em,p)^2 (closed form per electrode; an electrode never nearer
           to the emitter keeps g = 1).
  recorded r: mine = the 10 pair voltages at the sample where |V| peaks (r_raw); Denis = his recorded peak-to-peak (recorded_PTP).

Rules, per EOD over the fish with a prediction: 'voltage L2' = smallest ||r - v_f|| (volts); 'voltage cosine' = largest cosine of
r and v_f. Reference: each model's own shape rule (mine: L2 of max-normalised patterns; Denis: cosine of r and m_f, his pick).
Denis' 1/r model has near-singular predictions (a current point almost on an electrode: up to ~34,000 x the median) that dominate
a least-squares gain; two extra Denis rows refit the gains without terms whose |a(L) m| exceeds SINGULAR_RATIO x their electrode's
median ('fit without near-singular'), and also cap every prediction at that value when scoring ('..., predictions capped').
Fits: 'in-sample' = gains fitted on all clean-good EODs; '2-fold by time' = gains fitted on the clean-good EODs of one half of
the recording (by EOD order) and applied to the other half, and vice versa (out of sample). Control 'no length law': the same
2-fold L2 rule with one factor for every fish (only the electrode gains).

    python voltage_rule_controls.py
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/voltage_rule/: accuracy.csv, gains.csv, fish_lengths.csv,
predicted_voltages_<model>.npz (in-sample v, gains, nearer-pole index per EOD x fish x pair).

'all EODs' (from 2026-11-10) leaves out the few EODs with negative Time in the newest table (before the video starts; they
carry Frame 1 and are never in a clean good frame).
"""
import os
import numpy as np
import pandas as pd
import eod_assignment_control_groups as g
from eod_assign import load_electrode_pairs, MODEL_SIGN

P, PX_CM_LAW = 912, 912 / 60
LAW = {'mine': (0.010661805405381211, 0.31386049313202685), 'Denis': (0.00020448335330612151, 0.002989576395375126)}
REC = {'2-fish': dict(res='results/eod_ground_truth', emitter=1, letters='AB', zeus={'A': 1, 'B': 0}, n=2),
       '3-fish': dict(res='results/eod_ground_truth_3fish', emitter=0, letters='ABC', zeus={'A': 1, 'B': 2, 'C': 0}, n=3)}
MODELS = {'Nov 5: N150, square': ('mine', '{res}/conductivities/my_model_shape_synch_fix_newest.csv', 'electrodes_signed_px_coord'),
          'N200 first tank, no electrodes': ('mine', '{out}/N200_tank_none/eod_assignment_shape.csv', 'electrode_locations_updated'),
          'N150 enlarged tank, no electrodes': ('mine', '{out}/N150_tank_enlarged_none/eod_assignment_shape.csv',
                                                'electrode_locations_enlarged'),
          'N150 enlarged tank, discs r 0.5 cm': ('mine', '{out}/N150_tank_enlarged_r0.5_s1e7/eod_assignment_shape.csv',
                                                 'electrode_locations_enlarged'),
          'Denis analytic, my mask': ('Denis', None, 'electrodes_signed_px_coord')}


def nearer_pole(M, frames, pairs_px, n_fish):
    """(n_eods, n_fish, 10) signed-electrode index 2 (pair - 1) + (0 for the + pole, 1 for -), the pole nearer to each fish's
       midline (px); -1 where the fish has no midline."""
    cache = {}
    for k in np.unique(frames):
        out = np.full((n_fish, len(pairs_px)), -1)
        for f in range(n_fish):
            mid = M[int(k)][f]
            if not np.isfinite(mid).all(): continue
            for j, (pp, pm) in enumerate(pairs_px):
                dp, dm = (np.linalg.norm(mid - q, axis=1).min() for q in (pp, pm))
                out[f, j] = 2 * j + (0 if dp <= dm else 1)
        cache[int(k)] = out
    return np.stack([cache[int(k)] for k in frames])


SINGULAR_RATIO = 20                  # Denis' refit: a term is near-singular when |a(L) m| > 20 x its electrode's median (fit terms)


def singular_caps(Mem, Kem, sel, ratio=SINGULAR_RATIO):
    """Per signed electrode: ratio x the median |a(L) m| of the emitter's terms in sel (inf where an electrode has none)."""
    m, k = np.abs(Mem[sel]).ravel(), Kem[sel].ravel(); ok = np.isfinite(m) & (k >= 0)
    return np.array([ratio * np.median(m[ok & (k == e)]) if (ok & (k == e)).any() else np.inf for e in range(20)])


def fit_gains(R, Mem, Kem, sel, caps=None):
    """Least-squares gain per signed electrode from the true emitter's terms of the EODs in sel; with caps, terms whose |a(L) m|
       exceeds their electrode's cap (near-singular predictions) are left out."""
    r, m, k = R[sel].ravel(), Mem[sel].ravel(), Kem[sel].ravel()
    ok = np.isfinite(r) & np.isfinite(m) & (k >= 0)
    if caps is not None: ok &= np.abs(m) <= caps[np.where(k >= 0, k, 0)]
    num = np.bincount(k[ok], r[ok] * m[ok], minlength=20); den = np.bincount(k[ok], m[ok] ** 2, minlength=20)
    n = np.bincount(k[ok], minlength=20)
    return np.where(den > 0, num / np.where(den > 0, den, 1), 1.0), n


def picks(R, V):
    """Picks (n_eods,) of the voltage L2 and cosine rules over fish with a prediction (-1: none)."""
    d = np.linalg.norm(R[:, None] - V, axis=-1)
    cos = (V * R[:, None]).sum(-1) / (np.linalg.norm(V, axis=-1) * np.linalg.norm(R, axis=-1)[:, None])
    ok = np.isfinite(d).any(1)
    return (np.where(ok, np.argmin(np.where(np.isfinite(d), d, np.inf), 1), -1),
            np.where(ok, np.argmax(np.where(np.isfinite(cos), cos, -np.inf), 1), -1))


for rec, r in REC.items():
    c = g.REC[rec]; em, nf = r['emitter'], r['n']; out = f"{c['out']}/voltage_rule"; os.makedirs(out, exist_ok=True)
    q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv"); clean = q.clean_good.to_numpy(bool); frames = q.Frame.to_numpy()
    valid = pd.read_csv(c['newest'], usecols=['Time']).Time.to_numpy() >= 0   # 'all EODs' leaves out EODs before the video starts
    n = len(q); half = np.arange(n) < n // 2
    z = np.load(c['midlines']); M = dict(zip(z['frames'].tolist(), z['midlines_px']))
    Lv = np.stack([np.linalg.norm(np.diff(M[int(k)], axis=1), axis=-1).sum(-1) for k in np.unique(frames)]) * 1.0 / PX_CM_LAW
    L = np.array([np.nanpercentile(Lv[:, f], 90) for f in range(nf)])
    pd.DataFrame({'fish (ZEUS label)': range(nf), 'length_cm (90th pct, 15.2 px/cm)': L.round(3)}).to_csv(f'{out}/fish_lengths.csv', index=False)
    print(f'\n{rec}: fish lengths (cm, as for the law)', dict(enumerate(L.round(2))), flush=True)

    # Denis' predictions and recorded PTP, back on every EOD row (NaN where some fish had no midline)
    dn_all = pd.read_csv(f"{r['res']}/conductivities/denis_mask_midline_offset/denis_mask_midline_all_eods_synch_fix_newest.csv")
    has = dn_all.pick.notna().to_numpy()
    dk = pd.read_pickle(next(p for p in (f"{r['res']}/conductivities/denis_mask_midline_offset/synch_fix_newest/{x}"
                                          for x in os.listdir(f"{r['res']}/conductivities/denis_mask_midline_offset/synch_fix_newest"))
                             if p.endswith('.pkl')))
    assert has.sum() == len(dk) and (dk.Frame.to_numpy() == frames[has]).all()
    D_R = np.full((n, 10), np.nan); D_R[has] = np.stack(dk.recorded_PTP.to_numpy())
    D_M = np.full((n, nf, 10), np.nan)
    for X in r['letters']: D_M[has, r['zeus'][X]] = np.stack(dk[f'predicted_PTP_{X}'].to_numpy())
    denis_pick = dn_all.pick.map(r['zeus']).fillna(-1).to_numpy(int)

    rows, gains_rows = [], []
    for name, (kind, path, el) in MODELS.items():
        pairs = load_electrode_pairs(f'data/electrodes/{el}_{rec}.csv', P); pair_ids = [p for p, _, _ in pairs]
        pairs_px = [(pp[::-1] * (P - 1), pm[::-1] * (P - 1)) for _, pp, pm in pairs]           # (x, y) px
        if kind == 'mine':
            d = pd.read_csv(path.format(res=r['res'], out=c['out'])); assert (d.Frame.to_numpy() == frames).all()
            R = d[[f'r_raw_p{q_}' for q_ in pair_ids]].to_numpy()
            Mm = MODEL_SIGN * np.stack([d[[f'u{f}_raw_p{q_}' for q_ in pair_ids]].to_numpy() for f in range(nf)], axis=1)
            shape_pick = d.pick.to_numpy()
        else:
            R, Mm, shape_pick = D_R, D_M, denis_pick
        A = LAW[kind][0] * L + LAW[kind][1]                                    # fish factor per fish
        K = nearer_pole(M, frames, pairs_px, nf)
        Mem, Kem = A[em] * Mm[:, em], K[:, em]
        res = {'shape (no gains)': shape_pick == em}
        G, cnt = fit_gains(R, Mem, Kem, clean)                                 # in-sample
        V = A[None, :, None] * np.take(G, np.where(K >= 0, K, 0)) * Mm; V[K < 0] = np.nan
        pl2, pcos = picks(R, V); res['voltage L2, in-sample'] = pl2 == em; res['voltage cosine, in-sample'] = pcos == em
        np.savez_compressed(f"{out}/predicted_voltages_{name.split(':')[0].replace(' ', '_').replace(',', '')}.npz",
                            predicted_volts=V, recorded=R, gains=G, nearer_pole=K, fish_factor=A, fish_length_cm=L)
        cv_l2, cv_cos = np.zeros(n, bool), np.zeros(n, bool)                   # 2-fold by time
        for train in (half, ~half):
            Gt, _ = fit_gains(R, Mem, Kem, clean & train)
            Vt = A[None, :, None] * np.take(Gt, np.where(K >= 0, K, 0)) * Mm; Vt[K < 0] = np.nan
            a2, ac = picks(R[~train], Vt[~train]); cv_l2[~train] = a2 == em; cv_cos[~train] = ac == em
        res['voltage L2, 2-fold by time'] = cv_l2; res['voltage cosine, 2-fold by time'] = cv_cos
        cv_l2c = np.zeros(n, bool)                                             # control: same factor for every fish (no length law)
        for train in (half, ~half):
            Gt, _ = fit_gains(R, Mm[:, em], Kem, clean & train)
            Vt = np.take(Gt, np.where(K >= 0, K, 0)) * Mm; Vt[K < 0] = np.nan
            cv_l2c[~train] = picks(R[~train], Vt[~train])[0] == em
        res['voltage L2, 2-fold, no length law'] = cv_l2c
        for rule, ok in res.items():
            rows.append(dict(model=name, rule=rule, **{'all EODs': 100 * ok[valid].mean(), 'clean good': 100 * ok[clean].mean()}))
        if kind == 'Denis':                                                    # refit without near-singular terms (1/r spikes)
            Kc = np.where(K >= 0, K, 0)
            def volts(Gx, caps, cap_pred, with_len=True):
                m_all = (A[None, :, None] if with_len else 1.0) * Mm
                if cap_pred: m_all = np.sign(m_all) * np.minimum(np.abs(m_all), np.take(caps, Kc))
                Vx = np.take(Gx, Kc) * m_all; Vx[K < 0] = np.nan; return Vx
            caps_all = singular_caps(Mem, Kem, clean)
            n_cut = int(sum(((np.abs(Mem[clean]) > np.take(caps_all, np.where(Kem[clean] >= 0, Kem[clean], 0))) & (Kem[clean] >= 0)
                             & np.isfinite(Mem[clean])).sum() for _ in [0]))
            print(f'{rec}: Denis refit leaves out {n_cut} near-singular emitter terms of {int(np.isfinite(Mem[clean]).sum())} '
                  f'(|a m| > {SINGULAR_RATIO} x the electrode median)', flush=True)
            for cap_pred, label in ((False, 'fit without near-singular'), (True, 'fit without near-singular, predictions capped')):
                rr = {'shape (no gains)': shape_pick == em}
                Gr, nr = fit_gains(R, Mem, Kem, clean, caps_all)
                pl2, pcos = picks(R, volts(Gr, caps_all, cap_pred)); rr['voltage L2, in-sample'] = pl2 == em; rr['voltage cosine, in-sample'] = pcos == em
                if not cap_pred:
                    gains_rows += [dict(model=f'{name}: {label}', electrode=f'{(e // 2) + 1}{"+−"[e % 2]}', gain=Gr[e],
                                        n_terms=int(nr[e])) for e in range(20)]
                c2, cc, cn = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
                for train in (half, ~half):
                    ct = singular_caps(Mem, Kem, clean & train); Gt, _ = fit_gains(R, Mem, Kem, clean & train, ct)
                    a2, ac = picks(R[~train], volts(Gt, ct, cap_pred)[~train]); c2[~train] = a2 == em; cc[~train] = ac == em
                    ctn = singular_caps(Mm[:, em], Kem, clean & train); Gn, _ = fit_gains(R, Mm[:, em], Kem, clean & train, ctn)
                    cn[~train] = picks(R[~train], volts(Gn, ctn, cap_pred, with_len=False)[~train])[0] == em
                rr['voltage L2, 2-fold by time'] = c2; rr['voltage cosine, 2-fold by time'] = cc; rr['voltage L2, 2-fold, no length law'] = cn
                for rule, ok in rr.items():
                    rows.append(dict(model=f'{name}: {label}', rule=rule, **{'all EODs': 100 * ok[valid].mean(), 'clean good': 100 * ok[clean].mean()}))
        gains_rows += [dict(model=name, electrode=f'{(k // 2) + 1}{"+−"[k % 2]}', gain=G[k], n_terms=int(cnt[k])) for k in range(20)]
    T = pd.DataFrame(rows); T.to_csv(f'{out}/accuracy.csv', index=False); pd.DataFrame(gains_rows).to_csv(f'{out}/gains.csv', index=False)
    with pd.option_context('display.width', 220):
        print(T.pivot_table(index='model', columns='rule', values='clean good', sort=False).round(2).to_string())
        print('all EODs:'); print(T.pivot_table(index='model', columns='rule', values='all EODs', sort=False).round(2).to_string())
