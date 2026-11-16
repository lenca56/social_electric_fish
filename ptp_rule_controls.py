"""Denis' peak-to-peak (PTP) measure applied to my numerical model, 2- and 3-fish silent controls.

Recorded value per pair = signed peak-to-peak of its 25-sample window, as Denis' loader (src/data_loader.py):
    PTP = -(max - min) x sign(argmax - argmin)      (+ when the window's maximum comes before its minimum)
instead of my usual value at the one sample where |V| peaks over all pairs. My model is resistive (every pair sees the same EOD
waveform), so a pair's PTP is proportional to its unit-current voltage u, with one sign S for the whole recording; S is taken from
the data (the sign of the median cosine of u and PTP for the true emitter on clean-good EODs) and reported.

Rules, each model of mine (the four of voltage_rule_controls.MODELS and the no-fish-bodies run):
  PTP shape L2       L2 of S u / max|u| vs PTP / max|PTP| (smallest wins)
  PTP shape cosine   cosine of S u and PTP (largest wins; Denis' rule on my model)
  PTP voltage ...    voltage rule of voltage_rule_controls.py on PTP: a(L) x 20 signed-electrode gains x S u, gains by plain least
                     squares in volts on the true emitter of clean-good EODs (in-sample, 2-fold by time, and 2-fold without the
                     length law). The length law is the one fitted on the main-phase values; a PTP scale factor common to all fish
                     is taken up by the gains.
Everything else (lengths, law, nearer pole, gain fit, picks) is executed from voltage_rule_controls.py (the code above its main
loop), so the rules are the same code.

    python ptp_rule_controls.py
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/voltage_rule/: accuracy_ptp.csv, gains_ptp.csv, recorded_ptp.npz,
predicted_voltages_ptp_<model>.npz (in-sample predicted volts, gains, nearer pole, fish factor, sign).

'all EODs' (from 2026-11-10) leaves out the few EODs with negative Time in the newest table (before the video starts; they
carry Frame 1 and are never in a clean good frame).
"""
import os, re
import numpy as np
import pandas as pd

src = open('voltage_rule_controls.py').read()
exec(src[:src.index('\nfor rec, r in REC.items():')])                 # constants and functions of voltage_rule_controls.py

MINE_MODELS = {**{k: v for k, v in MODELS.items() if v[0] == 'mine'},
               'N200 tank, no fish bodies': ('mine', '{out}/N200_tank_nobodies_none/eod_assignment_shape.csv', 'electrode_locations_updated'),
               'N150 square, no fish bodies': ('mine', '{out}/N150_image_nobodies_none/eod_assignment_shape.csv', 'electrodes_signed_px_coord')}


def recorded_ptp(path, pair_ids):
    """(n_eods, n_pairs) signed peak-to-peak of an EOD table, Denis' sign rule."""
    head = pd.read_csv(path, nrows=0).columns
    pat = re.compile(r'electrode_(\d+)_t(?:ime)?_([mp]?)(\d+)$'); cols = {}
    for col in head:
        m = pat.match(col)
        if m: cols.setdefault(int(m[1]), []).append(((-1 if m[2] == 'm' else 1) * int(m[3]), col))
    order = [col for p in pair_ids for _, col in sorted(cols[p])]
    X = pd.read_csv(path, usecols=order)[order].to_numpy(float).reshape(-1, len(pair_ids), len(cols[pair_ids[0]]))
    bad = np.isnan(X)                                                  # a few windows have missing samples: ignore those samples
    hi, lo = np.where(bad, -np.inf, X), np.where(bad, np.inf, X)
    out = -(hi.max(2) - lo.min(2)) * np.sign(hi.argmax(2) - lo.argmin(2))
    return np.where(bad.all(2), np.nan, out)


def shape_picks(R, U):
    """Picks of the shape L2 (max-normalised) and the cosine rule (no gains); -1 where no fish has a model."""
    a = U / np.abs(U).max(-1, keepdims=True); b = R / np.abs(R).max(-1, keepdims=True)
    d = np.linalg.norm(a - b[:, None], axis=-1)
    cos = (U * R[:, None]).sum(-1) / (np.linalg.norm(U, axis=-1) * np.linalg.norm(R, axis=-1)[:, None])
    ok = np.isfinite(d).any(1)
    return (np.where(ok, np.argmin(np.where(np.isfinite(d), d, np.inf), 1), -1),
            np.where(ok, np.argmax(np.where(np.isfinite(cos), cos, -np.inf), 1), -1))


for rec, r in REC.items():
    c = g.REC[rec]; em, nf = r['emitter'], r['n']; out = f"{c['out']}/voltage_rule"
    q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv"); clean = q.clean_good.to_numpy(bool); frames = q.Frame.to_numpy()
    valid = pd.read_csv(c['newest'], usecols=['Time']).Time.to_numpy() >= 0   # 'all EODs' leaves out EODs before the video starts
    n = len(q); half = np.arange(n) < n // 2
    z = np.load(c['midlines']); M = dict(zip(z['frames'].tolist(), z['midlines_px']))
    L = pd.read_csv(f'{out}/fish_lengths.csv').iloc[:, 1].to_numpy()
    A = LAW['mine'][0] * L + LAW['mine'][1]
    pair_ids = list(range(1, 11)); R = recorded_ptp(c['newest'], pair_ids)

    # check against Denis' own recorded PTP (EODs where every fish has a midline)
    dn_all = pd.read_csv(f"{r['res']}/conductivities/denis_mask_midline_offset/denis_mask_midline_all_eods_synch_fix_newest.csv")
    has = dn_all.pick.notna().to_numpy(); ddir = f"{r['res']}/conductivities/denis_mask_midline_offset/synch_fix_newest"
    dk = pd.read_pickle(f"{ddir}/{next(x for x in os.listdir(ddir) if x.endswith('.pkl'))}")
    Dn = np.stack(dk.recorded_PTP.to_numpy()); both = np.isfinite(Dn) & np.isfinite(R[has])
    print(f'\n{rec}: recorded PTP vs Denis\' recorded_PTP: max |difference| = {np.abs(Dn - R[has])[both].max():.2e} on '
          f'{both.sum()} values; NaN pairs here {int(np.isnan(R).sum())}, in his {int(np.isnan(Dn).sum())}', flush=True)
    np.savez_compressed(f'{out}/recorded_ptp.npz', frame=frames, recorded_ptp=R, pair_ids=pair_ids)

    rows, gains_rows, Kcache = [], [], {}
    for name, (kind, path, el) in MINE_MODELS.items():
        d = pd.read_csv(path.format(res=r['res'], out=c['out'])); assert (d.Frame.to_numpy() == frames).all()
        U = np.stack([d[[f'u{f}_raw_p{p}' for p in pair_ids]].to_numpy() for f in range(nf)], axis=1)
        ue = U[clean, em]; ok = np.isfinite(ue).all(1) & np.isfinite(R[clean]).all(1)
        cos_em = (ue[ok] * R[clean][ok]).sum(1) / (np.linalg.norm(ue[ok], axis=1) * np.linalg.norm(R[clean][ok], axis=1))
        S = np.sign(np.median(cos_em)); agree = 100 * (np.sign(cos_em) == S).mean()
        print(f'  {name}: PTP sign S = {S:+.0f} (MODEL_SIGN for the peak sample is {MODEL_SIGN:+d}); the true emitter\'s cosine has '
              f'that sign on {agree:.1f}% of clean-good EODs', flush=True)
        Mm = S * U
        res = {}
        p_l2, p_cos = shape_picks(R, Mm); res['PTP shape L2'] = p_l2 == em; res['PTP shape cosine'] = p_cos == em
        if el not in Kcache:
            pairs = load_electrode_pairs(f'data/electrodes/{el}_{rec}.csv', P)
            Kcache[el] = nearer_pole(M, frames, [(pp[::-1] * (P - 1), pm[::-1] * (P - 1)) for _, pp, pm in pairs], nf)
        K = Kcache[el]; Kc = np.where(K >= 0, K, 0); Mem, Kem = A[em] * Mm[:, em], K[:, em]
        volts = lambda G_, with_len=True: (lambda V_: np.where(K >= 0, V_, np.nan))((A[None, :, None] if with_len else 1.0) * np.take(G_, Kc) * Mm)
        G, cnt = fit_gains(R, Mem, Kem, clean)
        np.savez_compressed(f"{out}/predicted_voltages_ptp_{name.split(':')[0].replace(',', '').replace(' ', '_')}.npz",
                            predicted_volts=volts(G), recorded_ptp=R, gains=G, nearer_pole=K, fish_factor=A, sign=S)
        a2, ac = picks(R, volts(G)); res['PTP voltage L2, in-sample'] = a2 == em; res['PTP voltage cosine, in-sample'] = ac == em
        gains_rows += [dict(model=name, electrode=f'{(k // 2) + 1}{"+−"[k % 2]}', gain=G[k], n_terms=int(cnt[k])) for k in range(20)]
        c2, cc, cn = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        for train in (half, ~half):
            Gt, _ = fit_gains(R, Mem, Kem, clean & train)
            a2, ac = picks(R[~train], volts(Gt)[~train]); c2[~train] = a2 == em; cc[~train] = ac == em
            Gn, _ = fit_gains(R, Mm[:, em], Kem, clean & train)
            cn[~train] = picks(R[~train], volts(Gn, with_len=False)[~train])[0] == em
        res['PTP voltage L2, 2-fold by time'] = c2; res['PTP voltage cosine, 2-fold by time'] = cc
        res['PTP voltage L2, 2-fold, no length law'] = cn
        for rule, ok_ in res.items():
            rows.append(dict(model=name, rule=rule, **{'all EODs': 100 * ok_[valid].mean(), 'clean good': 100 * ok_[clean].mean()}))
    T = pd.DataFrame(rows); T.to_csv(f'{out}/accuracy_ptp.csv', index=False); pd.DataFrame(gains_rows).to_csv(f'{out}/gains_ptp.csv', index=False)
    with pd.option_context('display.width', 250):
        print(T.pivot_table(index='model', columns='rule', values='clean good', sort=False).round(2).to_string())
