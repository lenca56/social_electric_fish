"""Assignment strategies on the 2- and 3-fish silent controls with Kaden's masks (efish4-track), for my numerical model in several
versions and Denis' analytic model, from the saved predicted voltages (no new field solves).

Models (runs of eod_assignment_control_groups.py --masks kaden, results in with_Kaden_tracking/<tag>/; Denis: his pipeline on
Kaden's mask midlines, denis_kaden_controls_runner.py):
  N150 square            walls at the frame edges, original electrodes, 15.2 px/cm, other fish skin + interior 100
  N200 tank              measured tank outline, updated electrodes, 14.2 px/cm, skin + interior 100
  N200 tank, no bodies   the same without the other fish
  N150 square, no bodies
  N150 tank, interior s  measured tank, other fish with the skin and an interior of s x water (s = 2, 20, 40, 100; 100 = original)
  Denis analytic         3D point currents with mirror images, frame-edge walls, peak-to-peak

Recorded value: the 10 pair voltages at the sample where |V| peaks ('peak sample'), or the signed peak-to-peak with Denis' sign
rule ('peak-to-peak'; my model uses S = +1 there, MODEL_SIGN = -1 on the peak sample; Denis' model has peak-to-peak only).
Rules: 'shape L2' (both scaled to max |v| = 1), 'shape cosine', and the voltage rule (voltage_rule_controls.py): predicted volts =
a(L) x signed-electrode gain (20, the pole nearer to the fish) x model, a(L) the 4-fish linear law of each model, L = 90th percentile
of the fish's Kaden mask-midline length (15.2 px/cm); gains by plain least squares in volts on the true emitter of good-mask EODs
of one half of the recording, scored on the other half (2-fold by time); 'voltage L2' (volts) and 'voltage cosine'; also
'voltage L2, no length law'.
Groups: all EODs in the video (Time >= 0), and good-mask EODs (kaden_mask_qc_controls.py: every fish passes the mask QC).
An EOD without a pick counts as wrong.

    python assignment_strategies_kaden.py [skin]
'skin': the skin sweep instead (N150 tank, interior 2, skin 0-3 cm of water, plus Denis); outputs get the suffix _skin.
Output in results/eod_assignment_<n>_fish_silent/with_Kaden_tracking/strategies/: accuracy.csv, gains.csv, fish_lengths.csv,
predicted_voltages_<model>_<peak|ptp>.npz (in-sample fit on all good-mask EODs), accuracy_<n>fish.png.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = open('ptp_rule_controls.py').read()
exec(src[:src.index('\nfor rec, r in REC.items():')])                 # voltage_rule_controls.py functions + recorded_ptp, shape_picks

KMODELS = {'N150 square': ('N150_image_none', 'electrodes_signed_px_coord'),
           'N200 tank': ('N200_tank_none', 'electrode_locations_updated'),
           'N200 tank, no bodies': ('N200_tank_nobodies_none', 'electrode_locations_updated'),
           'N150 square, no bodies': ('N150_image_nobodies_none', 'electrodes_signed_px_coord'),
           **{f'N150 tank, interior {s}': (f'N150_tank_int{s}_none', 'electrode_locations_updated') for s in (2, 20, 40, 100)}}
SET = sys.argv[1] if len(sys.argv) > 1 else 'main'; SUF = '' if SET == 'main' else f'_{SET}'
if SET == 'skin':   # N150 tank, interior 2, skin as resistive as l cm of water (0.1 = the run of the interior sweep)
    KMODELS = {f'N150 tank, int 2, skin {l:g}': ('N150_tank_int2_none' if l == 0.1 else f'N150_tank_int2_skin{l:g}_none', 'electrode_locations_updated')
               for l in (0, 0.03, 0.1, 0.3, 1, 3)}
PLOT_RULES = {'shape L2, peak sample': '#8a8a85', 'shape L2, peak-to-peak': '#eda100', 'shape cosine, peak-to-peak': '#e87ba4',
              'voltage L2, peak sample': '#2a78d6', 'voltage L2, peak-to-peak': '#eb6834'}


def voltage_rules(R, Mm, A, K, em, good, half, n):
    """In-sample predicted volts (fit on all good EODs), the gains, and 2-fold-by-time correctness of L2 / cosine / L2 without law."""
    Kc = np.where(K >= 0, K, 0); Mem, Kem = A[em] * Mm[:, em], K[:, em]
    volts = lambda G_, wl=True: np.where(K >= 0, (A[None, :, None] if wl else 1.0) * np.take(G_, Kc) * Mm, np.nan)
    G, cnt = fit_gains(R, Mem, Kem, good); V = volts(G)
    l2, cs, nl = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for train in (half, ~half):
        Gt, _ = fit_gains(R, Mem, Kem, good & train); a2, ac = picks(R[~train], volts(Gt)[~train]); l2[~train] = a2 == em; cs[~train] = ac == em
        Gn, _ = fit_gains(R, Mm[:, em], Kem, good & train); nl[~train] = picks(R[~train], volts(Gn, False)[~train])[0] == em
    return V, G, cnt, l2, cs, nl


for rec, r in REC.items():
    c = g.REC[rec]; em, nf = r['emitter'], r['n']; base = c['out_kaden']; out = f'{base}/strategies'; os.makedirs(out, exist_ok=True)
    qc = pd.read_csv(f'{base}/eod_mask_qc.csv'); valid = qc.in_video.to_numpy(bool); good = qc.mask_qc_good.to_numpy(bool)
    frames = qc.Frame.to_numpy(); n = len(qc); half = np.arange(n) < n // 2
    z = np.load(f'{base}/mask_midlines_by_frame_newest.npz'); M = dict(zip(z['frames'].tolist(), z['midlines_px']))
    Lv = np.stack([np.linalg.norm(np.diff(M[int(k)], axis=1), axis=-1).sum(-1) for k in np.unique(frames)]) / PX_CM_LAW
    L = np.array([np.nanpercentile(Lv[:, f], 90) for f in range(nf)])
    pd.DataFrame({'fish (ZEUS number)': range(nf), 'length_cm (90th pct, 15.2 px/cm)': L.round(3)}).to_csv(f'{out}/fish_lengths.csv', index=False)
    Rptp = recorded_ptp(c['newest'], list(range(1, 11)))
    rows, gains_rows, Kcache = [], [], {}
    def add(model, rule, ok):
        rows.append(dict(model=model, rule=rule, **{'all EODs': 100 * ok[valid].mean(), 'good mask EODs': 100 * ok[good].mean()}))
    def nearer(el):
        if el not in Kcache:
            pairs = load_electrode_pairs(f'data/electrodes/{el}_{rec}.csv', P)
            Kcache[el] = nearer_pole(M, frames, [(pp[::-1] * (P - 1), pm[::-1] * (P - 1)) for _, pp, pm in pairs], nf)
        return Kcache[el]
    print(f'\n{rec}: fish lengths {dict(enumerate(L.round(2)))}; {good.sum()} good-mask EODs of {valid.sum()}', flush=True)
    for name, (tag, el) in KMODELS.items():
        path = f'{base}/{tag}/eod_assignment_shape.csv'
        if not os.path.exists(path): print(f'  {name}: no run yet ({path})'); continue
        d = pd.read_csv(path); assert (d.Frame.to_numpy() == frames).all()
        R = d[[f'r_raw_p{p}' for p in range(1, 11)]].to_numpy()
        U = np.stack([d[[f'u{f}_raw_p{p}' for p in range(1, 11)]].to_numpy() for f in range(nf)], axis=1)
        add(name, 'shape L2, peak sample', d.pick.to_numpy() == em); add(name, 'shape cosine, peak sample', d.pick_cosine.to_numpy() == em)
        p2, pc = shape_picks(Rptp, U); add(name, 'shape L2, peak-to-peak', p2 == em); add(name, 'shape cosine, peak-to-peak', pc == em)
        A = LAW['mine'][0] * L + LAW['mine'][1]; K = nearer(el)
        for meas, RR, Mm in (('peak sample', R, MODEL_SIGN * U), ('peak-to-peak', Rptp, U)):
            V, G, cnt, l2, cs, nl = voltage_rules(RR, Mm, A, K, em, good, half, n)
            add(name, f'voltage L2, {meas}', l2); add(name, f'voltage cosine, {meas}', cs)
            if meas == 'peak sample': add(name, 'voltage L2, peak sample, no length law', nl)
            gains_rows += [dict(model=name, measure=meas, electrode=f'{k // 2 + 1}{"+−"[k % 2]}', gain=G[k], n_terms=int(cnt[k])) for k in range(20)]
            np.savez_compressed(f"{out}/predicted_voltages_{name.replace(',', '').replace(' ', '_')}_{'peak' if meas == 'peak sample' else 'ptp'}.npz",
                                predicted_volts=V, recorded=RR, gains=G, nearer_pole=K, fish_factor=A, fish_length_cm=L)
        print(f'  {name}: done', flush=True)
    # Denis
    dpath = f'{base}/denis_mask_midline_all_eods.csv'
    if os.path.exists(dpath):
        dn = pd.read_csv(dpath); has = dn.pick.notna().to_numpy(); letters = {'2-fish': {'A': 1, 'B': 0}, '3-fish': {'A': 1, 'B': 2, 'C': 0}}[rec]
        dk = pd.read_pickle(f"{base}/denis/{next(x for x in os.listdir(f'{base}/denis') if x.startswith('Assigned') and x.endswith('.pkl'))}")
        assert has.sum() == len(dk)
        D_R = np.full((n, 10), np.nan); D_R[has] = np.stack(dk.recorded_PTP.to_numpy()); D_M = np.full((n, nf, 10), np.nan)
        for X, f in letters.items(): D_M[has, f] = np.stack(dk[f'predicted_PTP_{X}'].to_numpy())
        add('Denis analytic', 'shape cosine, peak-to-peak', dn.pick.map(letters).fillna(-1).to_numpy(int) == em)
        add('Denis analytic', 'shape L2, peak-to-peak', shape_picks(D_R, D_M)[0] == em)
        A = LAW['Denis'][0] * L + LAW['Denis'][1]
        V, G, cnt, l2, cs, nl = voltage_rules(D_R, D_M, A, nearer('electrodes_signed_px_coord'), em, good, half, n)
        add('Denis analytic', 'voltage L2, peak-to-peak', l2); add('Denis analytic', 'voltage cosine, peak-to-peak', cs)
        gains_rows += [dict(model='Denis analytic', measure='peak-to-peak', electrode=f'{k // 2 + 1}{"+−"[k % 2]}', gain=G[k], n_terms=int(cnt[k])) for k in range(20)]
        np.savez_compressed(f'{out}/predicted_voltages_Denis_analytic_ptp.npz', predicted_volts=V, recorded=D_R, gains=G, fish_factor=A, fish_length_cm=L)
    T = pd.DataFrame(rows); T.to_csv(f'{out}/accuracy{SUF}.csv', index=False); pd.DataFrame(gains_rows).to_csv(f'{out}/gains{SUF}.csv', index=False)
    with pd.option_context('display.width', 250):
        for grp in ('good mask EODs', 'all EODs'):
            print(f'{rec}, {grp}'); print(T.pivot_table(index='model', columns='rule', values=grp, sort=False).round(2).to_string())

    # figure: % correct per model (x) and rule (colour), all EODs and good-mask EODs
    models = list(dict.fromkeys(T.model)); x = np.arange(len(models)); w = 0.8 / len(PLOT_RULES)
    fig, axs = plt.subplots(2, 1, figsize=(24, 12))
    for a, grp in zip(axs, ('all EODs', 'good mask EODs')):
        nn = int(valid.sum() if grp == 'all EODs' else good.sum()); vals, labs = [], []
        for j, (rule, colr) in enumerate(PLOT_RULES.items()):
            y = np.array([(lambda v: v.iloc[0] if len(v) else np.nan)(T[(T.model == m) & (T.rule == rule)][grp]) for m in models])
            k_ = np.round((100 - np.nan_to_num(y)) / 100 * nn); p_ = k_ / nn; zz = 1.96; den = 1 + zz ** 2 / nn
            mid = (p_ + zz ** 2 / (2 * nn)) / den; hw = zz * np.sqrt(p_ * (1 - p_) / nn + zz ** 2 / (4 * nn ** 2)) / den
            lo, hi = 100 - 100 * (mid + hw), 100 - 100 * (mid - hw)
            xo = x + (j - (len(PLOT_RULES) - 1) / 2) * w
            a.bar(xo, y, w * 0.92, color=colr, label=rule)
            a.errorbar(xo, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt='none', ecolor='0.35', elinewidth=0.8, capsize=1.5)
            vals += list(y[np.isfinite(y)]); labs.append((xo, y, hi, j))
        low = np.floor((min(vals) - 0.4) * 2) / 2; top = 100 + 0.14 * (100 - low); a.set_ylim(low, top); step = 0.035 * (top - low)
        for xo, y, hi, j in labs:
            for xi, v, h in zip(xo, y, hi):
                if np.isfinite(v): a.text(xi, h + 0.01 * (top - low) + (j % 2) * step, f'{v:.2f}', ha='center', va='bottom', fontsize=6.5, color='0.2')
        a.set_xticks(x, [m.replace(', ', ',\n') for m in models], fontsize=9)
        a.set_ylabel(f'% correct (axis starts at {low:g})'); a.set_title(f'{rec}, Kaden masks, {grp} (n = {nn:,})', fontsize=11)
        a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
        for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    axs[0].legend(fontsize=9, frameon=False, ncol=len(PLOT_RULES), loc='upper left', bbox_to_anchor=(0, 1.16))
    fig.text(0.5, 0.005, 'labels: % correct; whiskers: 95 % Wilson interval. Voltage rules: a(L) x 20 signed-electrode gains, fitted on '
             'good-mask EODs of one half of the recording and scored on the other. Denis: peak-to-peak only (his own rule = shape cosine).',
             ha='center', fontsize=8.5, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{out}/accuracy{SUF}_{rec.replace('-', '')}.png", dpi=120); plt.close(fig)
