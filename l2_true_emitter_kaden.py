"""How close each model's prediction for the TRUE emitter is to the recording, 2- and 3-fish silent controls with Kaden's masks
(the models of assignment_strategies_kaden.py): median and mean L2 per model and measure, on the same EODs for every model
(all EODs in the video, and good-mask EODs).

Measures (true emitter only, every EOD, right or wrong pick), as in l2_true_emitter_controls.py:
  shape, peak sample      L2 of MODEL_SIGN u / max|u| vs r / max|r| at the sample where |V| peaks (d<emitter> of each run)
  shape, peak-to-peak     L2 of u / max|u| vs PTP / max|PTP| (signed peak-to-peak, Denis' sign rule, S = +1); Denis' model: his
                          normalised predicted vs normalised recorded PTP
  voltage, peak sample    ||r - v|| / max|r|, v = a(L) x signed-electrode gain x model (gains fitted in-sample on all good-mask EODs,
                          strategies/predicted_voltages_*.npz)
  voltage, peak-to-peak   the same on PTP; Denis' model: his PTP with the same law x gains
Voltage L2 is divided by max|r| so that it is on the scale of the shape L2.
Common EODs: those where every model has a prediction for the emitter (none where the emitter, or for Denis any fish, has no midline).

    python l2_true_emitter_kaden.py [skin]     ('skin': the skin sweep of assignment_strategies_kaden.py, outputs with _skin)
Output in results/eod_assignment_<n>_fish_silent/with_Kaden_tracking/strategies/: l2_true_emitter.csv and
l2_true_emitter_{median,mean}_<n>fish.png.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io_utils import load_assignment, load_denis_assignment

src = open('ptp_rule_controls.py').read()
exec(src[:src.index('\nfor rec, r in REC.items():')])                 # REC, g, recorded_ptp, MODEL_SIGN
KM = open('assignment_strategies_kaden.py').read()
exec(KM[KM.index('KMODELS = '):KM.index('PLOT_RULES = ')])            # KMODELS (and the skin set, argv 'skin'), SUF

MEAS = {'shape, peak sample': '#8a8a85', 'shape, peak-to-peak': '#eda100', 'voltage, peak sample': '#2a78d6',
        'voltage, peak-to-peak': '#eb6834'}
LETTERS = {'2-fish': {'A': 1, 'B': 0}, '3-fish': {'A': 1, 'B': 2, 'C': 0}}
unit = lambda x: x / np.nanmax(np.abs(x), axis=-1, keepdims=True)
fname = lambda name: name.replace(',', '').replace(' ', '_')

for rec, r in REC.items():
    c = g.REC[rec]; em = r['emitter']; base = c['out_kaden']; out = f'{base}/strategies'
    qc = pd.read_csv(f'{base}/eod_mask_qc.csv'); valid = qc.in_video.to_numpy(bool); good = qc.mask_qc_good.to_numpy(bool)
    Rptp = recorded_ptp(c['newest'], list(range(1, 11)))
    vals = {}
    for name, (tag, _) in KMODELS.items():
        a = load_assignment(f'{base}/{tag}/eod_assignment_shape.csv'); assert a.pair_ids == list(range(1, 11)); u = a.predicted[:, em]
        zp = np.load(f'{out}/predicted_voltages_{fname(name)}_ptp.npz'); z = np.load(f'{out}/predicted_voltages_{fname(name)}_peak.npz')
        R = z['recorded']
        vals[name] = {'shape, peak sample': a.score[:, em],
                      'shape, peak-to-peak': np.linalg.norm(unit(u) - unit(Rptp), axis=1),
                      'voltage, peak sample': np.linalg.norm(R - z['predicted_volts'][:, em], axis=1) / np.abs(R).max(1),
                      'voltage, peak-to-peak': np.linalg.norm(Rptp - zp['predicted_volts'][:, em], axis=1) / np.abs(Rptp).max(1)}
    dn = load_denis_assignment(f'{base}/denis_mask_midline_all_eods.csv', f'{base}/denis', fish=LETTERS[rec])
    has = dn.pick >= 0; sh = np.full(len(qc), np.nan)
    sh[has] = np.linalg.norm(unit(dn.predicted[has, em]) - unit(dn.recorded[has]), axis=1)
    z = np.load(f'{out}/predicted_voltages_Denis_analytic_ptp.npz'); R = z['recorded']
    vals['Denis analytic'] = {'shape, peak-to-peak': sh,
                              'voltage, peak-to-peak': np.linalg.norm(R - z['predicted_volts'][:, em], axis=1) / np.abs(R).max(1)}
    common = np.logical_and.reduce([np.isfinite(a) for v in vals.values() for a in v.values()])
    rows = []
    for grp, sel in (('all EODs', valid & common), ('good mask EODs', good & common)):
        for name, v in vals.items():
            for m in MEAS:
                if m not in v: continue
                x = v[m][sel]
                rows.append(dict(group=grp, model=name, measure=m, n=int(sel.sum()), median=np.median(x), mean=x.mean(),
                                 sem=x.std(ddof=1) / np.sqrt(len(x)), q25=np.percentile(x, 25), q75=np.percentile(x, 75)))
    T = pd.DataFrame(rows); T.to_csv(f'{out}/l2_true_emitter{SUF}.csv', index=False)
    print(f'{rec}: common EODs: all {int((valid & common).sum())} of {int(valid.sum())}, good mask {int((good & common).sum())} of {int(good.sum())}')
    with pd.option_context('display.width', 220):
        for grp in ('all EODs', 'good mask EODs'):
            print(f'{rec}, {grp}, median'); print(T[T.group == grp].pivot_table(index='model', columns='measure', values='median', sort=False).round(3).to_string())

    models = list(vals); x = np.arange(len(models)); w = 0.8 / len(MEAS)
    for stat in ('median', 'mean'):
        fig, axs = plt.subplots(2, 1, figsize=(22, 12))
        for a, grp in zip(axs, ('all EODs', 'good mask EODs')):
            S_ = T[T.group == grp]; labels = []; top = 0
            get = lambda mo, m, col: (lambda s: s.iloc[0] if len(s) else np.nan)(S_[(S_.model == mo) & (S_.measure == m)][col])
            for j, (m, colr) in enumerate(MEAS.items()):
                y = np.array([get(mo, m, stat) for mo in models]); xo = x + (j - (len(MEAS) - 1) / 2) * w
                a.bar(xo, y, w * 0.92, color=colr, label=m)
                if stat == 'mean':
                    e = 1.96 * np.array([get(mo, m, 'sem') for mo in models]); tops = y + e
                    a.errorbar(xo, y, yerr=e, fmt='none', ecolor='0.35', elinewidth=0.8, capsize=1.5)
                else:
                    tops = y
                top = max(top, np.nanmax(tops)); labels.append((xo, y, tops, j))
            step = 0.035 * top * 1.2
            for xo, y, tops, j in labels:
                for xi, v, t in zip(xo, y, tops):
                    if np.isfinite(v): a.text(xi, t + 0.01 * top + (j % 2) * step, f'{v:.3f}', ha='center', va='bottom', fontsize=7, color='0.2')
            a.set_ylim(0, top * 1.2); a.set_xticks(x, [m.replace(', ', ',\n') for m in models], fontsize=9)
            a.set_ylabel(f'{stat} L2 of the true emitter (lower = closer)')
            a.set_title(f'{rec}, Kaden masks, {grp} (n = {S_.n.iloc[0]:,} EODs, the same for every model)', fontsize=11)
            a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
            for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
        axs[0].legend(fontsize=9, frameon=False, ncol=len(MEAS), loc='upper left', bbox_to_anchor=(0, 1.16))
        fig.text(0.5, 0.005, ('whiskers: 95 % interval of the mean' if stat == 'mean' else 'labels: medians') +
                 '. Shape: both scaled to max 1; voltage: ||r - v|| / max|r|, a(L) x 20 signed-electrode gains fitted in-sample on '
                 "good-mask EODs. Denis' model: peak-to-peak only. EODs where the emitter (or, for Denis, any fish) has no midline are left out.",
                 ha='center', fontsize=8.5, color='0.35')
        fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{out}/l2_true_emitter{SUF}_{stat}_{rec.replace('-', '')}.png", dpi=120); plt.close(fig)
