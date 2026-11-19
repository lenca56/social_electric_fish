"""How close each model's prediction for the TRUE emitter is to the recording, 2- and 3-fish silent controls: median and mean L2 per
model and measure, on the same EODs for every model (clean good, and all EODs with Time >= 0).

Measures (true emitter only, every EOD, right or wrong pick):
  shape, peak sample      L2 of MODEL_SIGN u / max|u| vs r / max|r|, r = the pair voltages at the sample where |V| peaks (my
                          usual shape rule; d<emitter> of each run)
  shape, peak-to-peak     L2 of S u / max|u| vs PTP / max|PTP| (signed peak-to-peak, Denis' sign rule; S = +1); Denis' model: his
                          normalised predicted vs normalised recorded PTP
  voltage, peak sample    ||r - v|| / max|r|, v = a(L) x signed-electrode gain x model (voltage_rule_controls.py, gains fitted
                          in-sample on all clean-good EODs)
  voltage, peak-to-peak   the same on PTP (ptp_rule_controls.py); Denis' model: his voltage rule (original gain fit)
Voltage L2 is divided by max|r| so that it is on the scale of the shape L2 (both vectors' largest entry ~ 1).
Common EODs: those where every model has a prediction for the emitter (Denis' model has none where some fish lacks a midline).

    python l2_true_emitter_controls.py
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/voltage_rule/: l2_true_emitter.csv, and the figures
l2_true_emitter_median.png and l2_true_emitter_mean.png (both recordings in each, saved in both folders).
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eod_assignment_control_groups as g
from eod_assign import MODEL_SIGN
from io_utils import load_eods, load_assignment, load_denis_assignment

REC = {'2-fish': dict(res='results/eod_ground_truth', emitter=1, letters={'A': 1, 'B': 0}, n=2),
       '3-fish': dict(res='results/eod_ground_truth_3fish', emitter=0, letters={'A': 1, 'B': 2, 'C': 0}, n=3)}
MODELS = {'Nov 5: N150, square': ('{res}/conductivities/my_model_shape_synch_fix_newest.csv', 'Nov_5'),
          'N200 first tank, no electrodes': ('{out}/N200_tank_none/eod_assignment_shape.csv', 'N200_first_tank_no_electrodes'),
          'N150 enlarged tank, no electrodes': ('{out}/N150_tank_enlarged_none/eod_assignment_shape.csv', 'N150_enlarged_tank_no_electrodes'),
          'N150 enlarged tank, discs r 0.5 cm': ('{out}/N150_tank_enlarged_r0.5_s1e7/eod_assignment_shape.csv', 'N150_enlarged_tank_discs_r_0.5_cm'),
          'N200 tank, no fish bodies': ('{out}/N200_tank_nobodies_none/eod_assignment_shape.csv', 'N200_tank_no_fish_bodies'),
          'N150 square, no fish bodies': ('{out}/N150_image_nobodies_none/eod_assignment_shape.csv', 'N150_square_no_fish_bodies'),
          'Denis analytic, my mask': (None, 'Denis_analytic_my_mask')}
MEAS = {'shape, peak sample': '#8a8a85', 'shape, peak-to-peak': '#eda100', 'voltage, peak sample': '#2a78d6',
        'voltage, peak-to-peak': '#eb6834'}
NL = '\n'
SHORT = {'Nov 5: N150, square': 'Nov 5' + NL + 'N150 square', 'N200 first tank, no electrodes': 'N200' + NL + 'first tank',
         'N150 enlarged tank, no electrodes': 'N150' + NL + 'enlarged tank', 'N150 enlarged tank, discs r 0.5 cm': 'N150 enlarged' + NL + '+ discs 0.5 cm',
         'N200 tank, no fish bodies': 'N200 tank,' + NL + 'no fish bodies', 'N150 square, no fish bodies': 'N150 square,' + NL + 'no fish bodies',
         'Denis analytic, my mask': 'Denis' + NL + 'analytic'}
unit = lambda x: x / np.nanmax(np.abs(x), axis=-1, keepdims=True)

rows, L2 = [], {}
for rec, r in REC.items():
    c = g.REC[rec]; em, nf = r['emitter'], r['n']; out = f"{c['out']}/voltage_rule"
    q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv"); clean = q.clean_good.to_numpy(bool)
    valid = load_eods(c['newest'], columns=['Time'], traces=False)[0].Time.to_numpy() >= 0
    Rptp = np.load(f'{out}/recorded_ptp.npz')['recorded_ptp']
    vals = {}
    for name, (path, tag) in MODELS.items():
        v = {}
        if path is not None:
            a = load_assignment(path.format(res=r['res'], out=c['out'])); assert a.pair_ids == list(range(1, 11))
            v['shape, peak sample'] = a.score[:, em]
            u = a.predicted[:, em]
            zp = np.load(f'{out}/predicted_voltages_ptp_{tag}.npz'); S = float(zp['sign'])
            v['shape, peak-to-peak'] = np.linalg.norm(unit(S * u) - unit(Rptp), axis=1)
            z = np.load(f'{out}/predicted_voltages_{tag}.npz'); R = z['recorded']
            v['voltage, peak sample'] = np.linalg.norm(R - z['predicted_volts'][:, em], axis=1) / np.abs(R).max(1)
            v['voltage, peak-to-peak'] = np.linalg.norm(Rptp - zp['predicted_volts'][:, em], axis=1) / np.abs(Rptp).max(1)
        else:                                                                  # Denis: peak-to-peak only
            ddir = f"{r['res']}/conductivities/denis_mask_midline_offset"
            dn = load_denis_assignment(f'{ddir}/denis_mask_midline_all_eods_synch_fix_newest.csv', f'{ddir}/synch_fix_newest', fish=r['letters'])
            has = dn.pick >= 0; sh = np.full(len(q), np.nan)
            sh[has] = np.linalg.norm(unit(dn.predicted[has, em]) - unit(dn.recorded[has]), axis=1)
            v['shape, peak-to-peak'] = sh
            z = np.load(f'{out}/predicted_voltages_{tag}.npz'); R = z['recorded']
            v['voltage, peak-to-peak'] = np.linalg.norm(R - z['predicted_volts'][:, em], axis=1) / np.abs(R).max(1)
        vals[name] = v
    common = np.logical_and.reduce([np.isfinite(a) for v in vals.values() for a in v.values()])
    for grp, sel in (('clean good', clean & common), ('all EODs', valid & common)):
        for name, v in vals.items():
            for m in MEAS:
                a = v.get(m)
                if a is None: continue
                x = a[sel]
                rows.append(dict(recording=rec, group=grp, model=name, measure=m, n=int(sel.sum()), median=np.median(x),
                                 mean=x.mean(), sem=x.std(ddof=1) / np.sqrt(len(x)), q25=np.percentile(x, 25), q75=np.percentile(x, 75)))
    print(f'{rec}: common EODs: clean good {int((clean & common).sum())} of {int(clean.sum())}, all {int((valid & common).sum())} of {int(valid.sum())}')
T = pd.DataFrame(rows)
for rec in REC: T[T.recording == rec].to_csv(f"{g.REC[rec]['out']}/voltage_rule/l2_true_emitter.csv", index=False)

for stat in ('median', 'mean'):
    fig, axs = plt.subplots(2, 2, figsize=(26, 12))
    for col, rec in enumerate(REC):
        for row, grp in enumerate(('clean good', 'all EODs')):
            a = axs[row, col]; S_ = T[(T.recording == rec) & (T.group == grp)]; models = list(MODELS); x = np.arange(len(models))
            w = 0.8 / len(MEAS); labels = []; top = 0
            for j, (m, colr) in enumerate(MEAS.items()):
                y = np.array([S_[(S_.model == mo) & (S_.measure == m)][stat].iloc[0] if ((S_.model == mo) & (S_.measure == m)).any()
                              else np.nan for mo in models])
                if stat == 'mean':
                    e = np.array([S_[(S_.model == mo) & (S_.measure == m)]['sem'].iloc[0] * 1.96 if ((S_.model == mo) & (S_.measure == m)).any()
                                  else np.nan for mo in models])
                else:
                    lo = np.array([S_[(S_.model == mo) & (S_.measure == m)]['q25'].iloc[0] if ((S_.model == mo) & (S_.measure == m)).any() else np.nan for mo in models])
                    hi = np.array([S_[(S_.model == mo) & (S_.measure == m)]['q75'].iloc[0] if ((S_.model == mo) & (S_.measure == m)).any() else np.nan for mo in models])
                xo = x + (j - (len(MEAS) - 1) / 2) * w
                a.bar(xo, y, w * 0.92, color=colr, label=m)
                if stat == 'mean':
                    a.errorbar(xo, y, yerr=e, fmt='none', ecolor='0.35', elinewidth=0.8, capsize=1.5); tops = y + e
                else:
                    tops = y
                top = max(top, np.nanmax(tops)); labels.append((xo, y, tops, j))
            step = 0.035 * top * 1.2
            for xo, y, tops, j in labels:
                for xi, v, t in zip(xo, y, tops):
                    if np.isfinite(v): a.text(xi, t + 0.01 * top + (j % 2) * step, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, color='0.2')
            a.set_ylim(0, top * 1.2)
            a.set_xticks(x, [SHORT[m] for m in models], fontsize=8.5)
            a.set_ylabel(f'{stat} L2 of the true emitter (lower = closer)')
            a.set_title(f"{rec}, {grp} (n = {S_.n.iloc[0]:,} EODs, the same for every model)", fontsize=10.5)
            a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
            for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    axs[0, 0].legend(fontsize=9, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.18))
    fig.suptitle(f"{stat.capitalize()} L2 between the true emitter's prediction and the recording (shape: both scaled to max 1; voltage: "
                 "||r - v|| / max|r|, gains fitted in-sample on clean good)", fontsize=11.5)
    fig.text(0.5, 0.005, ('whiskers: 95 % interval of the mean' if stat == 'mean' else 'labels: medians') +
             "; Denis' model has peak-to-peak measures only (his recorded value).", ha='center', fontsize=8.5, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for rec in REC: fig.savefig(f"{g.REC[rec]['out']}/voltage_rule/l2_true_emitter_{stat}.png", dpi=120)
    plt.close(fig)
with pd.option_context('display.width', 220):
    print(T[T.group == 'clean good'].pivot_table(index=['recording', 'model'], columns='measure', values='median', sort=False).round(3).to_string())
