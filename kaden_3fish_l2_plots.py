"""How far each model's prediction for the true emitter is from the recording (an L2 norm per EOD, lower = better), per rule and model
setting, for the 3-fish recordings (final masks), from the outputs of kaden_3fish_assign_rules.py: all EODs vs good-tracking EODs.

L2 of the emitter's prediction, one per rule (r = recorded 10 pairs, p = model_sign x model_raw, V = a_f e_k p, the rule's own scaling):
  shape L2        || r / max|r| - p / max|p| ||                 (the shape_l2 score)
  cosine          || r / ||r|| - p / ||p|| || = sqrt(2 - 2 cos) (the cosine rule as an L2 norm)
  voltage L2      || r - V ||  in volts                        (the voltage_l2 score; recorded r is my model's main phase but
                                                                 Denis' peak-to-peak, so the two models' volts differ in what they measure)
  voltage cosine  sqrt(2 - 2 cos(r, V))
Per recording: the median over its EODs; bars = median over the recordings, whiskers = 25th-75th percentile over the recordings.
EOD sets: all = every EOD with a prediction for the emitter; good = those on mask-QC good frames (kaden_3fish_mask_qc.py); both
without the double-detected EODs (eod_double_detections_3fish.py).

    python kaden_3fish_l2_plots.py
Output in results/eod_assignment_3_fish_silent/mask_stages/assignment/: l2_emitter_per_recording_final.csv,
l2_voltage_rules_final.png (voltage L2, voltage cosine) and l2_normalized_rules_final.png (shape L2, cosine); every model setting.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_kaden_3fish_common as C
from kaden_3fish_assign_rules import RULES, GOOD_FRAMES, DOUBLES

STAGE = 'final'
OUT = f'{C.OUT}/assignment'
SETTINGS = ['mine_int1', 'mine_int4', 'mine_int16', 'mine_int100', 'denis']
NAME = {'mine_int1': 'numerical, σ conductivity = 1', 'mine_int4': 'numerical, σ conductivity = 4',
        'mine_int16': 'numerical, σ conductivity = 16', 'mine_int100': 'numerical, σ conductivity = 100', 'denis': 'analytic'}
COL = {'mine_int1': '#9ec5f4', 'mine_int4': '#5598e7', 'mine_int16': '#256abf', 'mine_int100': '#104281', 'denis': '#eb6834'}
RULE = {'shape_l2': 'shape L2 (patterns scaled to max |v| = 1)', 'cosine': 'cosine, as L2 of unit-norm patterns',
        'voltage_l2': 'voltage L2 (V)', 'voltage_cosine': 'voltage cosine, as L2 of unit-norm patterns'}

EM = pd.read_csv(f'{OUT}/emitter.csv').set_index('video').emitter
G = pd.read_csv(GOOD_FRAMES); DD = pd.read_csv(DOUBLES)
rows = []
for st in SETTINGS:
    for v in C.VIDEOS:
        pk = f'{OUT}/picks/{st}/{v}/{STAGE}.csv'; pr = f'{C.PREDICTIONS.format(setting=st, video=v, stage=STAGE)}'
        if not (os.path.exists(pk) and os.path.exists(pr)): continue
        d = pd.read_csv(pk); e = EM[v]
        L = {'shape_l2': d[f'shape_l2_{e}'], 'cosine': np.sqrt(np.clip(2 - 2 * d[f'cosine_{e}'], 0, None)),
             'voltage_l2': d[f'voltage_l2_{e}'], 'voltage_cosine': np.sqrt(np.clip(2 - 2 * d[f'voltage_cosine_{e}'], 0, None))}
        single = ~d.eod_row.isin(DD.eod_row[DD.video == v])                                # double detections left out
        good = d.Frame.isin(G.frame[G.video == v]) & (d.Frame >= 0) & single
        for sub, k in (('all', (d.Frame >= 0) & single), ('good', good)):
            for r in RULES:
                x = np.asarray(L[r])[k.to_numpy()]; x = x[np.isfinite(x)]
                rows.append(dict(setting=st, video=v, subset=sub, rule=r, n=len(x), median_l2=np.median(x) if len(x) else np.nan))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/l2_emitter_per_recording_{STAGE}.csv', index=False)
Q = T.groupby(['subset', 'rule', 'setting']).median_l2.quantile([0.25, 0.5, 0.75]).unstack()
print(Q[0.5].unstack('setting').reindex(columns=SETTINGS).round(4).to_string())


def plot(rules, settings, path, title):
    subs = ['all', 'good']; w = 0.8 / len(settings)
    fig, axs = plt.subplots(len(subs), len(rules), figsize=(6.0 * len(rules), 3.7 * len(subs)), squeeze=False)
    for i, sub in enumerate(subs):
        for j, r in enumerate(rules):
            ax = axs[i, j]; q = Q.loc[(sub, r)].reindex(settings); best = q[0.5].idxmin()
            for k, st in enumerate(settings):
                a, m, b = q.loc[st]; x = (k - (len(settings) - 1) / 2) * w
                ax.bar(x, m, w * 0.9, color=COL[st], edgecolor='white', label=NAME[st])
                ax.errorbar(x, m, yerr=[[m - a], [b - m]], color='#33332f', elinewidth=1, capsize=2.5)
                ax.text(x, b + 0.02 * q[0.75].max(), f'{m:.3g}' + ('\n★' if st == best else ''), ha='center', va='bottom', fontsize=8,
                        weight='bold' if st == best else 'normal', color='#33332f')
            ax.set_xticks([]); ax.set_ylim(0, q[0.75].max() * 1.25)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
            if i == 0: ax.set_title(RULE[r], fontsize=10)
            if j == 0: ax.set_ylabel(f'{sub} EODs\nL2 of the emitter (lower = better)')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=9, frameon=False, loc='lower center')
    fig.suptitle(title + '\n(bar and label = median over the 8 recordings of each recording\'s median over its EODs, whiskers = '
                 '25th-75th percentile; ★ = lowest)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(path, dpi=110); plt.close(fig)


plot(['voltage_l2', 'voltage_cosine'], SETTINGS, f'{OUT}/l2_voltage_rules_{STAGE}.png',
     f'3-fish, {STAGE} masks, voltage rules (fish factor x electrode gain x prediction): L2 distance of the emitter\'s prediction')
plot(['shape_l2', 'cosine'], SETTINGS, f'{OUT}/l2_normalized_rules_{STAGE}.png',
     f'3-fish, {STAGE} masks, normalized rules (shape only): L2 distance of the emitter\'s prediction')
print(f'written {OUT}/l2_*_{STAGE}.png')
