"""Does it matter which recorded value the rules compare the models with? Main phase (all 10 pairs at the sample where |V| peaks;
what my numerical model uses) vs signed peak-to-peak (what Denis' analytic model uses), for my numerical model at interior 4 x water and
the analytic model, final masks: % of EODs given to the emitter per rule, all vs good EODs.

Runs first (each writes its own folder):
    RECORDED=main_phase ONLY_SETTINGS=mine_int4,denis ONLY_STAGES=final python kaden_3fish_assign_rules.py
    RECORDED=ptp        ONLY_SETTINGS=mine_int4,denis ONLY_STAGES=final python kaden_3fish_assign_rules.py
then
    python kaden_3fish_recorded_measure_plot.py
Output: results/eod_assignment_3_fish_silent/mask_stages/accuracy_main_phase_vs_ptp_final.png and l2_main_phase_vs_ptp_final.png
(the L2 distance of the emitter, as in kaden_3fish_l2_plots.py) (bar and label = median over the 8
recordings, whiskers = 25th-75th percentile).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_kaden_3fish_common as C
from kaden_3fish_assign_rules import RULES

STAGE = 'final'
BARS = [('mine_int4', 'main_phase', 'numerical σ = 4, main phase (its own)', '#256abf'),
        ('mine_int4', 'ptp', 'numerical σ = 4, peak-to-peak', '#9ec5f4'),
        ('denis', 'main_phase', 'analytic, main phase', '#f5b08f'),
        ('denis', 'ptp', 'analytic, peak-to-peak (its own)', '#eb6834')]
RULE = {'shape_l2': 'shape L2', 'cosine': 'cosine', 'voltage_l2': 'voltage L2'}

A = pd.concat([pd.read_csv(f'{C.OUT}/assignment_recorded_{m}/accuracy.csv').assign(measure=m) for m in ('main_phase', 'ptp')])
A = A[(A.stage == STAGE) & A.subset.isin(['all', 'good'])]
Q = A.groupby(['subset', 'setting', 'measure', 'rule']).accuracy.quantile([0.25, 0.5, 0.75]).unstack()
print(Q[0.5].unstack('rule').round(2).to_string())
subs = ['all', 'good']; w = 0.8 / len(BARS)
fig, axs = plt.subplots(len(subs), 1, figsize=(11, 4.0 * len(subs)), squeeze=False)
for ax, sub in zip(axs[:, 0], subs):
    lo = np.floor(Q.loc[sub][0.25].min() - 0.5)
    for k, (st, m, lab, col) in enumerate(BARS):
        for i, r in enumerate(RULES):
            a, med, b = Q.loc[(sub, st, m, r)]; x = i + (k - (len(BARS) - 1) / 2) * w
            ax.bar(x, med - lo, w * 0.9, bottom=lo, color=col, edgecolor='white', label=lab if i == 0 else None)
            ax.errorbar(x, med, yerr=[[med - a], [b - med]], color='#33332f', elinewidth=1, capsize=2.5)
            ax.text(x, b + 0.025 * (100.3 - lo), f'{med:.1f}', ha='center', va='bottom', fontsize=8.5, color='#33332f')
    ax.set_xticks(range(len(RULES))); ax.set_xticklabels([RULE[r] for r in RULES], fontsize=10); ax.set_ylim(lo, 100.3)
    ax.set_ylabel('% given to the emitter (right fish)')
    n = int(A[(A.subset == sub) & (A.setting == 'mine_int4') & (A.measure == 'ptp')].groupby('video').n.max().sum())
    ax.set_title(f'{sub} EODs ({n:,}, {A.video.nunique()} recordings; y axis starts at {lo:.0f}%)', fontsize=10, loc='left')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, fontsize=9.5, frameon=False, loc='lower center')
fig.suptitle(f'3-fish, {STAGE} masks: recorded main phase vs peak-to-peak as what the models are compared with, % of EODs given to the '
             'right fish\n(bar and label = median over recordings, whiskers = 25th-75th percentile)', fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0.07, 1, 1)); path = f'{C.OUT}/accuracy_main_phase_vs_ptp_{STAGE}.png'; fig.savefig(path, dpi=110)
print('written', path)


# ---------------------------------------------------------------------------- the same comparison as L2 of the emitter's prediction
# shape L2 and cosine (as L2, sqrt(2 - 2 cos)) as in kaden_3fish_l2_plots.py; voltage L2 in volts of that recorded measure (peak-to-peak
# is ~1.6 x the main phase, so its volts are larger by construction). Per recording the median over its EODs; bar = median over recordings.
from kaden_3fish_assign_rules import GOOD_FRAMES, DOUBLES
G = pd.read_csv(GOOD_FRAMES); DD = pd.read_csv(DOUBLES); rows = []
for m in ('main_phase', 'ptp'):
    D = f'{C.OUT}/assignment_recorded_{m}'; EM = pd.read_csv(f'{D}/emitter.csv').set_index('video').emitter
    for st in ('mine_int4', 'denis'):
        for v in C.VIDEOS:
            d = pd.read_csv(f'{D}/picks/{st}/{v}/{STAGE}.csv'); e = EM[v]
            L = {'shape_l2': d[f'shape_l2_{e}'], 'cosine': np.sqrt(np.clip(2 - 2 * d[f'cosine_{e}'], 0, None)), 'voltage_l2': d[f'voltage_l2_{e}']}
            single = ~d.eod_row.isin(DD.eod_row[DD.video == v])
            for sub, k in (('all', (d.Frame >= 0) & single), ('good', d.Frame.isin(G.frame[G.video == v]) & (d.Frame >= 0) & single)):
                for r in RULES:
                    x = np.asarray(L[r])[k.to_numpy()]; x = x[np.isfinite(x)]
                    rows.append(dict(measure=m, setting=st, video=v, subset=sub, rule=r, median_l2=np.median(x)))
T = pd.DataFrame(rows); QL = T.groupby(['subset', 'rule', 'setting', 'measure']).median_l2.quantile([0.25, 0.5, 0.75]).unstack()
print(QL[0.5].unstack('rule').round(4).to_string())
TITLE = {'shape_l2': 'shape L2', 'cosine': 'cosine (as L2)', 'voltage_l2': 'voltage L2 (V of that recorded measure)'}
fig, axs = plt.subplots(len(subs), len(RULES), figsize=(4.4 * len(RULES), 3.9 * len(subs)), squeeze=False)
for i, sub in enumerate(subs):
    for j, r in enumerate(RULES):
        ax = axs[i, j]; top = max(QL.loc[(sub, r, st, m), 0.75] for st, m, _, _ in BARS)
        for k, (st, m, lab, col) in enumerate(BARS):
            a, med, b = QL.loc[(sub, r, st, m)]
            ax.bar(k, med, 0.82, color=col, edgecolor='white', label=lab)
            ax.errorbar(k, med, yerr=[[med - a], [b - med]], color='#33332f', elinewidth=1, capsize=2.5)
            ax.text(k, b + 0.02 * top, f'{med:.3g}', ha='center', va='bottom', fontsize=8.5, color='#33332f')
        ax.set_xticks([]); ax.set_ylim(0, top * 1.2); ax.set_title(f'{TITLE[r]}\n{sub} EODs', fontsize=9.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
        if j == 0: ax.set_ylabel('L2 of the emitter (lower = better)')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, fontsize=9.5, frameon=False, loc='lower center')
fig.suptitle(f'3-fish, {STAGE} masks: recorded main phase vs peak-to-peak, L2 distance between the emitter\'s prediction and the recording\n'
             '(bar and label = median over the 8 recordings of each recording\'s median over its EODs, whiskers = 25th-75th percentile)',
             fontsize=10.5, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0.08, 1, 1)); path = f'{C.OUT}/l2_main_phase_vs_ptp_{STAGE}.png'; fig.savefig(path, dpi=110)
print('written', path)
