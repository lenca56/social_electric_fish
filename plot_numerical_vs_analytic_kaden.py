"""Numerical (my 2D model) vs analytic (Denis) model, shape only vs voltage prediction, 2- and 3-fish silent controls with Kaden's
masks: % correct on all EODs in the video and on good-mask EODs, from with_Kaden_tracking/strategies/accuracy.csv
(assignment_strategies_kaden.py).

Numerical model: N150 measured tank, other fish with skin + interior 2 x water (the best Kaden-mask run; tag N150_tank_int2_none).
Both models on the same recorded quantity and metric: the signed peak-to-peak of the 10 pairs (Denis' sign rule), L2.
  shape    both scaled to max |v| = 1, smallest L2 wins
  voltage  a(L) x 20 signed-electrode gains x model, gains fitted on good-mask EODs of one half of the recording and scored on the
           other half (2-fold by time), smallest L2 in volts wins
An EOD without a pick (no midline for a fish) counts as wrong.

    python plot_numerical_vs_analytic_kaden.py
Output: results/eod_assignment_{2,3}_fish_silent/with_Kaden_tracking/strategies/numerical_vs_analytic.png (the same figure in both).
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RECS = {'2-fish': 'results/eod_assignment_2_fish_silent/with_Kaden_tracking',
        '3-fish': 'results/eod_assignment_3_fish_silent/with_Kaden_tracking'}
MODELS = {'Numerical\n(N150 tank, interior 2)': 'N150 tank, interior 2', 'Analytic\n(Denis)': 'Denis analytic'}
RULES = {'shape only': ('shape L2, peak-to-peak', '#2a78d6'), 'voltage prediction': ('voltage L2, peak-to-peak', '#eb6834')}
GROUPS = {'all EODs': 'all EODs', 'good-mask EODs': 'good mask EODs'}


def wilson(p, n, z=1.96):
    den = 1 + z ** 2 / n; mid = (p + z ** 2 / (2 * n)) / den; hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - hw, mid + hw


fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for r, (rec, base) in enumerate(RECS.items()):
    T = pd.read_csv(f'{base}/strategies/accuracy.csv'); q = pd.read_csv(f'{base}/eod_mask_qc.csv')
    N = {'all EODs': int(q.in_video.sum()), 'good-mask EODs': int(q.mask_qc_good.sum())}
    for c, (gname, col) in enumerate(GROUPS.items()):
        a = axs[r, c]; n = N[gname]; x = np.arange(len(MODELS)); w = 0.36; ys = []
        for j, (rname, (rule, colr)) in enumerate(RULES.items()):
            y = np.array([T[(T.model == m) & (T.rule == rule)][col].iloc[0] for m in MODELS.values()])
            lo, hi = wilson(y / 100, n); xo = x + (j - 0.5) * (w + 0.03)
            a.bar(xo, y, w, color=colr, label=rname, zorder=2)
            a.errorbar(xo, y, yerr=[y - 100 * lo, 100 * hi - y], fmt='none', ecolor='0.3', elinewidth=0.9, capsize=3, zorder=3)
            for xi, yi, h in zip(xo, y, 100 * hi): a.text(xi, h + 0.03 * (100 - (y.min() - 1)), f'{yi:.2f}', ha='center', va='bottom', fontsize=10, color='0.15')
            ys += list(y)
        low = np.floor(min(ys) - 1.5); a.set_ylim(low, 100 + 0.12 * (100 - low))
        a.set_xticks(x, list(MODELS), fontsize=10); a.set_ylabel(f'% correct (axis starts at {low:g})')
        a.set_title(f'{rec}, {gname} (n = {n:,})', fontsize=11.5)
        a.grid(axis='y', alpha=0.3, zorder=0)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
axs[0, 0].legend(frameon=False, fontsize=10, loc='upper left', ncol=2, bbox_to_anchor=(0, 1.2))
fig.suptitle("Numerical vs analytic model on Kaden's masks: shape only vs voltage prediction", fontsize=13)
fig.text(0.5, 0.01, 'Both models on the signed peak-to-peak of the 10 pairs, smallest L2 wins. Voltage: fish-length law x 20 signed-electrode '
         'gains,\nfitted on good-mask EODs of one half of the recording, scored on the other. Whiskers: 95 % Wilson interval. '
         'All EODs: Time >= 0; an EOD without a midline counts as wrong.', ha='center', fontsize=8.5, color='0.35')
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
for base in RECS.values(): fig.savefig(f'{base}/strategies/numerical_vs_analytic.png', dpi=140)
