"""Figures of voltage_rule_controls.py (voltage-based assignment, 2- and 3-fish silent controls).

1. voltage_rule_accuracy.png: % of EODs assigned to the right fish (value on each bar), per model (x) and rule (colour): the model's shape rule, and
   the voltage rule (a(L) x 20 signed-electrode gains) scored by L2 in volts or by cosine, and L2 without the length law; gains
   fitted 2-fold by time (out of sample). Rows: clean good EODs, all EODs; columns: 2-fish, 3-fish. Whiskers: 95 % Wilson interval.
   Reads accuracy_combined.csv when it exists (adds the no-fish-bodies model and the shape-cosine rule from
   voltage_rule_extra_controls.py); a missing model x rule (Denis has no separate shape cosine: his shape rule is a cosine) has no bar.
1b. ptp_rule_accuracy.png: the same for the peak-to-peak rules of ptp_rule_controls.py next to the peak-sample ones and Denis.
2. voltage_rule_gains.png: the fitted signed-electrode gains (in-sample fit on all clean-good EODs) per model, each model's gains
   divided by their median so the models can be compared; marker size ~ number of fitted terms (EOD x pair with that pole nearer
   to the emitter).

    python plot_voltage_rule_controls.py
Output: results/eod_assignment_<n>_fish_silent/Nov_10_2026/voltage_rule/voltage_rule_{accuracy,gains}.png (the same figure in both).
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = {'2-fish': 'results/eod_assignment_2_fish_silent/Nov_10_2026/voltage_rule',
       '3-fish': 'results/eod_assignment_3_fish_silent/Nov_10_2026/voltage_rule'}
N_EOD = {'2-fish': {'clean good': 11803, 'all EODs': 18798}, '3-fish': {'clean good': 8972, 'all EODs': 19780}}   # all: Time >= 0
RULES = {'shape (no gains)': ('shape rule (current)', '#8a8a85'),
         'voltage L2, 2-fold by time': ('voltage, L2 in volts', '#2a78d6'),
         'voltage cosine, 2-fold by time': ('voltage, cosine', '#eb6834'),
         'voltage L2, 2-fold, no length law': ('voltage, L2, no length law', '#1baf7a'),
         'shape cosine (no gains)': ('shape, cosine (no gains)', '#eda100')}
MODELS = ['Nov 5: N150, square', 'N200 first tank, no electrodes', 'N150 enlarged tank, no electrodes',
          'N150 enlarged tank, discs r 0.5 cm', 'N200 tank, no fish bodies', 'N150 square, no fish bodies', 'Denis analytic, my mask']
MCOL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#7a7a74', '#008300']
NL = '\n'


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z ** 2 / n; mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return 100 * (mid - half), 100 * (mid + half)


# 1. accuracy (% correct; the y axis starts near the lowest value, not at 0, so the differences stay visible)
def accuracy_figure(tables, models, rules, fname, title, remap=None):
    """tables: rec -> DataFrame(model, rule, all EODs, clean good); remap: model -> {rule shown: rule in the table}."""
    fig, axs = plt.subplots(2, 2, figsize=(26, 12.5))
    for col, rec in enumerate(OUT):
        T = tables[rec]
        for row, grp in enumerate(('clean good', 'all EODs')):
            a = axs[row, col]; x = np.arange(len(models)); w = 0.8 / len(rules); n = N_EOD[rec][grp]; vals = []; labels = []
            for j, (rule, (lab, c)) in enumerate(rules.items()):
                get = lambda m: (lambda v: v.iloc[0] if len(v) else np.nan)(
                    T[(T.model == m) & (T.rule == (remap or {}).get(m, {}).get(rule, rule if m not in (remap or {}) else None))][grp])
                acc = np.array([get(m) for m in models]); vals += list(acc[np.isfinite(acc)])
                k = np.round((100 - np.nan_to_num(acc)) / 100 * n); lo, hi = wilson(k, n); lo, hi = 100 - hi, 100 - lo
                xo = x + (j - (len(rules) - 1) / 2) * w
                a.bar(xo, acc, w * 0.92, color=c, label=lab)
                a.errorbar(xo, acc, yerr=[np.clip(acc - lo, 0, None), np.clip(hi - acc, 0, None)], fmt='none', ecolor='0.35',
                           elinewidth=0.8, capsize=1.5)
                labels.append((xo, acc, hi, j))
            low = np.floor((min(vals) - 0.4) * 2) / 2; top = 100 + 0.14 * (100 - low)
            a.set_ylim(low, top); step = 0.035 * (top - low)
            for xo, acc, hi, j in labels:                                     # horizontal values, every other bar one line higher
                for xi, v, h in zip(xo, acc, hi):
                    if np.isfinite(v): a.text(xi, h + 0.01 * (top - low) + (j % 2) * step, f'{v:.2f}', ha='center', va='bottom',
                                              fontsize=6.5, color='0.2')
            a.set_xticks(x, [SHORT.get(m, m) for m in models], fontsize=8.5)
            a.set_ylabel(f'% correct (axis starts at {low:g})')
            a.set_title(f'{rec}, {grp} (n = {n:,})', fontsize=10.5)
            a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
            for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    axs[0, 0].legend(fontsize=9, frameon=False, ncol=len(rules), loc='upper left', bbox_to_anchor=(0, 1.2))
    fig.suptitle(title, fontsize=11.5)
    fig.text(0.5, 0.005, 'labels: % correct; whiskers: 95 % Wilson interval. "all EODs" leaves out the EODs before the video starts '
             '(negative Time). Voltage rules: a(L) from fish_factor_vs_length.png x 20 signed-electrode gains, fitted on one half of '
             'the recording and scored on the other.', ha='center', fontsize=8.5, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for d in OUT.values(): fig.savefig(f'{d}/{fname}', dpi=125)
    plt.close(fig)


SHORT = {'Nov 5: N150, square': 'Nov 5' + NL + 'N150 square', 'N200 first tank, no electrodes': 'N200' + NL + 'first tank',
         'N150 enlarged tank, no electrodes': 'N150' + NL + 'enlarged tank', 'N150 enlarged tank, discs r 0.5 cm': 'N150 enlarged' + NL + '+ discs 0.5 cm',
         'N200 tank, no fish bodies': 'N200 tank,' + NL + 'no fish bodies', 'N150 square, no fish bodies': 'N150 square,' + NL + 'no fish bodies',
         'Denis analytic, my mask': 'Denis' + NL + 'analytic'}
TAB = {rec: pd.read_csv(f'{OUT[rec]}/accuracy_combined.csv' if os.path.exists(f'{OUT[rec]}/accuracy_combined.csv') else f'{OUT[rec]}/accuracy.csv')
       for rec in OUT}
accuracy_figure(TAB, MODELS, RULES, 'voltage_rule_accuracy.png',
                'Voltage rule: predicted volts = a(L) x signed-electrode gain (20, the pole nearer the fish) x model (recorded value: '
                'main-phase peak sample for mine, peak-to-peak for Denis)')

# 1b. peak-to-peak for my model (ptp_rule_controls.py) next to the peak-sample rules and Denis' model
if all(os.path.exists(f'{OUT[rec]}/accuracy_ptp.csv') for rec in OUT):
    TABP = {rec: pd.concat([TAB[rec], pd.read_csv(f'{OUT[rec]}/accuracy_ptp.csv')]) for rec in OUT}
    RULES_PTP = {'shape (no gains)': ('shape L2, peak sample (current)', '#8a8a85'),
                 'PTP shape L2': ('shape L2, peak-to-peak', '#eda100'),
                 'PTP shape cosine': ('shape cosine, peak-to-peak', '#e87ba4'),
                 'voltage L2, 2-fold by time': ('voltage L2, peak sample', '#2a78d6'),
                 'PTP voltage L2, 2-fold by time': ('voltage L2, peak-to-peak', '#eb6834')}
    DENIS = {'Denis analytic, my mask': {'PTP shape cosine': 'shape (no gains)', 'PTP voltage L2, 2-fold by time': 'voltage L2, 2-fold by time'}}
    accuracy_figure(TABP, MODELS, RULES_PTP, 'ptp_rule_accuracy.png',
                    "Peak sample vs peak-to-peak (Denis' signed PTP) as the recorded value, shape and voltage rules; Denis' model uses "
                    'peak-to-peak only (his shape rule = cosine)', remap=DENIS)
    print('saved ptp_rule_accuracy.png')


# 2. gains
fig, axs = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True)
for a, rec in zip(axs, OUT):
    G = pd.concat([pd.read_csv(f'{OUT[rec]}/gains.csv')] +
                  ([pd.read_csv(f'{OUT[rec]}/gains_extra.csv')] if os.path.exists(f'{OUT[rec]}/gains_extra.csv') else [])); els = list(dict.fromkeys(G.electrode)); x = np.arange(len(els))
    for j, (m, c) in enumerate(zip(MODELS, MCOL)):
        lab_m = m
        if m == 'Denis analytic, my mask' and (G.model == m + ': fit without near-singular').any():
            m = m + ': fit without near-singular'; lab_m = m                   # Denis' gains without his 1/r spikes
        s = G[G.model == m].set_index('electrode').reindex(els)
        rel = s.gain / np.median(s.gain[s.n_terms > 0])
        xo = x + (j - (len(MODELS) - 1) / 2) * 0.12
        a.plot(xo, rel, '-', color=c, lw=1, alpha=0.6)
        a.scatter(xo, rel, s=12 + 60 * np.sqrt(s.n_terms / s.n_terms.max()), color=c, label=lab_m, zorder=3, edgecolors='w', linewidths=0.5)
    a.axhline(1, color='0.6', lw=0.8); a.axhline(0, color='0.75', lw=0.6)
    a.set_ylabel('gain / median gain of the model'); a.set_title(f'{rec}: fitted signed-electrode gains (in-sample, clean good)', fontsize=10.5)
    a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
axs[1].set_xticks(x, els); axs[1].set_xlabel('signed electrode (pair, pole)')
axs[0].legend(fontsize=8.5, frameon=False, ncol=6, loc='upper left', bbox_to_anchor=(0, 1.22))
fig.text(0.5, 0.005, 'marker area ~ sqrt(number of fitted terms: EOD x pair with that pole nearer to the emitter)', ha='center',
         fontsize=8.5, color='0.35')
fig.tight_layout(rect=(0, 0.02, 1, 1))
for d in OUT.values(): fig.savefig(f'{d}/voltage_rule_gains.png', dpi=125)
print('saved voltage_rule_accuracy.png and voltage_rule_gains.png in', ', '.join(OUT.values()))

# 3. my gains vs Denis' gains, electrode by electrode (each model's gains / its median, in-sample fit on all clean-good EODs)
MINE_VS = {'Nov 5: N150, square': '#2a78d6', 'N200 first tank, no electrodes': '#eb6834', 'N200 tank, no fish bodies': '#e87ba4'}
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
for a, rec in zip(axs, OUT):
    G = pd.concat([pd.read_csv(f'{OUT[rec]}/gains.csv')] +
                  ([pd.read_csv(f'{OUT[rec]}/gains_extra.csv')] if os.path.exists(f'{OUT[rec]}/gains_extra.csv') else []))
    rel = lambda m: (lambda s: s.set_index('electrode').gain / np.median(s.gain[s.n_terms > 0]))(G[G.model == m])
    DN = 'Denis analytic, my mask: fit without near-singular' if (G.model == 'Denis analytic, my mask: fit without near-singular').any() \
        else 'Denis analytic, my mask'
    dn = rel(DN); n_dn = G[G.model == DN].set_index('electrode').n_terms
    txt = []
    for m, c in MINE_VS.items():
        mi = rel(m).reindex(dn.index); ok = (n_dn > 0) & mi.notna()
        r_all = np.corrcoef(mi[ok], dn[ok])[0, 1]
        a.scatter(mi[ok], dn[ok], s=15 + 70 * np.sqrt(n_dn[ok] / n_dn.max()), color=c, alpha=0.85, edgecolors='w', linewidths=0.5,
                  label=f'{m}: r = {r_all:.2f}')
        txt.append((m, r_all))
        if m == 'Nov 5: N150, square':
            for e in dn.index[ok]: a.annotate(e, (mi[e], dn[e]), textcoords='offset points', xytext=(5, 3), fontsize=8, color='0.3')
    lim = max(a.get_xlim()[1], a.get_ylim()[1]); low = min(0, a.get_xlim()[0], a.get_ylim()[0])
    a.plot([low, lim], [low, lim], color='0.6', lw=0.8, ls='--'); a.axhline(0, color='0.75', lw=0.6); a.axvline(0, color='0.75', lw=0.6)
    a.set_xlim(low, lim); a.set_ylim(low, lim); a.set_aspect('equal')
    a.set_xlabel('my gain / median of my gains')
    a.set_ylabel("Denis' gain / median" + (' (refit without near-singular terms)' if DN.endswith('singular') else ''))
    a.set_title(f'{rec}: fitted signed-electrode gains, mine vs Denis\' (labels: electrode; size ~ n terms)', fontsize=10.5)
    a.legend(fontsize=8.5, frameon=False, loc='upper left'); a.grid(alpha=0.3); a.set_axisbelow(True)
    for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    print(rec, 'correlation with Denis\' gains:', {m: round(r, 3) for m, r in txt})
fig.text(0.5, 0.01, "mine: main-phase peak sample vs unit-current model; Denis: signed peak-to-peak vs his predicted PTP. Electrode "
         "labels for the Nov 5 square model (same electrode positions and tank edges as Denis').", ha='center', fontsize=8.5, color='0.35')
fig.tight_layout(rect=(0, 0.03, 1, 1))
for d in OUT.values(): fig.savefig(f'{d}/voltage_rule_gains_mine_vs_denis.png', dpi=125)
print('saved voltage_rule_gains_mine_vs_denis.png')
