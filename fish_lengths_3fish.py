"""Body length of each fish in each 3-fish recording and its fish factor from the 4-fish length law, for kaden_3fish_assign_rules.py
(FISH_FACTORS = 'length_law').

1. Lengths: fish_lengths.py on each recording's final Kaden masks (median midline length of the straightest clean frames; run it
   first, see below), in cm at the 3-fish video calibration (cluster_kaden_3fish_common.PX_PER_CM_VIDEO = 15.2 px/cm, as the law).
   Two sets of 3 fish (FISH_SETS): set 1 swims in the Oct 14 recordings and 2024-10-15T14_12_25 (A ~17, B ~14.4, C ~12 cm), set 2 in
   the later Oct 15 recordings, 14_30_53 on (A ~20, B ~16.9, C ~14.1 cm). Each recording uses one length per fish: the median of its
   set's recordings (L_cm); each recording's own measurement stays in L_cm_measured.
2. Fish factor a = slope x L + intercept, the linear law of the 4-fish Kaden fish x signed-electrode fit
   (kaden_3fish_assign_rules.LENGTH_LAW; results/eod_assign_4fish_kaden/clean_emitter_set/gain_fish_x_signed_electrode/).

Output in results/eod_assignment_3_fish_silent/fish_lengths/:
  fish_lengths_3fish.csv          video, fish_set, fish, L_cm (used: median of its set), L_px and L_cm_measured (this recording),
                                  n_frames_used, a_mine, a_denis
  fish_factor_vs_length_3fish.png the 4-fish law and fish (left: my model, right: Denis) with the 3-fish fish placed on it

    for v in <recordings>; do python fish_lengths.py --pkl data/3_fish_silent/masks/video$v/outputs/video$v.pkl --tag $v \\
        --json_dir results/eod_assignment_3_fish_silent/fish_lengths --fig_dir results/eod_assignment_3_fish_silent/fish_lengths; done
    python fish_lengths_3fish.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_kaden_3fish_common as C
from kaden_3fish_assign_rules import LENGTH_LAW, FISH_LENGTHS

DIR = os.path.dirname(FISH_LENGTHS)
LAW_4FISH = 'results/eod_assign_4fish_kaden/clean_emitter_set/gain_fish_x_signed_electrode/fish_factors_and_length_fits.csv'
L_4FISH_CM = {'A': 15.65, 'B': 13.10, 'C': 9.82, 'D': 8.51}   # the lengths the 4-fish law was fitted at (EOD_gain_fish_x_signed_electrode_4fish_kaden.ipynb)
GROUP = lambda v: '10-14' if v.startswith('2024-10-14') else ('10-15 14_12_25' if v == '2024-10-15T14_12_25' else '10-15 later')
MARK = {'10-14': 'o', '10-15 14_12_25': '^', '10-15 later': 's'}
FISH_COL = {'A': '#1F5FBF', 'B': '#17A0B0', 'C': '#F08A24'}
FISH_SETS = {'set 1': ['2024-10-14T15_47_02', '2024-10-14T16_07_02', '2024-10-14T16_47_02', '2024-10-15T14_12_25'],
             'set 2': ['2024-10-15T14_30_53', '2024-10-15T14_50_54', '2024-10-15T15_10_54', '2024-10-15T15_30_54']}
FISH_SET = {v: k for k, vs in FISH_SETS.items() for v in vs}
SET_MARK = {'set 1': 'o', 'set 2': 's'}

rows = []
for v in C.VIDEOS:
    d = json.load(open(f'{DIR}/fish_lengths_{v}.json'))
    for f, r in d['fish'].items():
        rows.append(dict(video=v, fish=f, L_px=r['L_px'], L_cm_measured=round(r['L_px'] / C.PX_PER_CM_VIDEO, 3) if r['L_px'] else np.nan,
                         n_frames_used=r['n_frames_used']))
T = pd.DataFrame(rows)
T.insert(1, 'fish_set', T.video.map(FISH_SET))
L_USED = T.groupby(['fish_set', 'fish']).L_cm_measured.median()                 # one length per fish of each set
T.insert(3, 'L_cm', [round(L_USED[(fs, f)], 3) for fs, f in zip(T.fish_set, T.fish)])
for m, (s, c) in LENGTH_LAW.items(): T[f'a_{m}'] = s * T.L_cm + c
T.to_csv(FISH_LENGTHS, index=False)
print(T.pivot(index='video', columns='fish', values='L_cm_measured').round(2).to_string())
print('used (median over each set\'s recordings):', L_USED.round(2).to_dict())

law = pd.read_csv(LAW_4FISH) if os.path.exists(LAW_4FISH) else None
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, (m, name) in zip(axs, (('mine', 'my model'), ('denis', "Denis' model"))):
    s, c = LENGTH_LAW[m]; x = np.linspace(7.5, 21, 50)
    ax.plot(x, s * x + c, color='k', lw=1.5, label=f'4-fish linear law: {s:.4g}·L + {c:.4g}')
    if law is not None:                                        # the 4-fish fish the law was fitted on
        q = law[law.model == ('mine' if m == 'mine' else 'Denis')].set_index('parameter')
        for f, L in L_4FISH_CM.items():
            ax.plot(L, q.estimate[f'fish {f}'], 'D', color='0.55', mec='k', ms=8, label='4-fish fish (fitted factor)' if f == 'A' else None)
            ax.annotate(f'4-fish {f}', (L, q.estimate[f'fish {f}']), xytext=(5, -12), textcoords='offset points', fontsize=7.5, color='0.4')
    for (fs, f), g in T.groupby(['fish_set', 'fish']):                    # measured in each recording
        ax.plot(g.L_cm_measured, s * g.L_cm_measured + c, SET_MARK[fs], mfc='none', mec=FISH_COL[f], mew=1.2, ms=6, alpha=0.8,
                label=f'{fs} fish {f}, measured ({len(g)} rec.)')
    for (fs, f), L in L_USED.items():                                     # used for that set's recordings
        ax.plot(L, s * L + c, SET_MARK[fs], color=FISH_COL[f], mec='k', mew=1, ms=12, zorder=5,
                label=f'{fs} fish {f} used: {L:.2f} cm, a = {s * L + c:.4g}')
    ax.set(xlabel='fish length L (cm, straightest frames)', ylabel=f'fish factor a ({"V per model unit" if m == "mine" else "V per Denis unit"})',
           title=f'{name}: fish factor from the 4-fish length law', xlim=(7.5, 21))
    ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7.5, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{DIR}/fish_factor_vs_length_3fish.png', dpi=110)
print(f'written {FISH_LENGTHS} and {DIR}/fish_factor_vs_length_3fish.png')
