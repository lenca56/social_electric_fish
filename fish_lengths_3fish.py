"""Body length of each fish in each 3-fish recording and its fish factor from the 4-fish length law, for kaden_3fish_assign_rules.py
(FISH_FACTORS = 'length_law').

1. Lengths: fish_lengths.py on each recording's final Kaden masks (median midline length of the straightest clean frames; run it
   first, see below), in cm at the 3-fish video calibration (cluster_kaden_3fish_common.PX_PER_CM_VIDEO = 15.2 px/cm, as the law).
   The same 3 fish swim in every recording, so every recording uses one length per fish: the median over the LENGTHS_FROM recordings
   (Oct 14). The later Oct 15 recordings measure every fish ~18 % longer (14_30_53 on; the pixel scale is unchanged); they are kept
   in L_cm_measured but not used.
2. Fish factor a = slope x L + intercept, the linear law of the 4-fish Kaden fish x signed-electrode fit
   (kaden_3fish_assign_rules.LENGTH_LAW; results/eod_assign_4fish_kaden/clean_emitter_set/gain_fish_x_signed_electrode/).

Output in results/eod_assignment_3_fish_silent/fish_lengths/:
  fish_lengths_3fish.csv          video, fish, L_cm (used), L_px and L_cm_measured (this recording), n_frames_used, a_mine, a_denis
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
LENGTHS_FROM = '2024-10-14'    # recordings (prefix) whose median length per fish is used for every recording

rows = []
for v in C.VIDEOS:
    d = json.load(open(f'{DIR}/fish_lengths_{v}.json'))
    for f, r in d['fish'].items():
        rows.append(dict(video=v, fish=f, L_px=r['L_px'], L_cm_measured=round(r['L_px'] / C.PX_PER_CM_VIDEO, 3) if r['L_px'] else np.nan,
                         n_frames_used=r['n_frames_used']))
T = pd.DataFrame(rows)
L_USED = T[T.video.str.startswith(LENGTHS_FROM)].groupby('fish').L_cm_measured.median()
T.insert(2, 'L_cm', T.fish.map(L_USED).round(3))
for m, (s, c) in LENGTH_LAW.items(): T[f'a_{m}'] = s * T.L_cm + c
T.to_csv(FISH_LENGTHS, index=False)
print(T.pivot(index='video', columns='fish', values='L_cm_measured').round(2).to_string())
print(f'used everywhere (median of {LENGTHS_FROM}*):', L_USED.round(2).to_dict())

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
    for (grp, f), g in T.groupby([T.video.map(GROUP), 'fish']):           # measured in each recording (only Oct 14 used)
        ax.plot(g.L_cm_measured, s * g.L_cm_measured + c, MARK[grp], mfc='none', mec=FISH_COL[f], mew=1.2, ms=6, alpha=0.8,
                label=f'3-fish {f} measured, {grp} ({len(g)} rec.)')
    for f, L in L_USED.items():                                          # used for every recording
        ax.plot(L, s * L + c, 'o', color=FISH_COL[f], mec='k', mew=1, ms=12, zorder=5, label=f'3-fish {f} used: {L:.2f} cm, a = {s * L + c:.4g}')
    ax.set(xlabel='fish length L (cm, straightest frames)', ylabel=f'fish factor a ({"V per model unit" if m == "mine" else "V per Denis unit"})',
           title=f'{name}: fish factor from the 4-fish length law', xlim=(7.5, 21))
    ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7.5, frameon=False, loc='upper left')
fig.tight_layout(); fig.savefig(f'{DIR}/fish_factor_vs_length_3fish.png', dpi=110)
print(f'written {FISH_LENGTHS} and {DIR}/fish_factor_vs_length_3fish.png')
