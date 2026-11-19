"""Figure for collaborators: the physics model vs the recorded electrode-pair voltages for two EODs where the
largest fish sits right next to electrode pair 2 and pair 5, showing those two pairs have the opposite sign.
Model: resistive, unit head-positive current, other fish as bodies. Recording: raw values at the EOD's first
(head-positive) phase P1. Both scaled to max |v| = 1; no signs changed anywhere.
Run from the repo root: python make_polarity_examples.py -> results/polarity_examples_pair2_pair5.png / .pdf"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compute_electric_field import mask_to_grid
from eod_assign import (eod_sample_count, recorded_pair_traces, eod_phase_samples, normalize_peak, candidate_field,
                        pair_voltages, iter_mgz_frames)
from io_utils import load_electrodes
from midline_methods import fish_midline_with_head

MGZ = 'data/video2024-02-20T21_16_26__223e__59997__masks_packed.mgz'
N, delta_h, FS = 200, 0.5, 20000.0
MAT = dict(sigma_water=1.0, eps_water=0.0, sigma_interior=200.0, eps_interior=0.0, sigma_skin=0.05, eps_skin=2.7e-6)
pairs = load_electrodes('data/electrodes_signed_px_coord_2024-02-13.csv', 912)
pair_ids = [pid for pid, _, _ in pairs]
df_all = pd.read_csv('data/Data_4fish_2024-02-20T21_16_26.csv')
n_samp = eod_sample_count(df_all)

EXAMPLES = [(2, 13113), (5, 7249)]          # (pair, df_eod row): fish 1, the largest fish, right at that pair
OUT = 'results/polarity_examples_pair2_pair5'

g = np.linspace(0, 1, N)
fish_colors = plt.cm.tab10.colors
frames = dict(iter_mgz_frames(MGZ, {int(df_all.loc[r, 'Frame']) for _, r in EXAMPLES}))


def plot_example(axes, pid, r):
    row = df_all.loc[r]; fr = int(row['Frame']); fm = frames[fr]; k = pair_ids.index(pid)
    # emitter = the fish nearest the pair
    pts = np.array([pairs[k][1], pairs[k][2]])
    dist = [np.inf if not m.any() else np.min(np.linalg.norm(np.argwhere(m)[:, None] / 911 - pts[None], axis=2))
            for m in fm]
    f = int(np.argmin(dist))
    largest = int(np.argmax([m.sum() for m in fm]))
    mids = [fish_midline_with_head(m, n_ctrl=101)['midline'] if m.sum() >= 50 else None for m in fm]
    phi = candidate_field(fm, f, mids[f], N, delta_h, MAT).real
    v_mod = pair_voltages(phi, pairs, N)
    V = recorded_pair_traces(row, pair_ids, n_samp)
    p1, p2 = eod_phase_samples(V)
    a, b = normalize_peak(v_mod), normalize_peak(V[:, p1])

    # (a) scene + model potential
    ax = axes[0]
    v = np.percentile(np.abs(phi), 99)
    ax.imshow(phi.T, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r', vmin=-v, vmax=v)
    for j, m in enumerate(fm):
        if not m.any(): continue
        ax.contour(g, g, mask_to_grid(m, N).T.astype(float), levels=[0.5], colors=['k' if j == f else '0.35'],
                   linewidths=2.0 if j == f else 0.8)
        if mids[j] is not None and j == f:
            ax.plot(*mids[j][-1], 'o', color='gold', ms=7, mec='k', mew=0.8, zorder=5, label='head')
            ax.plot(*mids[j][0], 's', color='w', ms=6, mec='k', mew=0.8, zorder=5, label='tail')
    for q, pp, pm in pairs:
        hl = q == pid
        ax.plot([pp[0], pm[0]], [pp[1], pm[1]], color='m' if hl else '0.55', lw=2.2 if hl else 0.6, zorder=3)
        ax.text(*pp, '+', color='#B03A2E', fontsize=15 if hl else 11, weight='bold', ha='center', va='center', zorder=4)
        ax.text(*pm, '−', color='#1F4E9A', fontsize=15 if hl else 11, weight='bold', ha='center', va='center', zorder=4)
        mid_pt = 0.5 * (pp + pm) if np.linalg.norm(pp - pm) < 1 else pp + 0.3 * (pm - pp)   # diagonals cross mid-tank
        ax.text(*(mid_pt + 0.04 * (0.5 - mid_pt) / max(np.linalg.norm(0.5 - mid_pt), 1e-9)), str(q), zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.7),
                fontsize=10 if hl else 7, color='m' if hl else '0.3', weight='bold' if hl else 'normal',
                ha='center', va='center')
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect='equal', xticks=[], yticks=[])
    who = 'largest fish' if f == largest else f'fish {f}'
    ax.set_title(f'frame {fr}: {who} (fish {f}) at pair {pid}\nmodel potential, head-positive current', fontsize=10)
    ax.legend(loc='center', fontsize=7, framealpha=0.8, bbox_to_anchor=(0.5, 0.42))

    # (b) recorded traces, all pairs
    ax = axes[1]
    t_ms = np.arange(n_samp) / FS * 1e3
    scale = np.max(np.abs(V))
    for i, q in enumerate(pair_ids):
        ax.plot(t_ms, V[i] / scale, color='m' if q == pid else '0.6', lw=2.2 if q == pid else 0.9,
                label=f'pair {pid}' if q == pid else ('other pairs' if i == 0 or (i == 1 and pair_ids[0] == pid) else None),
                zorder=3 if q == pid else 2)
    for p, lab in ((p1, 'P1'), (p2, 'P2')):
        ax.axvline(t_ms[p], color='k', ls='--' if lab == 'P1' else ':', lw=0.8)
        ax.text(t_ms[p], 1.08, lab, ha='center', fontsize=9)
    ax.axhline(0, color='0.85', lw=0.5)
    ax.set(xlabel='time (ms)', ylabel='recorded voltage (max |V| = 1)', ylim=(-1.15, 1.2),
           title=f'recorded EOD, all 10 pairs (df_eod row {r})')
    ax.legend(fontsize=8, loc='lower left')

    # (c) model vs recording at P1, both scaled to max |v| = 1, signs untouched
    ax = axes[2]
    x = np.arange(len(pair_ids)); w = 0.38
    ax.bar(x - w / 2, b, w, color='k', label=f'recorded at P1 (t = {t_ms[p1]:.2f} ms)')
    ax.bar(x + w / 2, a, w, color='#E67E22', label='model (head-positive current)')
    ax.axvspan(k - 0.5, k + 0.5, color='m', alpha=0.12, zorder=0)
    ax.annotate('opposite sign', (k, 1.08 if b[k] > 0 else -1.08), ha='center', fontsize=9, color='m', weight='bold')
    ax.axhline(0, color='k', lw=0.5)
    others = [i for i in range(len(pair_ids)) if i != k]
    ax.set(xticks=x, xticklabels=pair_ids, xlabel='electrode pair', ylabel='voltage (max |v| = 1)', ylim=(-1.25, 1.25),
           title=f'model vs recording; L2 over the other 9 pairs = {np.linalg.norm(a[others] - b[others]):.2f}')
    ax.legend(fontsize=8, loc='lower left')


fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(17, 5.6 * len(EXAMPLES)),
                         gridspec_kw=dict(width_ratios=[1, 1.1, 1.3]))
for (pid, r), ax_row in zip(EXAMPLES, axes):
    plot_example(ax_row, pid, r)
fig.suptitle('Electrode pairs 2 and 5: model and recording agree on 9 of 10 pairs, but have opposite sign on the pair '
             'next to the fish\n(model: resistive, other fish as bodies; electrode polarities as in '
             'electrodes_signed_px_coord_2024-02-13.csv; no signs changed)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT + '.png', dpi=150); fig.savefig(OUT + '.pdf')
print('saved', OUT + '.png/.pdf')
