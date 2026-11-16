"""Figure: one row of the system matrix A (row 12 of the 5 x 5 uniform-water grid), as a strip of 25 cells.
Run from the repo root: python make_A_row_figure.py -> results/A_row12_uniform_water_N5.png / .pdf"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from compute_electric_field import build_A_shell, flatten_2d_grid

N, K = 5, flatten_2d_grid(2, 2, 5)
OUT = 'results/A_row12_uniform_water_N5'
row = build_A_shell(np.zeros((N, N)), N, 0.0, 1.0, 1.0, 1.0).toarray().real[K]
CMAP, NORM = plt.cm.RdBu, TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
NAME = {K - N: 'N', K - 1: 'W', K: 'node\nitself', K + 1: 'E', K + N: 'S'}

fig, ax = plt.subplots(figsize=(14, 2.6))
ax.imshow(row[None, :], cmap=CMAP, norm=NORM, aspect='equal')
for m, v in enumerate(row):
    ax.text(m, 0, f'{v:g}', ha='center', va='center', fontsize=12, color='w' if abs(v) >= 3 else 'k',
            weight='bold' if v != 0 else 'normal')
    if m in NAME:
        ax.text(m, 0.95, NAME[m], ha='center', va='top', fontsize=9, color='0.25')
for t in np.arange(-0.5, N * N, 1):
    ax.axvline(t, color='0.75', lw=0.6)
for t in np.arange(N, N * N, N) - 0.5:                                      # grid-row boundaries
    ax.axvline(t, color='0.35', lw=1.4)
ax.set(xticks=range(N * N), yticks=[0], yticklabels=[f'row {K}'], xlabel='column m (node)', ylim=(1.6, -0.5))
ax.tick_params(labelsize=9)
ax.set_title(f'row {K} of A ({N} × {N} grid, uniform water, σ = 1):  ϕ₇ + ϕ₁₁ − 4ϕ₁₂ + ϕ₁₃ + ϕ₁₇ = −s₁₂\n'
             f'5 nonzeros out of {N * N}, summing to 0; dark lines separate the grid rows i = 0 … {N - 1}', fontsize=11)
for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout()
fig.savefig(OUT + '.png', dpi=170, bbox_inches='tight'); fig.savefig(OUT + '.pdf', bbox_inches='tight')
print('saved', OUT + '.png/.pdf', '| nonzeros:', {m: row[m] for m in np.flatnonzero(row)})
