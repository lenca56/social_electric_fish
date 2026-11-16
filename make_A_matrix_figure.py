"""Figure: the system matrix A in uniform water on a tiny N x N grid, built by compute_electric_field.build_A_shell.

Left: the grid, nodes numbered k = i N + j, with the 5-point stencil of one interior node. Right: A itself, every nonzero entry
annotated: -4 on the diagonal of interior nodes, 1 for each neighbour; wall nodes have fewer neighbours (-3 on an edge, -2 in a
corner), because no current flows through the wall (Neumann).
Run from the repo root: python make_A_matrix_figure.py -> results/A_matrix_uniform_water_N5.png / .pdf"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from compute_electric_field import build_A_shell, flatten_2d_grid

N = 5
OUT = 'results/A_matrix_uniform_water_N5'
A = build_A_shell(np.zeros((N, N)), N, 0.0, 1.0, 1.0, 1.0).toarray().real      # no fish: all water, sigma = 1
assert np.allclose(A, A.T) and np.allclose(A.sum(axis=1), 0)

K = flatten_2d_grid(2, 2, N)                                                  # interior node whose stencil is drawn
nb = [flatten_2d_grid(i, j, N) for i, j in ((1, 2), (3, 2), (2, 1), (2, 3))]
CMAP, NORM = plt.cm.RdBu, TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1, 1.75]))

# (a) the grid and the stencil of node K
for t in range(N):
    ax0.plot([0, N - 1], [t, t], color='0.6', lw=0.8, zorder=1); ax0.plot([t, t], [0, N - 1], color='0.6', lw=0.8, zorder=1)
for i in range(N):
    for j in range(N):
        k = flatten_2d_grid(i, j, N)
        c = CMAP(NORM(-4)) if k == K else (CMAP(NORM(1)) if k in nb else 'w')
        ax0.scatter(j, i, s=520, c=[c], edgecolors='0.3', zorder=2)
        ax0.text(j, i, str(k), ha='center', va='center', fontsize=10, zorder=3,
                 color='w' if k == K else 'k', weight='bold' if k in [K] + nb else 'normal')
for k_, (i, j) in zip(nb, ((1, 2), (3, 2), (2, 1), (2, 3))):
    ax0.annotate('', (j, i), (2, 2), arrowprops=dict(arrowstyle='->', lw=1.6, color=CMAP(NORM(1.6)), shrinkA=14, shrinkB=14))
    ax0.text(2 + 0.62 * (j - 2) + 0.18 * (i - 2), 2 + 0.62 * (i - 2) - 0.18 * (j - 2), '1', ha='center', va='center', fontsize=10)
ax0.text(2.28, 1.72, '−4', ha='center', va='center', fontsize=10, color=CMAP(NORM(-4)), weight='bold')
ax0.plot([-0.5, N - 0.5, N - 0.5, -0.5, -0.5], [-0.5, -0.5, N - 0.5, N - 0.5, -0.5], color='k', lw=2.5)
ax0.text((N - 1) / 2, N - 0.15, 'insulating tank wall (no edge through it)', ha='center', va='bottom', fontsize=9)
ax0.set(xlim=(-0.8, N - 0.2), ylim=(N + 0.3, -0.8), aspect='equal', xticks=range(N), yticks=range(N),
        xlabel='j (column)', ylabel='i (row)')
ax0.set_title(f'{N} × {N} grid in uniform water, node k = i·N + j\n'
              f'row {K} of A: σ(ϕ_E + ϕ_W + ϕ_N + ϕ_S − 4ϕ₀) = −s₀', fontsize=10)

# (b) A, every nonzero entry annotated
im = ax1.imshow(np.where(A == 0, np.nan, A), cmap=CMAP, norm=NORM)
for r in range(N * N):
    for c in range(N * N):
        if A[r, c] != 0:
            ax1.text(c, r, f'{A[r, c]:g}', ha='center', va='center', fontsize=7,
                     color='w' if abs(A[r, c]) >= 3 else 'k', weight='bold' if r == K else 'normal')
ax1.add_patch(plt.Rectangle((-0.5, K - 0.5), N * N, 1, fill=False, ec='k', lw=1.5))
for t in np.arange(N, N * N, N) - 0.5:                                          # block boundaries = grid rows
    ax1.axhline(t, color='0.75', lw=0.6); ax1.axvline(t, color='0.75', lw=0.6)
ax1.set(xticks=range(N * N), yticks=range(N * N), xlabel='column m (neighbour node)', ylabel='row k (node)')
ax1.tick_params(labelsize=7)
ax1.set_title(f'A ({N * N} × {N * N}), blank = 0: diagonal −4 inside, −3 on a wall, −2 in a corner; 1 per neighbour\n'
              f'every row sums to 0 (A·1 = 0, singular); symmetric; row {K} boxed', fontsize=10)
cb = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02); cb.set_label('entry of A (σ = 1)')

plt.tight_layout()
fig.savefig(OUT + '.png', dpi=160, bbox_inches='tight'); fig.savefig(OUT + '.pdf', bbox_inches='tight')
print('saved', OUT + '.png/.pdf | diagonal values:', sorted(set(np.diag(A).astype(int))))
