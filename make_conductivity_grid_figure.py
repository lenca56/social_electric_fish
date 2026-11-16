"""Figure: the finite-volume grid around one fish, coloured by the conductivity each part of the model uses (BEM set).

The fish is drawn as a passive body (interior + skin sheet), as every non-emitting fish is in the model; the emitter itself is
transparent (water). Grid and materials as in model_voltages_4fish.py: N = 150, water 1, interior 100, skin 0.04998 over
0.005 cm (all x water = 1e-4 S/cm). Each grid node is water or interior (occupancy > 0.5); an edge cut by the fish boundary gets
the series conductivity of compute_electric_field._edge_weight, with the skin as a thin sheet at the crossing.
Run from the repo root: python make_conductivity_grid_figure.py -> results/conductivity_grid_frame330.png / .pdf"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from compute_electric_field import occupancy_field, theta_outer, _edge_weight

MASKS_NPY = 'data/video2024-02-20T21_16_26__223e__59997__masks_packed.mgz_masks_first_1000_frames.npy'
FRAME, FISH = 330, 0                       # second-largest fish (mask 0, 2575 px): cleanest outline in this frame
OUT = 'results/conductivity_grid_frame330'

# BEM set (EOD_assignment_3_fish_conductivities.ipynb, model_voltages_4fish.py), relative to water = 1e-4 S/cm
P, N = 912, 150
PX_PER_CM = P / 60
CELL_CM = (P - 1) / (N - 1) / PX_PER_CM
SKIN_CM, LAMBDA_CM = 0.005, 1e-4 / 1e-3
SIG_W, SIG_I = 1.0, 100.0
SIG_S = 1 / (LAMBDA_CM / SKIN_CM + 1 / SIG_I)
DELTA_H = SKIN_CM / CELL_CM
S_PER_CM = 1e-4

masks = np.load(MASKS_NPY, mmap_mode='r')
fm = np.asarray(masks[:, FRAME])
m = fm[FISH]
occ = occupancy_field(m[None], N)
inside = occ > 0.5
tx, ty = theta_outer(occ)

node_cm = np.arange(N) * (P - 1) / (N - 1) / PX_PER_CM            # node i sits at pixel i (P-1)/(N-1)
pix = np.argwhere(m)
r0, c0 = pix.min(0) / PX_PER_CM - 1.2
r1, c1 = pix.max(0) / PX_PER_CM + 1.2

# cut edges: segment between the two nodes, coloured by the conductivity the solver puts on that edge
segs, vals, cross = [], [], []
for axis, th in ((0, tx), (1, ty)):
    for i, j in zip(*np.where(~np.isnan(th))):
        a = (i, j); b = (i + 1, j) if axis == 0 else (i, j + 1)
        g = _edge_weight(SIG_I if inside[a] else SIG_W, SIG_I if inside[b] else SIG_W, th[i, j], DELTA_H, SIG_S, SIG_I)
        pa = np.array([node_cm[a[1]], node_cm[a[0]]]); pb = np.array([node_cm[b[1]], node_cm[b[0]]])   # (x = col, y = row)
        segs.append([pa, pb]); vals.append(abs(g)); cross.append(pa + th[i, j] * (pb - pa))
vals, cross = np.array(vals), np.array(cross)

# one conductivity colour scale for everything: log10 σ, red = resistive (skin) -> white = water -> blue = conductive (interior)
CMAP = LinearSegmentedColormap.from_list('sigma', plt.cm.RdBu(np.linspace(0.06, 0.94, 256)))
NORM = TwoSlopeNorm(vmin=np.log10(SIG_S), vcenter=0.0, vmax=np.log10(SIG_I))
col = lambda sig: CMAP(NORM(np.log10(sig)))
C_W, C_I, C_S = col(SIG_W), col(SIG_I), col(SIG_S)

fig = plt.figure(figsize=(18, 7.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 2.3, 1.15], height_ratios=[1.25, 1], wspace=0.12, hspace=0.25)

# (a) whole tank, all four fish, zoom box
ax = fig.add_subplot(gs[:, 0])
ax.add_patch(Rectangle((0, 0), P / PX_PER_CM, P / PX_PER_CM, fc=C_W, ec='0.4', lw=1))
for j, mj in enumerate(fm):
    if mj.any():
        ax.contour(np.arange(P) / PX_PER_CM, np.arange(P) / PX_PER_CM, mj.astype(float), levels=[0.5],
                   colors=['k' if j == FISH else '0.45'], linewidths=1.6 if j == FISH else 0.8)
ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec='k', lw=1.5, ls=':'))
ax.set(xlim=(0, P / PX_PER_CM), ylim=(P / PX_PER_CM, 0), aspect='equal', xlabel='x (cm)', ylabel='y (cm)',
       title=f'tank, frame {FRAME}: zoomed fish in black (dotted box)\n{N} × {N} grid, h = {CELL_CM:.2f} cm')

# (b) zoom: node media, grid, cut edges, skin sheet
ax = fig.add_subplot(gs[:, 1])
edges = np.r_[node_cm - node_cm[1] / 2, node_cm[-1] + node_cm[1] / 2]       # control-volume (cell) boundaries
ax.pcolormesh(edges, edges, np.log10(np.where(inside, SIG_I, SIG_W)), cmap=CMAP, norm=NORM,
              edgecolors='none')                                   # shading = each node's control volume (box)
sel = (node_cm >= min(r0, c0) - 1) & (node_cm <= max(r1, c1) + 1)
ax.vlines(node_cm[sel], node_cm[sel].min(), node_cm[sel].max(), color='0.6', lw=0.5, zorder=1)   # grid lines through the nodes
ax.hlines(node_cm[sel], node_cm[sel].min(), node_cm[sel].max(), color='0.6', lw=0.5, zorder=1)
X, Y = np.meshgrid(node_cm, node_cm)
ax.plot(X[np.ix_(sel, sel)].ravel(), Y[np.ix_(sel, sel)].ravel(), 'o', color='0.45', ms=1.8, zorder=2)
lc = LineCollection(segs, array=np.log10(vals), cmap=CMAP, norm=NORM, linewidths=3.4, zorder=3,
                    path_effects=[pe.Stroke(linewidth=5, foreground='0.15'), pe.Normal()])
ax.add_collection(lc)
ax.contour(np.arange(P) / PX_PER_CM, np.arange(P) / PX_PER_CM, m.astype(float), levels=[0.5], colors='k', linewidths=0.9,
           linestyles='--', zorder=4)
ax.contour(node_cm, node_cm, occ, levels=[0.5], colors=[C_S], linewidths=2.6, zorder=5)
ax.plot(cross[:, 0], cross[:, 1], 'o', ms=3.8, mfc=C_S, mec='w', mew=0.6, zorder=6)
ax.set(xlim=(c0, c1), ylim=(r1, r0), aspect='equal', xlabel='x (cm)', ylabel='y (cm)')
ax.set_title('grid around the fish as a passive body (emitter: transparent, i.e. water)', fontsize=11)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=ax, orientation='horizontal', fraction=0.05, pad=0.2, aspect=40)
ticks = [SIG_S, 0.1, 0.3, 1, 3, 10, 30, SIG_I]
cb.set_ticks(np.log10(ticks)); cb.set_ticklabels([f'{t:g}' if t != SIG_S else f'{t:.2f}' for t in ticks])
cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb.set_label('conductivity σ (× water, log scale): red = more resistive, white = water, blue = more conductive')
for sig, lab in ((SIG_S, 'skin'), (SIG_W, 'water'), (SIG_I, 'interior')):
    cb.ax.annotate(lab, (np.log10(sig), 1), (np.log10(sig), 1.9), xycoords=('data', 'axes fraction'),
                   textcoords=('data', 'axes fraction'), ha='center', fontsize=9, arrowprops=dict(arrowstyle='-', lw=0.8))
lo, hi = np.log10(vals.min()), np.log10(vals.max())
cb.ax.annotate('', (lo, 1.25), (hi, 1.25), xycoords=('data', 'axes fraction'), arrowprops=dict(arrowstyle='|-|', lw=1))
cb.ax.text((lo + hi) / 2, 1.45, 'cut edges', ha='center', fontsize=8, transform=cb.ax.get_xaxis_transform())

leg = [Patch(fc=C_W, ec='0.6', label=f'water node: σ = {SIG_W:g} (= {SIG_W * S_PER_CM:.0e} S/cm)'),
       Patch(fc=C_I, ec='none', label=f'interior node: σ = {SIG_I:g} (= {SIG_I * S_PER_CM:.0e} S/cm)\n'
                                           f'  (shading = the node\'s control volume, faces halfway between nodes)'),
       Line2D([], [], color=C_S, lw=2.6, marker='o', ms=4, mfc=C_S, mec='w',
              label=f'skin sheet at occupancy 0.5 (dots: crossings θ): σ = {SIG_S:.3f} (= {SIG_S * S_PER_CM:.0e} S/cm),\n'
                    f'  {SKIN_CM * 1e4:.0f} µm thick = {DELTA_H:.4f} cell (not to scale) → {SKIN_CM / (SIG_S * S_PER_CM) / 1e3:.0f} kΩ·cm²'),
       Line2D([], [], color=col(1.6), lw=3.4, path_effects=[pe.Stroke(linewidth=5, foreground='0.15'), pe.Normal()],
              label='cut edge (coloured by its σ_eff): water | skin | interior in series,\n  1/σ_eff = θ/σ₁ + (1−θ)/σ₂ + δ (1/σ_skin − 1/σ_interior)'),
       Line2D([], [], color='k', lw=0.9, ls='--', label='mask outline (video pixels)'),
       Line2D([], [], color='0.6', lw=0.8, marker='o', mfc='0.45', mec='0.45', ms=3,
              label=f'grid: nodes ϕ(i, j) at the line crossings, edges between them,\n  h = {CELL_CM:.3f} cm ({(P - 1) / (N - 1):.1f} px)')]
lax = fig.add_subplot(gs[0, 2]); lax.axis('off')
lax.legend(handles=leg, loc='upper left', fontsize=9, framealpha=0.95, borderaxespad=0)

# (c) one cut edge, schematic: water | skin | interior in series (skin thickness exaggerated)
ax = fig.add_subplot(gs[1, 2])
th, skin_w = 0.4, 0.06                                    # drawn crossing and skin width (true skin: DELTA_H = 0.012 cell)
ax.add_patch(Rectangle((0, 0), th, 1, fc=C_W, ec='0.6', lw=0.6))
ax.add_patch(Rectangle((th, 0), skin_w, 1, fc=C_S, ec='none'))
ax.add_patch(Rectangle((th + skin_w, 0), 1 - th - skin_w, 1, fc=C_I, ec='none'))
for x, lab in ((0, 'ϕ₁\nwater node'), (1, 'ϕ₂\ninterior node')):
    ax.plot(x, 0.5, 'o', color='0.2', ms=8, zorder=3); ax.text(x, -0.12, lab, ha='center', va='top', fontsize=9)
ax.annotate('', (0, 1.12), (th, 1.12), arrowprops=dict(arrowstyle='<->', lw=0.9))
ax.text(th / 2, 1.17, 'θh', ha='center', va='bottom', fontsize=10)
ax.annotate('', (th + skin_w, 1.12), (1, 1.12), arrowprops=dict(arrowstyle='<->', lw=0.9))
ax.text((1 + th + skin_w) / 2, 1.17, '(1−θ)h', ha='center', va='bottom', fontsize=10)
ax.text(th / 2, 0.5, f'σ = {SIG_W:g}', ha='center', va='center', fontsize=9)
ax.annotate(f'skin sheet, width δh\nσ = {SIG_S:.2f}', (th + skin_w / 2, 0.02), (th + skin_w / 2, -0.32), ha='center', va='top',
            fontsize=9, color=C_S, arrowprops=dict(arrowstyle='->', color=C_S, lw=1))
ax.text((1 + th + skin_w) / 2, 0.5, f'σ = {SIG_I:g}', ha='center', va='center', fontsize=9, color='w')
ax.set(xlim=(-0.15, 1.15), ylim=(-0.75, 1.35), xticks=[], yticks=[])
for sp_ in ax.spines.values(): sp_.set_visible(False)
ax.set_title(f'one cut edge: resistances in series\n(skin width exaggerated; true δ = {DELTA_H:.4f} cell)', fontsize=9)

fig.suptitle('Conductivities in the finite-volume model (BEM set, relative to water)', fontsize=13, y=0.98)
fig.savefig(OUT + '.png', dpi=160, bbox_inches='tight'); fig.savefig(OUT + '.pdf', bbox_inches='tight')
print('saved', OUT + '.png/.pdf', '| cut edges:', len(vals), '| sigma_eff range', vals.min().round(3), vals.max().round(3),
      '| delta_h', round(DELTA_H, 4), '| sigma_skin', round(SIG_S, 5))
