"""Effective 2D conductivity of the other fish (effective_sigma_3d.py) for water depths 12, 15 and 20 cm: vs the height difference to
the emitter, per orientation and emitter distance; skin 0.1 (solid) and 1 cm of water (dashed).
    python plot_effective_sigma_depths.py      -> results/effective_sigma_3d/effective_sigma_by_depth.png
"""
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = 'results/effective_sigma_3d'; HS = {12: '#2a78d6', 15: '#3fa34d', 20: '#eb6834'}
T = pd.concat([pd.read_csv(f'{D}/effective_sigma_H{h}.csv').assign(H=h) for h in HS])
fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True)
for r, o in enumerate(('parallel', 'perpendicular')):
    for c, d in enumerate((4.0, 8.0, 15.0)):
        a = axs[r, c]
        for h, col in HS.items():
            for lam, ls in ((0.1, '-'), (1.0, '--')):
                s = T[(T.H == h) & (T.orientation == o) & (T.distance_cm == d) & (T.skin_lambda_cm == lam)].sort_values('dz_cm')
                a.plot(s.dz_cm, s.sigma_eff, ls, marker='o', ms=4, color=col, label=f'{h} cm water, skin {lam:g}')
            e = T[(T.H == h) & (T.orientation == o) & (T.distance_cm == d) & (T.skin_lambda_cm == 0.1) & (T.dz_cm == 0)].fish_effect_rel.iloc[0]
            a.text(0.98, 0.97 - 0.06 * list(HS).index(h), f'{h} cm: fish changes electrodes by {100 * e:.2g} %', transform=a.transAxes,
                   ha='right', va='top', fontsize=8, color=col)
        a.axhline(1, color='0.6', lw=0.8, ls=':'); a.set_title(f'{o}, emitter {d:g} cm away', fontsize=10)
        a.grid(alpha=0.3)
        if c == 0: a.set_ylabel('effective 2D conductivity (x water)')
        if r == 1: a.set_xlabel('height difference emitter - other fish (cm)')
        for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
axs[0, 0].legend(fontsize=7.5, frameon=False, loc='lower left')
fig.suptitle('Effective 2D conductivity of the other fish (ellipsoid 14.5 x 1.6 x 3.6 cm at mid-depth, interior 100 x water) by water depth; '
             'solid: skin 0.1, dashed: skin 1 cm of water', fontsize=10.5)
fig.text(0.5, 0.005, 'parallel: emitter side by side (field along the other fish); perpendicular: emitter head pointing at its middle '
         '(field across it). Emitter kept at least 1 cm under the surface. Dotted: water.', ha='center', fontsize=8.5, color='0.35')
fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(f'{D}/effective_sigma_by_depth.png', dpi=130)
