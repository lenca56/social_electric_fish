"""Combine the per-recording fish lengths (results/fish_lengths_<recording>.json, from fish_lengths.py) into one fixed
L per fish: the median over recordings. Writes results/fish_lengths_fixed.json and figures/fish_lengths_across_recordings.png.

Units as in fish_lengths.py: L_px in video pixels, L_tank = L_px / (image_size_px - 1) -> multipole_sources(..., L=L_tank).

    python fish_lengths_combine.py
"""
import glob, json, os
import numpy as np
import pandas as pd

OUT_JSON, OUT_FIG = 'results/fish_lengths_fixed.json', 'figures/fish_lengths_across_recordings.png'
OUTLIER = 0.10            # a recording whose L is more than 10 % from the fish's median is flagged


def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = {}
    for p in sorted(glob.glob('results/fish_lengths_*.json')):
        if os.path.abspath(p) == os.path.abspath(OUT_JSON): continue
        d = json.load(open(p)); runs[d['recording'].replace('video2024-02-20T', '')] = d
    sizes = {d['image_size_px'] for d in runs.values()}
    assert len(sizes) == 1, f'recordings with different image sizes: {sizes}'
    P = sizes.pop()
    T = pd.DataFrame({rec: {L: r['L_px'] for L, r in d['fish'].items()} for rec, d in runs.items()}).T.sort_index().astype(float)
    med = T.median()
    fixed = {}
    for L in T.columns:
        v = T[L].dropna(); dev = (v / med[L] - 1)
        fixed[L] = dict(L_px=round(float(med[L]), 3), L_tank=round(float(med[L]) / (P - 1), 6), n_recordings=int(len(v)),
                        L_px_iqr=[round(float(v.quantile(.25)), 2), round(float(v.quantile(.75)), 2)], L_px_range=[round(float(v.min()), 2), round(float(v.max()), 2)],
                        outlier_recordings=sorted(dev.index[dev.abs() > OUTLIER].tolist()), per_recording_L_px={k: round(float(x), 3) for k, x in v.items()})
    json.dump(dict(image_size_px=int(P), units='L_px in video pixels; L_tank = L_px / (image_size_px - 1) -> pass multipole_sources(..., L=L_tank)',
                   method='median over recordings of fish_lengths.py (median midline length of the straightest clean frames)',
                   recordings=list(T.index), fish=fixed), open(OUT_JSON, 'w'), indent=1)
    fig, ax = plt.subplots(figsize=(11, 4.5)); colors = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4']
    for k, L in enumerate(T.columns):
        v = T[L]; x = k + np.linspace(-0.25, 0.25, len(v))
        ax.scatter(x, v, color=colors[k % 4], s=28, zorder=3)
        ax.hlines(med[L], k - 0.35, k + 0.35, color='k', lw=2, zorder=4)
        ax.text(k + 0.38, med[L], f'{med[L]:.1f} px', va='center', fontsize=9)
        for xi, (rec, val) in zip(x, v.items()):
            if np.isfinite(val) and abs(val / med[L] - 1) > OUTLIER: ax.annotate(rec, (xi, val), xytext=(4, 0), textcoords='offset points', fontsize=7)
    ax.set_xticks(range(len(T.columns)), [f'fish {L}' for L in T.columns]); ax.set_ylabel('L (px, midline of straight frames)')
    ax.set_title(f'Fish length per recording ({len(T)} recordings); black line = median (fixed L); labelled = > {OUTLIER:.0%} from median', loc='left')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e5e0'); ax.set_axisbelow(True)
    os.makedirs('figures', exist_ok=True); fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
    pd.set_option('display.width', 200)
    print('L_px per recording:'); print(T.round(1).to_string())
    for L, r in fixed.items():
        print(f"fish {L}: fixed L = {r['L_px']:.2f} px = {r['L_tank']:.5f} tank units (median of {r['n_recordings']} recordings, "
              f"IQR {r['L_px_iqr'][0]}-{r['L_px_iqr'][1]}, range {r['L_px_range'][0]}-{r['L_px_range'][1]}; outliers: {r['outlier_recordings'] or 'none'})")
    print(f'saved {OUT_JSON} and {OUT_FIG}')


if __name__ == '__main__':
    main()
