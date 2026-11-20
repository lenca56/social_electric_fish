"""Double-detected EODs in the 3-fish EOD tables (Data_3fish_<rec>.csv): find them and list the rows to drop.

What they are: consecutive EODs less than DT_MIN_MS apart (1-7 ms; the emitter's own intervals are >= ~7.5 ms, with a peak at 8-10 ms
in the later Oct 15 recordings whose pairs look alike: correlation ~0.97, amplitude ratio ~0.95). In such a close pair one member is the
real EOD (it looks like the EODs around it: median correlation 0.94, normal amplitude) and the other is a small, broad second detection
(the tail of the same discharge or an artifact: ~0.2 x the recording's median amplitude, correlation with the EODs around it ~0.06),
which can come first or second.

Rule: while two consecutive kept EODs are closer than DT_MIN_MS, drop the one that is less similar to its neighbours (cosine of its 10 x
25-sample recorded traces with each of the N_NEIGHBOURS kept EODs before and after the pair, median), keep the other.

    python eod_double_detections_3fish.py
Output in results/eod_assignment_3_fish_silent/eod_intervals/:
  double_detections_3fish.csv   video, eod_row, Frame, Time, dt_ms (to its partner), partner_row, amp_rel, corr_neighbours,
                                partner_corr_neighbours: the EODs to drop
  eod_intervals_3fish_clean.png intervals between consecutive EODs before and after dropping them
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_kaden_3fish_common as C
from eod_assign import recorded_pair_traces_all

DT_MIN_MS = 7.0           # consecutive EODs closer than this are a double detection
N_NEIGHBOURS = 2          # EODs before and after the pair used to judge which member is the real EOD
N_SAMPLES = 25
OUT = 'results/eod_assignment_3_fish_silent/eod_intervals'


def find_doubles(video):
    d = pd.read_csv(C.EOD_TABLE.format(video=video), usecols=['Frame', 'Time'] + [f'electrode_{p}_time_{i}' for p in range(1, 11)
                                                                                 for i in range(1, N_SAMPLES + 1)])
    V = recorded_pair_traces_all(d, list(range(1, 11)), N_SAMPLES).reshape(len(d), -1)
    U = V / np.linalg.norm(V, axis=1, keepdims=True); amp = np.abs(V).max(1); t = d.Time.to_numpy()
    keep = np.ones(len(d), bool); rows = []
    while True:
        k = np.flatnonzero(keep); dt = np.diff(t[k]) * 1e3; close = np.flatnonzero(dt < DT_MIN_MS)
        if not len(close): break
        for c in close:                                         # pairs are disjoint enough; re-check after each sweep
            a, b = k[c], k[c + 1]
            if not (keep[a] and keep[b]): continue
            pos = np.flatnonzero(keep); ia = np.searchsorted(pos, a); ib = np.searchsorted(pos, b)
            nb = np.r_[pos[max(ia - N_NEIGHBOURS, 0):ia], pos[ib + 1:ib + 1 + N_NEIGHBOURS]]
            ca, cb = (np.median(U[x] @ U[nb].T) if len(nb) else 0.0 for x in (a, b))
            drop, other, cd, co = (a, b, ca, cb) if ca < cb else (b, a, cb, ca)
            keep[drop] = False
            rows.append(dict(video=video, eod_row=int(drop), Frame=int(d.Frame[drop]), Time=float(t[drop]), dt_ms=abs(t[b] - t[a]) * 1e3,
                             partner_row=int(other), amp_rel=amp[drop] / np.median(amp), corr_neighbours=cd, partner_corr_neighbours=co))
    return pd.DataFrame(rows), np.diff(t) * 1e3, np.diff(t[keep]) * 1e3


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = {v: find_doubles(v) for v in C.VIDEOS}
    T = pd.concat([r[0] for r in res.values()], ignore_index=True).sort_values(['video', 'eod_row'])
    T.to_csv(f'{OUT}/double_detections_3fish.csv', index=False)
    S = pd.DataFrame([dict(video=v, eods=len(r[1]) + 1, dropped=len(r[0]), min_interval_before_ms=r[1].min(), min_interval_after_ms=r[2].min())
                      for v, r in res.items()]).round(3)
    print(S.to_string(index=False)); print(f'dropped {len(T)} of {S.eods.sum():,} EODs; amplitude of the dropped ones (x median): '
                                            f'median {T.amp_rel.median():.2f}; their neighbour correlation median {T.corr_neighbours.median():.2f} '
                                            f'vs kept partner {T.partner_corr_neighbours.median():.2f}')
    # intervals before vs after
    plain = FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}')
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8)); bins = np.logspace(np.log10(0.8), np.log10(2000), 120); b2 = np.arange(0, 40.5, 0.5)
    before = np.concatenate([r[1] for r in res.values()]); after = np.concatenate([r[2] for r in res.values()])
    for ax, b in zip(axs, (bins, b2)):
        ax.hist(before, b, color='#b5b5af', label=f'all EODs ({len(before):,} intervals; min {before.min():.2f} ms)')
        ax.hist(after, b, histtype='step', color='#2a78d6', lw=1.6, label=f'double detections dropped ({len(after):,}; min {after.min():.2f} ms)')
        ax.set_yscale('log'); ax.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]))
        ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(plain); ax.axvline(DT_MIN_MS, color='#eb6834', ls='--', lw=1)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e6e6e3', lw=0.6); ax.set_ylabel('count (log spacing)')
    axs[0].set_xscale('log'); axs[0].xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
    axs[0].xaxis.set_minor_locator(NullLocator()); axs[0].xaxis.set_major_formatter(plain)
    axs[0].set(xlabel='interval between consecutive EODs (ms, log spacing)', title='all intervals, 8 recordings pooled')
    axs[1].set(xlabel='interval between consecutive EODs (ms)', title='0-40 ms (0.5 ms bins)', xlim=(0, 40)); axs[1].set_xticks(range(0, 41, 5))
    axs[0].legend(fontsize=8.5, frameon=False, loc='upper left')
    fig.suptitle(f'3-fish: intervals between consecutive EODs before and after dropping double detections (pairs closer than {DT_MIN_MS:g} ms: '
                 'the member less like its neighbouring EODs)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(f'{OUT}/eod_intervals_3fish_clean.png', dpi=110)
    print(f'written {OUT}/double_detections_3fish.csv and eod_intervals_3fish_clean.png')
