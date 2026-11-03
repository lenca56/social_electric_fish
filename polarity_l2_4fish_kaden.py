"""Per-pair L2 error of Denis' mask-midline prediction on the 4-fish Kaden recordings, each pair as wired in the 4-fish
electrode file vs flipped (denis_mask_midline_kaden_4fish.py output). Patterns max-normalized over the 10 pairs.

1. Per pair p, leave one pair out: emitter = fish with the highest cosine similarity on the other 9 pairs (clear: best >= 0.9,
   best - second >= 0.1); error of pair p |r_p - u_p| (as wired) vs |r_p + u_p| (flipped), on EODs where |u_p| > 0.2.
2. Whole pattern: for the polarity as wired and for each pair flipped in turn (and pairs 2 + 5 together), the best fish is
   picked again and the L2 of its pattern to the recording is taken; median over EODs.

    python polarity_l2_4fish_kaden.py [--offset +0]
"""
import argparse, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DIR = 'results/eod_assign_4fish/denis_mask_midline_kaden'
OUT = f'{DIR}/polarity_check'
MIN_SIM, MIN_MARGIN, STRONG = 0.9, 0.1, 0.2
cos = lambda R, U: np.einsum('nk,nfk->nf', R, U) / (np.linalg.norm(R, axis=1)[:, None] * np.linalg.norm(U, axis=2))
nrm = lambda X: X / np.abs(X).max(-1, keepdims=True)

ap = argparse.ArgumentParser(); ap.add_argument('--offset', default='+0'); a = ap.parse_args()
per_pair, whole = [], []
for path in sorted(glob.glob(f'{DIR}/denis_mask_midline_*_offset{a.offset}.csv')):
    t = path.split('/')[-1].removeprefix('denis_mask_midline_').split('_offset')[0]
    d = pd.read_csv(path, usecols=lambda c: 'PTP' in c).dropna()
    R = d[[f'recorded_PTP_pair{p}' for p in range(1, 11)]].to_numpy()
    U = np.stack([d[[f'predicted_PTP_{f}_pair{p}' for p in range(1, 11)]].to_numpy() for f in 'ABCD'], axis=1)
    Rn, Un, n = nrm(R), nrm(U), len(d)
    for p in range(10):
        keep = np.arange(10) != p
        C = cos(R[:, keep], U[:, :, keep]); S = np.sort(C, axis=1); em = C.argmax(1)
        r, u = Rn[:, p], Un[np.arange(n), em, p]
        s = (S[:, -1] >= MIN_SIM) & (S[:, -1] - S[:, -2] >= MIN_MARGIN) & (np.abs(u) > STRONG)
        e_w, e_f = np.abs(r[s] - u[s]), np.abs(r[s] + u[s])
        per_pair.append(dict(recording=t, pair=p + 1, n=s.sum(), median_err_as_wired=np.median(e_w), median_err_flipped=np.median(e_f),
                             mean_sq_err_as_wired=np.mean(e_w ** 2), mean_sq_err_flipped=np.mean(e_f ** 2), flip_lowers_err=np.mean(e_f < e_w)))
    hyps = {'as wired': np.ones(10)} | {f'pair {p + 1} flipped': np.where(np.arange(10) == p, -1.0, 1.0) for p in range(10)}
    hyps['pairs 2 + 5 flipped'] = np.where(np.isin(np.arange(10), [1, 4]), -1.0, 1.0)
    row = dict(recording=t, n=n)
    for h, fl in hyps.items():
        Uh = Un * fl; best = cos(R, Uh).argmax(1)
        row[h] = np.median(np.linalg.norm(Rn - Uh[np.arange(n), best], axis=1))
    whole.append(row)
P, W = pd.DataFrame(per_pair), pd.DataFrame(whole)
P.to_csv(f'{OUT}/per_pair_error_flip_offset{a.offset}.csv', index=False); W.to_csv(f'{OUT}/whole_pattern_l2_flip_offset{a.offset}.csv', index=False)
pd.set_option('display.width', 250)
for col in ('median_err_as_wired', 'median_err_flipped', 'flip_lowers_err', 'n'):
    print(f'\n{col}:'); print(P.pivot(index='recording', columns='pair', values=col).round(3).to_string())
print('\nmedian whole-pattern L2 of the best fish, per polarity hypothesis:'); print(W.set_index('recording').round(3).T.to_string())

fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
g = P.groupby('pair')
x = np.arange(1, 11)
for col, c, lab, dx in [('median_err_as_wired', '#1F5FBF', 'as wired (4-fish file)', -0.2), ('median_err_flipped', '#C0392B', 'pair flipped', 0.2)]:
    ax[0].bar(x + dx, g[col].median(), 0.4, color=c, label=lab)
    for _, r in P.iterrows(): ax[0].plot(r.pair + dx, r[col], 'k.', ms=3)
for p in (2, 5): ax[0].axvspan(p - 0.5, p + 0.5, color='#F2C94C', alpha=0.25, lw=0)
ax[0].set(xticks=x, xlabel='electrode pair', ylabel='median |recorded - predicted| of that pair',
          title=f'Per-pair error, emitter from the other 9 pairs (offset {a.offset})\nbars: median over recordings, dots: each recording')
ax[0].legend()
hy = [c for c in W.columns if c not in ('recording', 'n')]
for _, r in W.iterrows(): ax[1].plot(range(len(hy)), r[hy].to_numpy(float), '.-', color='0.6', lw=1)
ax[1].plot(range(len(hy)), W[hy].median().to_numpy(), 'ko-', lw=2, label='median over recordings')
ax[1].set_xticks(range(len(hy))); ax[1].set_xticklabels(hy, rotation=45, ha='right')
ax[1].set(ylabel='median L2, best fish vs recorded', title='Whole pattern: polarity as wired vs one pair flipped'); ax[1].legend()
plt.tight_layout(); fig.savefig(f'{OUT}/per_pair_error_flip_offset{a.offset}.png', dpi=130)
