"""Polarity of each electrode pair on the 4-fish Kaden recordings, from Denis' mask-midline run (denis_mask_midline_kaden_4fish.py).
No ground truth emitter here, so leave one pair out: the emitter is the fish with the highest cosine similarity on the other 9
pairs (kept when it is clear: best >= MIN_SIM and best - second >= MIN_MARGIN); then the left-out pair's recorded sign is
compared with that fish's predicted sign, where both are strong (|value| > STRONG of the max over the 10 pairs).
A correctly wired pair agrees most of the time, a flipped one almost never. Also: mean best similarity (all 10 pairs) with the
file's polarity vs pairs 2 and 5 flipped (the 2-fish file's polarity).

    python polarity_check_4fish_kaden.py [--offset +0]
"""
import argparse, glob
import numpy as np, pandas as pd
DIR = 'results/eod_assign_4fish/denis_mask_midline_kaden'
MIN_SIM, MIN_MARGIN, STRONG = 0.9, 0.1, 0.2

def cos(R, U):          # R (n, k), U (n, 4, k) -> (n, 4)
    return np.einsum('nk,nfk->nf', R, U) / (np.linalg.norm(R, axis=1)[:, None] * np.linalg.norm(U, axis=2))

ap = argparse.ArgumentParser(); ap.add_argument('--offset', default='+0'); a = ap.parse_args()
rows, glob_rows = [], []
for path in sorted(glob.glob(f'{DIR}/denis_mask_midline_*_offset{a.offset}.csv')):
    t = path.split('/')[-1].removeprefix('denis_mask_midline_').split('_offset')[0]
    d = pd.read_csv(path, usecols=lambda c: 'PTP' in c or c == 'discharging_fish').dropna()
    R = d[[f'recorded_PTP_pair{p}' for p in range(1, 11)]].to_numpy()
    U = np.stack([d[[f'predicted_PTP_{f}_pair{p}' for p in range(1, 11)]].to_numpy() for f in 'ABCD'], axis=1)
    Rn = R / np.abs(R).max(1, keepdims=True); Un = U / np.abs(U).max(2, keepdims=True)
    for p in range(10):
        keep = np.arange(10) != p
        S = np.sort(cos(R[:, keep], U[:, :, keep]), axis=1)
        em = cos(R[:, keep], U[:, :, keep]).argmax(1)
        clear = (S[:, -1] >= MIN_SIM) & (S[:, -1] - S[:, -2] >= MIN_MARGIN)
        r, u = Rn[:, p], Un[np.arange(len(d)), em, p]
        s = clear & (np.abs(r) > STRONG) & (np.abs(u) > STRONG)
        rows.append(dict(recording=t, pair=p + 1, n=s.sum(), sign_agree=(np.sign(r[s]) == np.sign(u[s])).mean()))
    flip = np.ones(10); flip[[1, 4]] = -1
    glob_rows.append(dict(recording=t, n=len(d), mean_best_sim_file=cos(R, U).max(1).mean(),
                          mean_best_sim_pairs_2_5_flipped=cos(R, U * flip).max(1).mean()))
T = pd.DataFrame(rows); G = pd.DataFrame(glob_rows)
W = T.pivot(index='recording', columns='pair', values='sign_agree')
pd.set_option('display.width', 220)
print(f'sign agreement of the left-out pair (offset {a.offset}):'); print(W.round(2).to_string())
print('\nn strong clear EODs per pair:'); print(T.pivot(index='recording', columns='pair', values='n').to_string())
print(); print(G.round(3).to_string(index=False))
T.to_csv(f'{DIR}/polarity_check/leave_one_pair_out_offset{a.offset}.csv', index=False)
G.to_csv(f'{DIR}/polarity_check/mean_best_similarity_offset{a.offset}.csv', index=False)
