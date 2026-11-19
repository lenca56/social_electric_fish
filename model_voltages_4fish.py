"""Unit-current model voltages of every EOD frame of a 4-fish recording: my model, BEM conductivities (interior 100 x water,
skin 0.05 x water over 0.005 cm), sink at the tail tip, each fish in turn the emitter with the other three as bodies.

The voltages depend only on the frame's masks, so they are solved once per frame that has an EOD (mask frame = Frame: the head
and tail keypoints stored in each EOD row lie on the masks of frame Frame). Saved as {frame: (4, n_pairs)} (NaN for a fish
without a mask or midline), checkpointed so that a run can resume:

    python model_voltages_4fish.py --rec 2024-02-20T21_16_26 --workers 7

Fish order (0-3): 'area_rank' ranks the four masks of each frame by area (largest = 0; the tracker ids of that .mgz swap);
'zeus' orders the ZEUS labels by size (largest = 0, remnants dropped); 'kaden' uses Kaden's letters A-D (io_utils.load_masks).
"""
import argparse, os, pickle, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from compute_electric_field import theta_outer, multipole_sources, build_multipole_b, occupancy_field, unflatten_to_grid, gammas
from midline_methods import fish_midline_with_head
from eod_assign import pair_voltages
from io_utils import load_electrodes, load_masks, load_eods, unpack_crop

FEB, OCT = 'data/electrodes/electrodes_signed_px_coord_4-fish.csv', 'data/electrodes/electrodes_signed_px_coord_2024-10-23.csv'
KADEN = 'data/track_Kaden/results_{t}_full/video2024-02-20T{t}/outputs/video2024-02-20T{t}.pkl'
KADEN_CSV = 'data/Denis Assignment/with Kaden Tracking/Data_4fish_2024-02-20T{t}_no_ref_per.csv'
RECS = {
    '2024-02-20T21_16_26': dict(tracking='Shiv', masks='results/isolated_frames/video2024-02-20T21_16_26_masks_cropped.pkl', identity='area_rank',
                                eod='data/track_Shiv/Group_01/Data_4fish_2024-02-20T21_16_26_no_ref_per.csv', electrodes=FEB, period='Feb'),
    '2024-02-20T18_16_24': dict(tracking='Shiv', masks='data/elephantnose_18_16_24_tracking_final.pkl', identity='zeus',
                                eod='data/track_Shiv/Group_01/Data_4fish_2024-02-20T18_16_24_no_ref_per.csv', electrodes=FEB, period='Feb'),
    '2024-10-23T15_18_28': dict(tracking='Shiv', masks='data/elephantnose_2024-10-23_15_18_28_tracking_final.pkl', identity='zeus',
                                eod='data/track_Shiv/Group_02/Data_4fish_2024-10-23T15_18_28_no_ref_per.csv', electrodes=OCT, period='Oct'),
    **{f'2024-02-20T{t}': dict(tracking='Kaden', masks=KADEN.format(t=t), identity='kaden', eod=KADEN_CSV.format(t=t), electrodes=FEB,
                               period='Feb', lengths=f'results/fish_lengths_{t}.json')
       for t in ('12_32_58', '15_16_23', '17_56_24')},
}
OUT = 'results/eod_assign_4fish'

# model: as EOD_assignment_3_fish_conductivities.ipynb (BEM set), sink at the tail tip
P, N, N_POLES, SINK = 912, 150, 100, 0.0
H = 1.0 / (N - 1)
CELL_CM = (P - 1) / (N - 1) / (P / 60)
SKIN_CM, LAMBDA_CM = 0.005, 1e-4 / 1e-3
MAT = dict(sigma_water=1.0, eps_water=0.0, sigma_interior=100.0, eps_interior=0.0,
           sigma_skin=1 / (LAMBDA_CM / SKIN_CM + 1 / 100.0), eps_skin=0.0)
DELTA_H = SKIN_CM / CELL_CM
GAM = gammas(0.0, **MAT)


def build_A_fast(occ, N, delta_h, g_water, g_skin, g_interior):
    '''build_A_shell, vectorized (as EOD_assignment_3_fish_conductivities.ipynb, checked there entry for entry).'''
    tx, ty = theta_outer(occ)
    gnode = np.where(occ > 0.5, g_interior, g_water).astype(complex)
    skin = delta_h * (1 / g_skin - 1 / g_interior)
    def w(gC, gM, thc):
        with np.errstate(invalid='ignore'):
            return np.where(np.isnan(thc), gC, 1.0 / (thc / gC + (1 - thc) / gM + skin))
    k = np.arange(N * N).reshape(N, N)
    W = {'up': w(gnode[1:, :], gnode[:-1, :], 1 - tx), 'down': w(gnode[:-1, :], gnode[1:, :], tx),
         'left': w(gnode[:, 1:], gnode[:, :-1], 1 - ty), 'right': w(gnode[:, :-1], gnode[:, 1:], ty)}
    src = {'up': k[1:, :], 'down': k[:-1, :], 'left': k[:, 1:], 'right': k[:, :-1]}
    dst = {'up': k[:-1, :], 'down': k[1:, :], 'left': k[:, :-1], 'right': k[:, 1:]}
    full = {d: np.zeros((N, N), complex) for d in W}
    full['up'][1:, :], full['down'][:-1, :] = W['up'], W['down']
    full['left'][:, 1:], full['right'][:, :-1] = W['left'], W['right']
    ws = ((0j + full['up']) + full['down']) + full['left'] + full['right']
    r = np.concatenate([src[d].ravel() for d in W] + [k.ravel()])
    c = np.concatenate([dst[d].ravel() for d in W] + [k.ravel()])
    v = np.concatenate([W[d].ravel() for d in W] + [-ws.ravel()])
    return sp.coo_matrix((v, (r, c)), shape=(N * N, N * N), dtype=complex).tocsr()


def solve(A, b, pin=0):
    '''solve_potential (pinned node, zero-mean gauge), real arithmetic (omega = 0).'''
    M = (-A).tocsr().real
    keep = np.ones(M.shape[0]); keep[pin] = 0
    M = (sp.diags(keep) @ M + sp.csr_matrix(([1.0], ([pin], [pin])), shape=M.shape)).tocsc()
    rhs = np.asarray(b).ravel().real.copy(); rhs[pin] = 0
    phi = splu(M).solve(rhs); return phi - phi.mean()


ZEUS_BY_SIZE = (1, 2, 3, 0)                         # ZEUS labels, largest fish first: fish 0-3


def video_masks(cfg):
    '''masks[k] -> (4, P, P) bool masks of frame k in fish order 0-3 (empty where a fish has no mask).'''
    if cfg['identity'] == 'kaden': return load_masks(cfg['masks'])
    if cfg['identity'] == 'zeus': return load_masks(cfg['masks'], 'shiv', fish_order=ZEUS_BY_SIZE)
    return AreaRankMasks(cfg['masks'])


class AreaRankMasks:
    '''The cropped masks of 2024-02-20T21_16_26 (no ZEUS label key; its tracker ids swap): each frame's four masks ranked by
       area (largest = 0); with fewer than four masks, each goes to the fish of nearest median area.'''
    def __init__(self, path):
        d = pickle.load(open(path, 'rb')); assert d['shape'][0] == P
        self.frames, self.n = d['frames'], len(d['frames'])
        area = pd.DataFrame([(l, a) for k in self.frames for l, a in zip(self.frames[k]['labels'], self.frames[k]['areas']) if l in (0, 1, 2, 3)],
                            columns=['l', 'a'])
        self.med_log = np.log(area.groupby('l').a.median().sort_values(ascending=False).to_numpy())   # median area of fish 0..3

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        m = np.zeros((4, P, P), bool)
        fr = self.frames.get(k)
        if fr is None: return m
        items = [(l, p, b, a) for l, p, b, a in zip(fr['labels'], fr['masks'], fr['boxes'], fr['areas']) if l in (0, 1, 2, 3)]
        if sorted(l for l, *_ in items) == [0, 1, 2, 3]:
            fish = list(np.argsort(np.argsort([-a for *_, a in items], kind='stable'), kind='stable'))   # rank by area
        else:                                                                # fewer than 4 masks: nearest median area, each fish once
            fish, free = [None] * len(items), set(range(4))
            for j in np.argsort([-a for *_, a in items], kind='stable'):
                f = min(free, key=lambda f: abs(np.log(max(items[j][3], 1)) - self.med_log[f])); fish[j] = f; free.discard(f)
        for f, (l, p, (y0, y1, x0, x1), a) in zip(fish, items):
            m[f, y0:y1, x0:x1] |= unpack_crop(p, (y0, y1, x0, x1))
        return m


def frame_voltages(k):
    fm = MASKS[k]
    u = np.full((4, len(PAIRS)), np.nan)
    for f in range(4):
        if not fm[f].any(): continue
        try:
            mid = fish_midline_with_head(fm[f], n_ctrl=N_POLES + 1)['midline']
        except Exception:
            continue
        if mid is None or not np.isfinite(mid).all(): continue
        b = build_multipole_b(*multipole_sources(mid, I0=1.0, sink_frac=SINK), H, N)
        u[f] = pair_voltages(unflatten_to_grid(solve(build_A_fast(occupancy_field(fm, N, skip=f), N, DELTA_H, *GAM), b), N).real, PAIRS, N)
    return k, u


def main():
    global MASKS, PAIRS
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--save-every', type=int, default=3000)
    a = ap.parse_args()
    cfg = RECS[a.rec]
    os.makedirs(f'{OUT}/{a.rec}', exist_ok=True)
    path = f'{OUT}/{a.rec}/model_voltages_N{N}_sink0_bem_by_frame.pkl'
    MASKS, PAIRS = video_masks(cfg), load_electrodes(cfg['electrodes'], P)
    frames = np.unique(load_eods(cfg['eod'], columns=['Frame'], traces=False)[0].Frame.to_numpy())
    done = pd.read_pickle(path) if os.path.exists(path) else {}
    todo = [int(k) for k in frames if int(k) not in done]
    print(f'{a.rec}: {len(frames)} EOD frames, {len(done)} cached, {len(todo)} to solve', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('fork')) as pool:
        for j, (k, u) in enumerate(pool.map(frame_voltages, todo, chunksize=8), 1):
            done[k] = u
            if j % a.save_every == 0 or j == len(todo):
                pd.to_pickle(done, path + '.tmp'); os.replace(path + '.tmp', path)
                el = time.time() - t0
                print(f'  {j}/{len(todo)} frames, {el / 60:.1f} min, ~{el / j * (len(todo) - j) / 60:.0f} min left', flush=True)
    pd.to_pickle(done, path + '.tmp'); os.replace(path + '.tmp', path)
    print('done', path, flush=True)


if __name__ == '__main__':
    main()
