"""Fixed body length L per fish, measured once on straight, clean masks of one recording.

For each fish (efish4-track / Kaden letters A-D):
  1. every STRIDE-th frame: keep masks without contact (Kaden's occluded / needs_review flags, and the masks of
     two fish not within CONTACT_GAP px of each other), area within AREA_TOL of the fish's median area, a single
     connected piece, at least EDGE_PX from the image edge; rank them by elongation (PCA of the mask pixels)
  2. the slice-spline midline (fish_midline_with_head) of the N_CANDIDATES most elongated masks
  3. straight: the midline stays within MAX_DEV_PX of the straight line between its two ends (total turning is
     not used: on a nearly straight fish it still adds up to 10-30 deg from tiny direction changes near the tips)
  4. L = median midline length of the N_USE straightest of those (robust to a single odd mask); the plotted
     frame is the straight frame whose length is closest to that median

Units: midlines from midline_methods are in [0, 1] tank coordinates (pixel / (P - 1)), and multipole_sources
measures/uses L in those units, so pass  L = L_tank  (= L_px / (P - 1)). L_px is saved too.

    python fish_lengths.py                              # recording 2024-02-20T12_32_58
    python fish_lengths.py --pkl <other efish4-track .pkl> --tag <name>
    python fish_lengths.py --pkl <.pkl> --tag <name> --json_dir <dir> --fig_dir <dir>   other experiments (kept out of fish_lengths_combine.py)
"""
import argparse, json, os, pickle
import numpy as np
from scipy.ndimage import label
import midline_methods as mm
from io_utils import rle_decode, unpack_crop

PKL = 'data/track_Kaden/results_12_32_58_full/video2024-02-20T12_32_58/outputs/video2024-02-20T12_32_58.pkl'
STRIDE, CONTACT_GAP, AREA_TOL, EDGE_PX = 5, 3, 0.10, 10
N_CANDIDATES, MAX_DEV_PX, N_USE, MIN_USE = 300, 2.5, 30, 5


def elongation(m):
    '''sqrt of the ratio of the two principal variances of the mask pixels: high for a straight, thin body'''
    pix = np.argwhere(m).astype(float); ev = np.linalg.eigvalsh(np.cov(pix.T))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-9)))


def zeus_lengths(pkl, fish):
    '''Fixed L per fish label from a ZEUS tracking pkl ({'frames': {k: labels, masks, boxes, areas}, 'shape'}), with the rules
       above (every STRIDE-th frame; clean, contact-free masks; the N_CANDIDATES most elongated; median midline length of the
       N_USE straightest). Returns {label: dict(L_px, L_tank, n_frames_used, L_px_range, clean_candidates, frames_used)}.
       pkl can also be the loaded dict itself (e.g. with relabelled fish).'''
    if isinstance(pkl, dict):
        trk = pkl
    else:
        with open(pkl, 'rb') as f:
            trk = pickle.load(f)
    frames, P = trk['frames'], trk['shape'][0]

    def masks_of(k):                                           # full-frame mask per fish (None if missing or duplicated)
        fr = frames[k]; labs = list(fr['labels']); out = [None] * len(fish)
        for l, p, (y0, y1, x0, x1) in zip(labs, fr['masks'], fr['boxes']):
            if l in fish and labs.count(l) == 1:
                m = np.zeros((P, P), bool); m[y0:y1, x0:x1] = unpack_crop(p, (y0, y1, x0, x1))
                out[fish.index(l)] = m
        return out

    ks = sorted(frames)
    area = {l: [] for l in fish}
    for k in ks:
        for l, a in zip(frames[k]['labels'], frames[k]['areas']):
            if l in fish: area[l].append(a)
    med_area = {l: np.median(area[l]) for l in fish}
    cand = {l: [] for l in fish}
    for k in ks[::STRIDE]:
        ms = masks_of(k)
        contact, _ = mm.contact_frames(lambda _k: ms, [k], gap=CONTACT_GAP)
        for i, l in enumerate(fish):
            m = ms[i]
            if m is None or contact[0, i] or abs(m.sum() / med_area[l] - 1) > AREA_TOL: continue
            r, c = np.nonzero(m)
            if min(r.min(), c.min(), P - 1 - r.max(), P - 1 - c.max()) < EDGE_PX: continue
            if label(m, structure=np.ones((3, 3)))[1] != 1: continue
            cand[l].append((elongation(m), k))
    out = {}
    for i, l in enumerate(fish):
        rows = []
        for _, k in sorted(cand[l], reverse=True)[:N_CANDIDATES]:
            px = mm.fish_midline_with_head(masks_of(k)[i])['midline'] * (P - 1)
            u = (px[-1] - px[0]) / np.linalg.norm(px[-1] - px[0]); d = px - px[0]
            rows.append(dict(frame=k, L_px=np.linalg.norm(np.diff(px, axis=0), axis=1).sum(), dev=np.abs(d[:, 0] * u[1] - d[:, 1] * u[0]).max()))
        use = sorted([r for r in rows if r['dev'] <= MAX_DEV_PX], key=lambda r: r['dev'])[:N_USE]
        Ls = np.array([r['L_px'] for r in use])
        out[l] = dict(L_px=round(float(np.median(Ls)), 3), L_tank=round(float(np.median(Ls)) / (P - 1), 6), n_frames_used=len(use),
                      L_px_range=[round(float(Ls.min()), 2), round(float(Ls.max()), 2)], clean_candidates=len(cand[l]),
                      frames_used=[int(r['frame']) for r in use])
    return out


def main():
    global STRIDE, N_CANDIDATES, MAX_DEV_PX, MIN_USE
    ap = argparse.ArgumentParser(); ap.add_argument('--pkl', default=PKL); ap.add_argument('--tag', default='12_32_58')
    ap.add_argument('--json_dir', default='results'); ap.add_argument('--fig_dir', default='figures')   # other recordings: keep apart from the 4-fish files
    ap.add_argument('--stride', type=int, default=STRIDE); ap.add_argument('--candidates', type=int, default=N_CANDIDATES)
    ap.add_argument('--max_dev', type=float, default=MAX_DEV_PX); ap.add_argument('--min_use', type=int, default=MIN_USE)
    a = ap.parse_args()
    STRIDE, N_CANDIDATES, MAX_DEV_PX, MIN_USE = a.stride, a.candidates, a.max_dev, a.min_use
    import matplotlib; matplotlib.use('Agg')                 # here, so importing rle_decode leaves a notebook's backend alone
    import matplotlib.pyplot as plt
    K = pickle.load(open(a.pkl, 'rb')); LET = K['letters']; P = K['image_size'][0]; n = K['n_frames']
    flagged = K['occluded'] | K['needs_review']
    med_area = [np.median(K['area_visible'][:, i][K['area_visible'][:, i] > 0]) for i in range(len(LET))]

    # 1. clean, contact-free, elongated masks (every STRIDE-th frame)
    cand = {L: [] for L in LET}
    for f in range(0, n, STRIDE):
        masks = [None if r is None else rle_decode(r) for r in K['mask_visible'][f]]
        contact, _ = mm.contact_frames(lambda _f: masks, [f], gap=CONTACT_GAP)
        for i, L in enumerate(LET):
            m = masks[i]
            if m is None or flagged[f, i] or contact[0, i]: continue
            if abs(m.sum() / med_area[i] - 1) > AREA_TOL: continue
            r, c = np.nonzero(m)
            if min(r.min(), c.min(), P - 1 - r.max(), P - 1 - c.max()) < EDGE_PX: continue
            if label(m, structure=np.ones((3, 3)))[1] != 1: continue
            cand[L].append((elongation(m), f))
    results, chosen = {}, {}
    fig, axs = plt.subplots(2, len(LET), figsize=(5 * len(LET), 9), gridspec_kw=dict(height_ratios=[3, 1]))
    for i, L in enumerate(LET):
        top = sorted(cand[L], reverse=True)[:N_CANDIDATES]
        # 2-3. midlines of the most elongated masks; keep the straight ones
        rows = []
        for _, f in top:
            m = rle_decode(K['mask_visible'][f][i]); body = mm.fish_midline_with_head(m)
            px = body['midline'] * (P - 1)
            length = np.linalg.norm(np.diff(px, axis=0), axis=1).sum()
            u = (px[-1] - px[0]) / np.linalg.norm(px[-1] - px[0]); d = px - px[0]
            dev = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0]).max()                   # distance from the tip-to-tip line
            rows.append(dict(frame=f, L_px=length, dev=dev, turning=mm.turning_deg(px)))
        straight = sorted([r for r in rows if r['dev'] <= MAX_DEV_PX], key=lambda r: r['dev'])
        if len(straight) < MIN_USE:                                            # not enough: no L from this recording
            results[L] = dict(L_px=None, L_tank=None, n_frames_used=len(straight), clean_candidates=len(cand[L]), straight_found=len(straight))
            axs[0, i].set_title(f'fish {L}: only {len(straight)} straight clean frames, no L', fontsize=10); axs[0, i].axis('off'); axs[1, i].axis('off')
            continue
        use = straight[:N_USE]; Ls = np.array([r['L_px'] for r in use]); L_px = float(np.median(Ls))
        rep = min(use, key=lambda r: abs(r['L_px'] - L_px))                       # frame shown
        results[L] = dict(L_px=round(L_px, 3), L_tank=round(L_px / (P - 1), 6), frame_shown=int(rep['frame']),
                          L_px_shown_frame=round(rep['L_px'], 3), n_frames_used=len(use), frames_used=[int(r['frame']) for r in use],
                          L_px_spread=[round(float(Ls.min()), 2), round(float(np.percentile(Ls, 25)), 2), round(float(np.percentile(Ls, 75)), 2), round(float(Ls.max()), 2)],
                          max_deviation_px_used=round(max(r['dev'] for r in use), 2), clean_candidates=len(cand[L]), straight_found=len(straight))
        # plot the chosen frame with its midline and the straight line between the tips
        m = rle_decode(K['mask_visible'][rep['frame']][i]); body = mm.fish_midline_with_head(m); px = body['midline'] * (P - 1)
        ax = axs[0, i]                                                  # the whole tank, same scale in every panel
        ax.imshow(m, cmap='gray', extent=(-.5, P - .5, P - .5, -.5), interpolation='nearest')
        ax.plot([px[0, 1], px[-1, 1]], [px[0, 0], px[-1, 0]], color='#eda100', lw=1, ls='--', label='straight line tail tip - head tip')
        ax.plot(px[:, 1], px[:, 0], color='#2a78d6', lw=1.5, label='slice-spline midline')
        ax.plot(px[-1, 1], px[-1, 0], 'o', color='#2a78d6', mec='k', ms=5, label='head end')
        ax.set_xlim(-.5, P - .5); ax.set_ylim(P - .5, -.5)
        ax.set_title(f'fish {L}, frame {rep["frame"]}\nL = {L_px:.1f} px = {L_px / (P - 1):.4f} tank units\n'
                     f'(this frame {rep["L_px"]:.1f} px; max {rep["dev"]:.1f} px from a straight line)', fontsize=10)
        ax.set_xlabel('x (px)', fontsize=8); ax.set_ylabel('y (px)', fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='lower right')
        b = axs[1, i]; b.hist([r['L_px'] for r in straight], bins=25, color='#a3a29b', label='all straight frames')
        b.hist(Ls, bins=25, color='#2a78d6', label=f'{len(use)} straightest (used)'); b.axvline(L_px, color='k', lw=1)
        b.set_xlabel('midline length (px)'); b.set_ylabel('frames'); b.legend(fontsize=7); b.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    os.makedirs(a.fig_dir, exist_ok=True); os.makedirs(a.json_dir, exist_ok=True)
    fig_path = f'{a.fig_dir}/fish_lengths_{a.tag}.png'; fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    out = dict(recording=K.get('recording', a.tag), source=a.pkl, image_size_px=int(P),
               units='L_px in video pixels; L_tank = L_px / (image_size_px - 1), the [0, 1] tank units of the midlines -> '
                     'pass multipole_sources(..., L=L_tank)',
               method=dict(stride=STRIDE, contact_gap_px=CONTACT_GAP, area_tolerance=AREA_TOL, edge_px=EDGE_PX, candidates=N_CANDIDATES,
                           max_deviation_from_straight_px=MAX_DEV_PX, n_used=N_USE, midline='fish_midline_with_head (slice spline)'),
               fish=results)
    json_path = f'{a.json_dir}/fish_lengths_{a.tag}.json'; json.dump(out, open(json_path, 'w'), indent=1)
    for L, r in results.items():
        if r['L_px'] is None: print(f"fish {L}: no L ({r['straight_found']} straight of {r['clean_candidates']} clean masks)"); continue
        print(f"fish {L}: L = {r['L_px']:.2f} px = {r['L_tank']:.5f} tank units  (median of {r['n_frames_used']} straight frames, "
              f"range {r['L_px_spread'][0]}-{r['L_px_spread'][3]} px, IQR {r['L_px_spread'][1]}-{r['L_px_spread'][2]}; "
              f"shown frame {r['frame_shown']}; {r['straight_found']} straight of {r['clean_candidates']} clean masks)")
    print(f'saved {json_path} and {fig_path}')


if __name__ == '__main__':
    main()
