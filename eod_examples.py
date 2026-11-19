"""Example-EOD figures, my model vs Denis' pipeline: loading a recording, masks, video frames, midlines and the tank +
zoom + voltages figure (plot_example). Moved unchanged from EOD_examples_3fish_better_frames.ipynb."""
import os, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from midline_methods import fish_midline_with_head
from eod_assign import peak_pattern, rank1_pattern
from io_utils import load_electrodes, load_masks, load_eods

REC = {
    '3-fish': dict(
        data='data/3_fish_silent', eod_csv='Assigned_Data_3fish_2024-10-15T14_12_25.csv',
        track='elephantnose_2024-10-15T14-12-25_Shiv_tracking_final.pkl', video='video2024-10-15T14_12_25.avi',
        electrodes='data/electrodes/electrodes_signed_px_coord_3-fish.csv',
        res='results/eod_ground_truth_3fish/conductivities',
        cache='model_voltages_N150_sinks0and0.05_elc5c81b_bem_a453a8.pkl',
        denis_pkl='results/eod_assignment_3_fish_silent/electrodes_3-fish/Assigned_Data_3fish_2024-10-15T14_12_25.pkl',
        fish=(0, 1, 2), emitter=0, denis_map={'A': 1, 'B': 2, 'C': 0},
        fish_color={0: '#E0B000', 1: '#D45087', 2: '#2E8B57'}),
    '2-fish': dict(
        data='data/2_fish_silent', eod_csv='Data_1fish_2024-02-13T14_33_21.csv',
        track='elephantnose_2024-02-13T14-33-21_Shiv_tracking_final.pkl', video='video2024-02-13T14_33_21.avi',
        electrodes='data/electrodes/electrodes_signed_px_coord_2-fish.csv',
        res='results/eod_ground_truth/conductivities',
        cache='model_voltages_N150_sinks0and0.05_el46ee63_bem_a453a8.pkl',
        denis_pkl='results/eod_assignment_2_fish_silent/electrodes_2-fish/Assigned_Data_1fish_2024-02-13T14_33_21.pkl',
        fish=(0, 1), emitter=1, denis_map={'A': 1, 'B': 0},
        fish_color={0: '#2E8B57', 1: '#D45087'}),
}
MINE = 'BEM σ (interior 100, skin 0.05, 0.005 cm), sink at tail tip'   # column of eod_assignments.csv
DENIS = "Denis' pipeline (latest run)"
C_MINE, C_DENIS, C_REC = '#1F5FBF', '#E8710A', 'k'   # my model blue, Denis orange, recorded black
PX_PER_CM = 912 / 60                                  # Denis' pixel2cm
FS = 50000.0                                          # electrode sampling rate (Hz)
n_poles = 100
OUTCOMES = ['both right', 'only mine right', 'only Denis right', 'both wrong']
N_BINS, SEED = 5, 0
OUTLINE = [pe.withStroke(linewidth=3, foreground='k')]
C_PAIR = '#C00000'                                    # highlighted electrode pair


def load_recording(name):
    '''Everything the examples need for one recording, as a dict.'''
    c = dict(REC[name], name=name)
    masks = load_masks(f"{c['data']}/{c['track']}", 'shiv', fish_order=c['fish'])       # masks[k]: (n_fish, P, P)
    c.update(masks=masks, P=masks.shape[0], fps=masks.fps)
    c['pairs'] = load_electrodes(c['electrodes'], c['P'])
    c['pair_ids'] = [pid for pid, _, _ in c['pairs']]
    df, c['V'], _ = load_eods(f"{c['data']}/{c['eod_csv']}", pairs=c['pair_ids'])     # df: per-EOD info; V: (n_eods, pairs, samples)
    c['df'] = df; assert c['V'].shape[1] == 10
    c['assigned'] = pd.read_csv(f"{c['res']}/eod_assignments.csv", index_col='row')
    c['cache'] = pd.read_pickle(f"{c['res']}/{c['cache']}")
    c['den'] = pd.read_pickle(c['denis_pkl'])
    n = len(df)
    assert len(c['assigned']) == len(c['cache']) == len(c['den']) == n
    for t in (c['assigned'].Frame, c['cache'].frame, c['den'].Frame):
        assert (np.asarray(t) == df.Frame.to_numpy()).all()
    return c

def recorded_pair_traces(c, i):
    '''(n_pairs, n_samp) recorded voltages of EOD row i, each pair demeaned.'''
    V = c['V'][i]
    return V - np.nanmean(V, axis=1, keepdims=True)

def frame_masks(c, k):
    '''(n_fish, P, P) full-frame bool masks of frame k, indexed by fish (empty where a fish has no mask).'''
    return c['masks'][k]

def video_frame(c, k):
    '''Grayscale (P, P) frame k of the raw video (ffmpeg).'''
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f"{k / c['fps']:.4f}", '-i', f"{c['data']}/{c['video']}", '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(c['P'], c['P'])

def fish_midlines(fm):
    '''Midline (n_poles + 1, 2) in tank coords (row, col) / (P - 1), tail -> head, of every fish; None where it fails.'''
    out = []
    for m in fm:
        try:
            mid = fish_midline_with_head(m, n_ctrl=n_poles + 1)['midline'] if m.any() else None
            out.append(mid if mid is not None and np.isfinite(mid).all() else None)
        except Exception:
            out.append(None)
    return out

def denis_mask_of(c, d, X, fm, min_score=0.5):
    '''Mask label holding most confident SLEAP keypoints of Denis' fish X (-1 if none lies on a mask).'''
    votes = []
    for kp in ('mouth', 'head', 'middle', 'tail'):
        if not d[f'score_{kp}_{X}'] > min_score: continue
        x, y = np.round(np.asarray(d[f'{kp}_{X}']) * PX_PER_CM).astype(int)
        if 0 <= x < c['P'] and 0 <= y < c['P']: votes += [f for f in c['fish'] if fm[f][y, x]]
    return max(set(votes), key=votes.count) if votes else -1

def my_patterns(c, i):
    '''Recorded peak pattern r (max |r| = 1) and my model patterns (n_fish, n_pairs), sign fixed (x -1), max |u| = 1.'''
    r = np.asarray(c['cache'].v_rec[i]); r = r / np.abs(r).max()
    U = -np.asarray(c['cache']['u_sink0'][i]); U = U / np.abs(U).max(axis=1, keepdims=True)
    return r, U

def plot_example(c, i, head, path, pair=None):
    '''Tank + voltages figure of EOD row i; `head` starts the title, `pair` (id) is highlighted, saved to `path`.'''
    df, den, P, em, fish = c['df'], c['den'], c['P'], c['emitter'], c['fish']
    k = int(df.Frame[i]); fm = frame_masks(c, k); mids = fish_midlines(fm); dp = den.loc[i]
    letter = {f: X for X, f in c['denis_map'].items()}                       # mask label -> Denis' fish
    V = recorded_pair_traces(c, i); r, ts = peak_pattern(V); rmax = np.abs(r).max()
    r_c, U = my_patterns(c, i)
    assert np.allclose(r / rmax, r_c)                                        # same recording as the cached run
    w = rank1_pattern(V)[1]; w = w / w[ts]                                   # EOD waveform, 1 at the peak sample
    t_ms = (np.arange(V.shape[1]) - V.shape[1] // 2) / FS * 1e3
    sigma = np.sign(np.asarray(dp.recorded_PTP) @ r)                         # Denis' peak-to-peak sign -> peak-sample sign
    mine_pick, den_pick = int(c['assigned'].loc[i, f'fish | {MINE}']), int(c['assigned'].loc[i, f'fish | {DENIS}'])
    d_mine = np.linalg.norm(U - r_c, axis=1)

    nf = len(fish)
    fig = plt.figure(figsize=(30, 3.0 * nf + 2.6))
    gs = fig.add_gridspec(nf, 13, width_ratios=[2.7 * nf / 3 + 2.6, 3.0, 2.6] + [1] * 10, wspace=0.1, hspace=0.3)

    # tank
    a = fig.add_subplot(gs[:, 0])
    img = video_frame(c, k); a.imshow(img, cmap='gray')
    over = np.zeros((P, P, 4))
    for f in fish: over[fm[f]] = to_rgba(c['fish_color'][f], 0.45)
    a.imshow(over)
    for f in fish:
        if mids[f] is not None:
            mid = mids[f] * (P - 1)                                          # (row, col), tail -> head
            a.plot(mid[:, 1], mid[:, 0], color=C_MINE, lw=2, path_effects=OUTLINE)
            a.plot(mid[-1, 1], mid[-1, 0], 'o', color=C_MINE, mec='w', ms=8)
        dm = np.asarray(dp[f'currents_locations_{letter[f]}']) * PX_PER_CM   # (x, y) px, tail -> mouth
        a.plot(dm[:, 0], dm[:, 1], color=C_DENIS, lw=2, ls='--', path_effects=OUTLINE)
        a.plot(dm[-1, 0], dm[-1, 1], 's', color=C_DENIS, mec='w', ms=7)
        for kp, mk in (('mouth', '^'), ('head', 'o'), ('middle', 's'), ('tail', 'v')):
            x, y = np.asarray(dp[f'{kp}_{letter[f]}']) * PX_PER_CM
            a.plot(x, y, mk, color='w', mec='k', ms=6, mew=1)
        if fm[f].any():
            yx = np.argwhere(fm[f]); y, x = yx[np.argmax(yx[:, 0])]          # label just below the mask
            a.text(x, min(y + 25, P - 5), f"fish {f}" + (' (emitter)' if f == em else ''), color='w', fontsize=11,
                   fontweight='bold', ha='center', va='top', path_effects=OUTLINE)
    for pid, pp, pm in c['pairs']:                                           # electrodes: pair number and polarity
        for pt, sgn in ((pp, '+'), (pm, '−')):
            y, x = pt * (P - 1)
            a.plot(x, y, 'o', color='w', mec='k', ms=5)
            off = np.array([P / 2 - x, P / 2 - y]); off = 30 * off / np.linalg.norm(off)
            a.text(x + off[0], y + off[1], f'{pid}{sgn}', color='k', fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.85))
    for pid, pp, pm in c['pairs']:                                           # ring the highlighted pair's electrodes
        if pid == pair:
            for pt in (pp, pm): a.plot(*(pt * (P - 1))[::-1], 'o', ms=20, mfc='none', mec=C_PAIR, mew=2.5)
    a.set(xlim=(-10, P + 10), ylim=(P + 10, -10)); a.set_axis_off()
    a.legend(handles=[Patch(color=c['fish_color'][f], alpha=0.6, label=f'mask fish {f}') for f in fish] +
                     [Line2D([], [], color=C_MINE, lw=2, marker='o', mec='w', label='my midline (dot = head)'),
                      Line2D([], [], color=C_DENIS, lw=2, ls='--', marker='s', mec='w', label="Denis' midline (square = head)"),
                      Line2D([], [], marker='^', color='w', mec='k', ls='', label='SLEAP mouth'),
                      Line2D([], [], marker='o', color='w', mec='k', ls='', label='SLEAP head'),
                      Line2D([], [], marker='s', color='w', mec='k', ls='', label='SLEAP middle'),
                      Line2D([], [], marker='v', color='w', mec='k', ls='', label='SLEAP tail'),
                      Line2D([], [], marker='o', color='w', mec='k', ms=5, ls='', label='electrode (pair, polarity)')],
             loc='upper center', bbox_to_anchor=(0.5, -0.005), ncol=2, fontsize=9, frameon=True, facecolor='0.9')

    # zoom on the emitter (the true one, always fish c['emitter'])
    z = fig.add_subplot(gs[:, 1])
    yx = np.argwhere(fm[em]); (y0, x0), (y1, x1) = yx.min(0), yx.max(0)
    hz = min(max(y1 - y0, x1 - x0) / 2 + 45, (P - 1) / 2)
    cy, cx = np.clip([(y0 + y1) / 2, (x0 + x1) / 2], hz, P - 1 - hz)               # square crop kept inside the tank
    z.imshow(img, cmap='gray')
    ov = np.zeros((P, P, 4)); ov[fm[em]] = to_rgba(c['fish_color'][em], 0.35); z.imshow(ov)
    z.contour(fm[em].astype(float), levels=[0.5], colors=[c['fish_color'][em]], linewidths=2)
    X = letter[em]; L_mine = L_sleap = np.nan; agree = None
    if mids[em] is not None:
        mid = mids[em] * (P - 1)
        z.plot(mid[:, 1], mid[:, 0], color=C_MINE, lw=2.5, path_effects=OUTLINE)
        z.plot(mid[-1, 1], mid[-1, 0], 'o', color=C_MINE, mec='w', ms=12)
        L_mine = np.linalg.norm(np.diff(mid, axis=0), axis=1).sum() / PX_PER_CM
        mouth = np.asarray(dp[f'mouth_{X}'])[::-1] * PX_PER_CM                      # (row, col) px
        agree = np.linalg.norm(mid[-1] - mouth) < np.linalg.norm(mid[0] - mouth)
    dm = np.asarray(dp[f'currents_locations_{X}']) * PX_PER_CM
    z.plot(dm[:, 0], dm[:, 1], color=C_DENIS, lw=2.5, ls='--', path_effects=OUTLINE)
    z.plot(dm[-1, 0], dm[-1, 1], 's', color=C_DENIS, mec='w', ms=10)
    kp_px = {kp: np.asarray(dp[f'{kp}_{X}']) * PX_PER_CM for kp in ('mouth', 'head', 'middle', 'tail')}
    for kp, mk in zip(kp_px, ('^', 'o', 's', 'v')):
        z.plot(*kp_px[kp], mk, color='w', mec='k', ms=10, mew=1.2)
    L_sleap = sum(np.linalg.norm(kp_px[b] - kp_px[a_]) for a_, b in (('mouth', 'head'), ('head', 'middle'), ('middle', 'tail'))) / PX_PER_CM
    for pid, pp, pm in c['pairs']:                                           # electrodes inside the zoom
        for pt, sgn in ((pp, '+'), (pm, '−')):
            y, x = pt * (P - 1)
            if abs(y - cy) < hz and abs(x - cx) < hz:
                z.plot(x, y, 'o', color=C_PAIR if pid == pair else 'w', mec='k', ms=9)
                z.annotate(f'{pid}{sgn}', (x, y), xytext=(6, 6), textcoords='offset points', fontsize=10, fontweight='bold',
                           color=C_PAIR if pid == pair else 'k', bbox=dict(boxstyle='round,pad=0.1', fc='w', ec='none', alpha=0.85))
    z.set(xlim=(cx - hz, cx + hz), ylim=(cy + hz, cy - hz)); z.set_xticks([]); z.set_yticks([])
    for s_ in z.spines.values(): s_.set_color(c['fish_color'][em]); s_.set_linewidth(2.5)
    ht = '?' if agree is None else ('same as SLEAP' if agree else 'OPPOSITE to SLEAP')
    z.set_title(f"fish {em} (emitter), zoomed\nlength: my midline {L_mine:.1f} cm, SLEAP {L_sleap:.1f} cm\n"
                f"(usual: {c['len_med'][0]:.1f} and {c['len_med'][1]:.1f} cm)\nmy head–tail direction: {ht}",
                fontsize=10, color='k' if agree in (None, True) else C_PAIR)

    # voltages: one row per candidate fish
    for row, f in enumerate(fish):
        X = letter[f]
        pd_hat = sigma * np.asarray(dp[f'normalized_predicted_PTP_{X}']); pd_hat = pd_hat / np.abs(pd_hat).max()
        la = fig.add_subplot(gs[row, 2]); la.set_axis_off()
        lines = [(f"fish {f} as emitter" + (' (true)' if f == em else ''), 'k', 'bold'),
                 (f"mine: d = {d_mine[f]:.2f}" + ('  ◀ pick' if f == mine_pick else ''), C_MINE, 'bold' if f == mine_pick else 'normal'),
                 (f"Denis: sim = {dp[f'similarity_{X}']:+.2f}" + ('  ◀ pick' if f == den_pick else ''), C_DENIS,
                  'bold' if f == den_pick else 'normal')]
        for j, (s, col, fw) in enumerate(lines):
            la.text(0.8, 0.68 - 0.2 * j, s, color=col, fontweight=fw, fontsize=10, ha='right', va='center', transform=la.transAxes)
        for kk, pid in enumerate(c['pair_ids']):
            ax = fig.add_subplot(gs[row, 3 + kk])
            ax.axhline(0, color='0.85', lw=0.6)
            ax.plot(t_ms, V[kk] / rmax, color=C_REC, lw=2.2)
            ax.plot(t_ms, U[f, kk] * w, color=C_MINE, lw=1.8)
            ax.plot(t_ms, pd_hat[kk] * w, color=C_DENIS, lw=1.8, ls='--')
            ax.set_ylim(-1.25, 1.25); ax.set_xticks([-0.2, 0, 0.2]); ax.tick_params(labelsize=8)
            if row == 0: ax.set_title(f'pair {pid}', fontsize=10, color=C_PAIR if pid == pair else 'k',
                                      fontweight='bold' if pid == pair else 'normal')
            if kk: ax.set_yticklabels([])
            if row < nf - 1: ax.set_xticklabels([])
            else: ax.set_xlabel('time (ms)', fontsize=9)
            if f in (mine_pick, den_pick):                                   # frame the picked rows
                for sp in ax.spines.values(): sp.set_color('0.3'); sp.set_linewidth(1.2)

    ok = lambda p: '✓' if p == em else '✗'
    fig.suptitle(f"{head} | frame {k} (EOD row {i})\n"
                 f"my pick: fish {mine_pick} {ok(mine_pick)}     Denis' pick: fish {den_pick} {ok(den_pick)}",
                 fontsize=13, y=0.99)
    fig.legend(handles=[Line2D([], [], color=C_REC, lw=2.2, label='recorded (all traces scaled to max |v| = 1)'),
                        Line2D([], [], color=C_MINE, lw=1.8, label='my model'),
                        Line2D([], [], color=C_DENIS, lw=1.8, ls='--', label="Denis' model")],
               loc='lower center', bbox_to_anchor=(0.66, -0.02), ncol=3, fontsize=11, frameon=False)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.show()

def use_better_list(c, path='results/eod_ground_truth_3fish/good_frames_better_list.csv'):
    """Replace c['assigned']['good frames'] by the frames of `path` (good_frames_better_list.csv)."""
    c['assigned']['good frames'] = c['assigned'].Frame.isin(pd.read_csv(path).frame).to_numpy()
    return c

def body_lengths(c, i):
    """(my midline, SLEAP polyline) length of the emitter in EOD row i, cm."""
    m = fish_midlines(frame_masks(c, int(c['df'].Frame[i])))[c['emitter']]; X = {f: x for x, f in c['denis_map'].items()}[c['emitter']]
    k = {kp: np.asarray(c['den'].loc[i, f'{kp}_{X}']) for kp in ('mouth', 'head', 'middle', 'tail')}
    sl = sum(np.linalg.norm(k[b] - k[a]) for a, b in (('mouth', 'head'), ('head', 'middle'), ('middle', 'tail')))   # cm
    return (np.nan if m is None else np.linalg.norm(np.diff(m * (c['P'] - 1), axis=0), axis=1).sum() / PX_PER_CM), sl

def emitter_usual_lengths(c, n=300, seed=0):
    """c['len_med']: median (my midline, SLEAP) length of the emitter over n random good-frame EODs (cm)."""
    smp = np.random.default_rng(seed).choice(np.flatnonzero(c['assigned']['good frames'].to_numpy()), n, replace=False)
    c['len_med'] = np.nanmedian([body_lengths(c, c['assigned'].index[j]) for j in smp], axis=0)
    return c
