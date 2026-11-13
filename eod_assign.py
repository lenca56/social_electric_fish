"""Assigning recorded EODs to fish, and comparing model and recorded electrode-pair voltages.

Conventions (checked against the data in eod_electrode_comparison.ipynb):
  - Masks, midlines and electrodes share the (row, col) pixel axes of the (P, P) video frame; tank coordinates are
    pixel / (P - 1). An electrode or keypoint pixel (x, y) is (col, row).
  - df_eod.Frame is the 0-indexed video frame, i.e. the EOD in a row goes with masks[:, Frame] (the fish keypoints
    stored in the same row sit on those masks, and are 1 frame off from masks[:, Frame - 1]).
  - Model: resistive (omega = 0) field of one candidate emitter with unit current along its midline; the other fish are
    skin + interior bodies, the emitter's own body is transparent.
  - Comparison: the 10 pair voltages of the model and of the recording are each scaled to max |v| = 1, and compared by
    the L2 norm of their difference. The model enters with a constant sign, MODEL_SIGN = -1: the recorded peak sample
    has the opposite sign to the unit-current model (sink at the tail, sources toward the head) for ~97-99% of the
    true emitter's EODs (3-fish, 2-fish and 4-fish 2024-02-20 recordings, 2026-10-28).

  History (no longer used): the model vector used to be compared with both signs and the smaller distance kept, since the
    recorded sample can sit in either phase of the EOD (sign of I(t) unknown). That let a wrong fish win by fitting upside
    down (58% of the 3-fish errors); fixing the sign raised 3-fish good-frame accuracy from 89.5 to 94.5%.
"""
import re
import gzip
import pickle
import hashlib
import numpy as np
import pandas as pd
from midline_methods import fish_midline_with_head, oriented_midline, contact_frames, heads_with_continuity
from compute_electric_field import (multipole_sources, occupancy_field, build_A_shell, solve_potential,
                                    unflatten_to_grid, gammas, build_multipole_b, sample_phi_point, SINK_FRAC)
from io_utils import load_electrodes as load_electrode_pairs   # the reader lives in io_utils; old name kept for its users


# electrodes and recordings

def model_cache_tag(N, *electrode_csvs):
    """Tag for files of cached model voltages: grid size, sink position (compute_electric_field.SINK_FRAC) and the
       contents of the electrode file(s), so voltages of another source layout or electrode polarity are never reused."""
    h = hashlib.md5(b''.join(open(p, 'rb').read() for p in electrode_csvs)).hexdigest()[:6]
    return f'N{N}_sink{SINK_FRAC:g}_el{h}'

def eod_sample_count(df):
    """Number of samples per EOD in the electrode_<pair>_time_<i> columns."""
    return max(int(m.group(1)) for c in df.columns for m in [re.match(r'electrode_\d+_time_(\d+)', c)] if m)

def mask_frame(row):
    """Index into the mask frames for one df_eod row."""
    return int(row['Frame'])

def eods_with_masks(df, n_frames, skip_frames=(1,)):
    """Rows of df whose EOD falls in mask frames 0..n_frames-1. Frame 1 holds a pile-up of 25 events at the start
       of the recording, so it is skipped by default."""
    return df[(df['Frame'] >= 0) & (df['Frame'] < n_frames) & ~df['Frame'].isin(skip_frames)]

def recorded_pair_traces(row, pair_ids, n_samp):
    """(n_pairs, n_samp) recorded voltages of one df_eod row, each pair demeaned."""
    V = np.stack([row[[f'electrode_{pid}_time_{i}' for i in range(1, n_samp + 1)]].to_numpy(float)
                  for pid in pair_ids])
    return V - np.nanmean(V, axis=1, keepdims=True)

def recorded_pair_traces_all(df, pair_ids, n_samp):
    """(n_rows, n_pairs, n_samp) recorded voltages of every df_eod row (same as recorded_pair_traces, vectorized)."""
    cols = [f'electrode_{pid}_time_{i}' for pid in pair_ids for i in range(1, n_samp + 1)]
    V = df[cols].to_numpy(float).reshape(len(df), len(pair_ids), n_samp)
    return V - np.nanmean(V, axis=2, keepdims=True)

def _unpack_mask(shape, packed):
    n = int(np.prod(shape))
    buf = np.frombuffer(packed, dtype=np.uint8) if isinstance(packed, (bytes, bytearray)) else packed
    return np.unpackbits(buf, count=n).reshape(shape).astype(bool)

def iter_mgz_frames(mgz_path, frames, n_fish=None):
    """Yield (frame, (n_fish, P, P) bool masks) for the requested 0-indexed frames of a packed .mgz mask file, in file
       order, decoding only those frames (one pass, stops after the last one). Mask f is object id f (the order of the
       cached first-1000-frames .npy); a fish missing from a frame gets an empty mask, so indices never shift.
       n_fish: number of fish; None = 1 + the largest object id in the file's first frame."""
    wanted = set(int(f) for f in frames)
    if not wanted: return
    last = max(wanted)
    with gzip.open(mgz_path, 'rb') as fh:
        pickle.load(fh)                                         # header record
        while wanted:
            try:
                frame, objs = pickle.load(fh)
            except EOFError:
                break
            if n_fish is None: n_fish = max(objs) + 1
            if frame > last: break
            if frame in wanted:
                wanted.discard(frame)
                shape = next(iter(objs.values()))[0]
                masks = np.zeros((n_fish,) + tuple(shape), bool)
                for o, packed in objs.items(): masks[o] = _unpack_mask(*packed)
                yield frame, masks

def keypoint_head_agrees(row, mask, midline):
    """Whether a midline's head end matches the tracked head/tail keypoints stored in a df_eod row (columns
       head_x_<k>, head_y_<k>, middle_..., tail_...), for the tracked fish k whose middle keypoint lies on `mask`.
       Returns True / False, or None when no tracked fish's middle keypoint lies on the mask."""
    P = mask.shape[0]
    ks = sorted(int(m.group(1)) for c in row.index for m in [re.match(r'middle_x_(\d+)$', c)] if m)
    for k in ks:
        x, y = row[f'middle_x_{k}'], row[f'middle_y_{k}']
        if np.isnan(x) or not mask[min(int(round(y)), P - 1), min(int(round(x)), P - 1)]: continue
        hd = np.array([row[f'head_y_{k}'], row[f'head_x_{k}']]) / (P - 1)
        tl = np.array([row[f'tail_y_{k}'], row[f'tail_x_{k}']]) / (P - 1)
        if np.isnan(hd).any() or np.isnan(tl).any(): continue
        H, T = midline[-1], midline[0]
        return bool(np.linalg.norm(H - hd) + np.linalg.norm(T - tl) < np.linalg.norm(H - tl) + np.linalg.norm(T - hd))
    return None

def peak_pattern(V):
    """Recorded pair voltages at the one sample where |V| peaks over all pairs. Returns (v (n_pairs,), t_star)."""
    t_star = int(np.unravel_index(np.nanargmax(np.abs(V)), V.shape)[1])
    return V[:, t_star], t_star

def rank1_pattern(V):
    """Best rank-1 fit V ~ outer(u, w) of a recorded window (SVD), with ||w|| = 1. In the resistive case every pair sees
       the same waveform, so u is the spatial pattern and w the EOD waveform; their joint sign is arbitrary.
       Returns (u (n_pairs,), w (n_samp,), fraction of the window's variance in this pattern)."""
    U, s, Wt = np.linalg.svd(np.nan_to_num(V), full_matrices=False)
    return U[:, 0] * s[0], Wt[0], s[0] ** 2 / np.sum(s ** 2)

def eod_phase_samples(V):
    """Samples of the two main phases of a recorded window: (first phase P1, second phase P2), the extrema of opposite
       sign of its rank-1 waveform, in time order. Signs are not touched: V[:, P1] and V[:, P2] are raw recordings."""
    _, w, _ = rank1_pattern(V)
    ta = int(np.argmax(np.abs(w)))
    tb = int(np.argmax(-np.sign(w[ta]) * w))
    return min(ta, tb), max(ta, tb)

def eod_waveform_template(waveforms, n_iter=3):
    """Common EOD waveform (unit norm) of time-aligned rank-1 waveforms (n_events, n_samp), each sign-flipped to agree."""
    W = np.asarray(waveforms, float)
    T = W[0] / np.linalg.norm(W[0])
    for _ in range(n_iter):
        T = (W * np.sign(W @ T)[:, None]).mean(0); T /= np.linalg.norm(T)
    return T

def oriented_pattern(V, template):
    """Rank-1 spatial pattern of a recorded window, signed so that its waveform matches `template`. The sign of each
       entry then means the same EOD phase in every event. Returns (u, correlation of the waveform with the template)."""
    u, w, _ = rank1_pattern(V)
    c = float(w @ template)
    return u * np.sign(c), abs(c)


# fish geometry

def midlines_with_continuity(masks, n_ctrl, min_pixels=50):
    """Midline of every fish in every frame, tail -> head, with the head side chained through mask contacts.
       masks: (n_fish, n_frames, P, P) bool (an np.load mmap is fine); frames must be consecutive.
       A fish with fewer than min_pixels mask pixels in a frame counts as not visible there (midline None).
       Returns (midlines[f][fr]: (n_ctrl, 2) in [0, 1] or None, rule[f][fr]: 'area' / 'continuity' / ... / None)."""
    n_fish, n_frames = masks.shape[0], masks.shape[1]
    get_masks = lambda fr: [np.asarray(masks[f, fr]) for f in range(n_fish)]
    tips = [[None] * n_frames for _ in range(n_fish)]
    for fr in range(n_frames):
        for f, m in enumerate(get_masks(fr)):
            if m.sum() >= min_pixels:
                tips[f][fr] = fish_midline_with_head(m, n_ctrl=n_ctrl)
    contact, _ = contact_frames(get_masks, range(n_frames))
    midlines, rules = [], []
    for f in range(n_fish):
        head_is_a, rule = heads_with_continuity(tips[f], contact[:, f])
        midlines.append([None if t is None else oriented_midline(t, a) for t, a in zip(tips[f], head_is_a)])
        rules.append(rule)
    return midlines, rules


# forward model at the electrodes

def candidate_field(frame_masks, f, midline, N, delta_h, mat, omega=0.0, bodies=True):
    """(N,N) complex potential of fish f emitting unit current along `midline` (multipole_sources: sink 5% of the body
       length from the tail tip, equal sources from there to the head). bodies=True: the other fish of frame_masks (n_fish, P, P) are skin + interior bodies;
       bodies=False: plain water. mat: sigma_/eps_ water, skin, interior (see compute_electric_field.gammas)."""
    h = 1.0 / (N - 1)
    pos, cur = multipole_sources(midline, I0=1.0)
    b = build_multipole_b(pos, cur, h, N)
    occ = occupancy_field(frame_masks, N, skip=f) if bodies else np.zeros((N, N))
    gw, gs, gi = gammas(omega, **mat)
    return unflatten_to_grid(solve_potential(build_A_shell(occ, N, delta_h, gw, gs, gi), b), N)

def pair_voltages(phi, pairs, N):
    """Electrode-pair voltages V(+) - V(-) of an (N,N) potential, one per pair."""
    h = 1.0 / (N - 1)
    return np.array([sample_phi_point(phi, pp, h, N) - sample_phi_point(phi, pm, h, N) for _, pp, pm in pairs])

def free_space_pair_voltages(midline, pairs):
    """Pair voltages V(+) - V(-) of a fish's multipole sources (sink near the tail, head-side sources, unit current) in unbounded
       uniform water, phi(x) = -sum_i I_i ln|x - x_i| / (2 pi): no walls, no other fish. Its sign depends only on where
       the fish's head and tail are relative to the two electrodes."""
    pos, cur = multipole_sources(midline, I0=1.0)
    phi = lambda p: -np.sum(cur * np.log(np.linalg.norm(pos - p, axis=1))) / (2 * np.pi)
    return np.array([phi(pp) - phi(pm) for _, pp, pm in pairs])

def fish_distances(frame_masks, points):
    """Distance (tank units) from the nearest of `points` (k, 2) in [0, 1] to each fish's mask; inf for an empty mask."""
    P = frame_masks.shape[-1]
    out = []
    for m in frame_masks:
        pix = np.argwhere(m) / (P - 1)
        out.append(np.inf if len(pix) == 0 else
                   min(np.min(np.linalg.norm(pix - p, axis=1)) for p in np.atleast_2d(points)))
    return np.array(out)

def predict_pair_traces(I_t, frame_masks, f, midline, pairs, N, delta_h, mat, fs, bodies=True):
    """(n_pairs, n_t) model pair voltages over a recorded window for emitted current waveform I_t (demeaned), solved at
       every rfft frequency with gamma = sigma + i*omega*eps, so a capacitive skin can change the waveform shape."""
    n_t = len(I_t)
    om = 2 * np.pi * np.fft.rfftfreq(n_t, 1 / fs)
    Psi = np.zeros((len(om), len(pairs)), complex)                  # DC left at 0: I_t is demeaned
    for k in range(1, len(om)):
        Psi[k] = pair_voltages(candidate_field(frame_masks, f, midline, N, delta_h, mat, om[k], bodies), pairs, N)
    return np.fft.irfft(np.fft.rfft(I_t)[:, None] * Psi, n=n_t, axis=0).T


# comparison and assignment

def normalize_peak(v):
    """v scaled so that max |v| = 1."""
    v = np.asarray(v, float)
    return v / np.max(np.abs(v))

MODEL_SIGN = -1                    # constant sign of the unit-current model against the recorded peak sample (see top)

def compare_patterns(v_sim, v_rec):
    """L2 distance between the peak-normalized model (times MODEL_SIGN) and recorded pair-voltage vectors.
       Returns (distance, model vector normalized and multiplied by MODEL_SIGN, recording normalized)."""
    a, b = MODEL_SIGN * normalize_peak(v_sim), normalize_peak(v_rec)
    # History (no longer used): the model's sign was chosen per EOD to fit the recording best:
    # a = normalize_peak(v_sim)
    # if np.linalg.norm(a + b) < np.linalg.norm(a - b): a = -a
    return np.linalg.norm(a - b), a, b

def cosine_similarity(v_sim, v_rec):
    """Cosine similarity of the model (times MODEL_SIGN) and the recorded pair-voltage vectors (scale-free; Denis' rule)."""
    a, b = MODEL_SIGN * np.asarray(v_sim, float), np.asarray(v_rec, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def polarity_candidates(patterns, k, min_share=0.5):
    """Row labels of the EODs where pair k carries at least min_share of the largest |pattern| entry, most dominant
       first. patterns: {df row label: template-oriented rank-1 pattern (see oriented_pattern)}."""
    share = {r: abs(u[k]) / np.max(np.abs(u)) for r, u in patterns.items()}
    return [r for r in sorted(share, key=share.get, reverse=True) if share[r] >= min_share]

def polarity_event(row, u, frame_masks, pairs, k, midlines=None, n_ctrl=101, min_ratio=2.0, contact_gap=3):
    """Polarity test of pair k on one recorded EOD, without fitting the model to the recordings.
       The fish nearest the pair's two electrodes is taken as the emitter if it is at least min_ratio times closer than
       any other fish. Recorded sign: sign of the pair's entry in u (template-oriented rank-1 pattern). Predicted sign:
       what a head-positive current along the emitter's midline gives at the pair in free space. Correctly wired pairs
       all share one sign of recorded x predicted; a reversed pair has the other.
       midlines: per-fish midlines of this frame (e.g. from midlines_with_continuity); None -> the emitter's area-rule
       midline, and the event is skipped if the emitter touches another fish (the area rule is unreliable in contact).
       Returns a dict, or None if the event is skipped. head_keypoints: does the midline head agree with the tracked head
       keypoint in this row (True / False / None = no match)."""
    _, pp, pm = pairs[k]
    d = fish_distances(frame_masks, np.array([pp, pm]))
    o = np.argsort(d); f = int(o[0])
    if not np.isfinite(d[f]) or d[o[1]] < min_ratio * d[f]: return None
    if midlines is None:
        if contact_frames(lambda _: list(frame_masks), [0], gap=contact_gap)[0][0, f]: return None
        mid = fish_midline_with_head(frame_masks[f], n_ctrl=n_ctrl)['midline']
    else:
        mid = midlines[f]
        if mid is None: return None
    rec, pred = int(np.sign(u[k])), int(np.sign(free_space_pair_voltages(mid, [pairs[k]])[0]))
    return dict(pair=pairs[k][0], row=row.name, frame=mask_frame(row), fish=f, dist=d[f], dist_next=d[o[1]],
                share=abs(u[k]) / np.max(np.abs(u)), amplitude=abs(u[k]), rec_sign=rec, pred_sign=pred,
                product=rec * pred, head_keypoints=keypoint_head_agrees(row, frame_masks[f], mid))

def polarity_events(df, patterns, masks, midlines, pairs, k, n_events=10, min_share=0.5, min_gap=20, min_ratio=2.0,
                    N=None, delta_h=0.5, mat=None):
    """polarity_event for pair k on the most dominant EODs (polarity_candidates) of an in-memory mask array
       (n_fish, n_frames, P, P) with midlines[f][frame], keeping up to n_events at least min_gap frames apart.
       With N and mat given, the full solver's sign (walls + other fish) is added as a check (solver_product).
       Returns a DataFrame, one row per event."""
    out, frames = [], []
    for r in polarity_candidates(patterns, k, min_share):
        if len(out) == n_events: break
        row = df.loc[r]; fr = mask_frame(row)
        if any(abs(fr - c) < min_gap for c in frames): continue
        fm = np.asarray(masks[:, fr])
        e = polarity_event(row, patterns[r], fm, pairs, k, [m[fr] for m in midlines], min_ratio=min_ratio)
        if e is None: continue
        frames.append(fr)
        if N is not None:
            v = pair_voltages(candidate_field(fm, e['fish'], midlines[e['fish']][fr], N, delta_h, mat).real,
                              [pairs[k]], N)[0]
            e['solver_product'] = e['rec_sign'] * int(np.sign(v))
        out.append(e)
    return pd.DataFrame(out)

def polarity_events_mgz(df, patterns, mgz_path, pairs, min_share=0.5, min_gap=20, min_ratio=2.0,
                        max_candidates=400):
    """polarity_event for every pair over a whole recording, streaming only the needed mask frames from the .mgz in one
       pass. Per pair, the max_candidates most dominant EODs at least min_gap frames apart are tested, with area-rule
       heads (emitters touching another fish are skipped). Returns a DataFrame of every event that passes, all pairs."""
    need = {}
    for k in range(len(pairs)):
        frames = []
        for r in polarity_candidates(patterns, k, min_share):
            if len(frames) == max_candidates: break
            fr = mask_frame(df.loc[r])
            if any(abs(fr - c) < min_gap for c in frames): continue
            frames.append(fr); need.setdefault(fr, []).append((r, k))
    out = []
    for fr, fm in iter_mgz_frames(mgz_path, need):
        for r, k in need[fr]:
            e = polarity_event(df.loc[r], patterns[r], fm, pairs, k, min_ratio=min_ratio)
            if e is not None: out.append(e)
    return pd.DataFrame(out)

def assign_eod(row, frame_masks, frame_midlines, pairs, N, delta_h, mat, n_samp, bodies=True):
    """Attribute one recorded EOD (a df_eod row) to the fish whose resistive model pattern is nearest.
       frame_masks: (n_fish, P, P) masks of the row's frame; frame_midlines: per fish, its midline there or None.
       Returns dict: fish (winner), distance, distances {fish: distance}, v_sim {fish: normalized model vector x MODEL_SIGN},
       v_rec (normalized recording), t_star; fish is None if no fish is visible."""
    v_rec, t_star = peak_pattern(recorded_pair_traces(row, [pid for pid, _, _ in pairs], n_samp))
    distances, v_sim = {}, {}
    for f, mid in enumerate(frame_midlines):
        if mid is None: continue
        v = pair_voltages(candidate_field(frame_masks, f, mid, N, delta_h, mat, 0.0, bodies).real, pairs, N)
        distances[f], v_sim[f], v_rec_n = compare_patterns(v, v_rec)
    if not distances:
        return dict(fish=None, distance=np.nan, distances={}, v_sim={}, v_rec=normalize_peak(v_rec), t_star=t_star)
    fish = min(distances, key=distances.get)
    return dict(fish=fish, distance=distances[fish], distances=distances, v_sim=v_sim, v_rec=v_rec_n, t_star=t_star)
