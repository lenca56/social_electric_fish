"""Identity decoding from motion continuity + body size, for Shiv's and Kaden's 4-fish tracks.

Each frame the true identity of the four labels is one of the 24 permutations. The decoder
(Viterbi) finds the most likely permutation over time given

  * where a switch is possible at all (motion continuity): identities can only change between
    two labels at a frame where they are in contact (centroids closer than half the sum of their
    typical lengths), or where one of them is lost / reappears / teleports. Everywhere else the
    identity is carried forward unchanged.
  * how likely a switch is there: a contact frame alone gives a small prior (P_CONTACT); if the
    two labels exchange positions from one frame to the next (each label is much closer to the
    other label's previous position than to its own: an abrupt swap, impossible for real fish)
    a swap is near-certain (P_EXCHANGE, staying gets the rest); a label that reappears after a
    gap or teleports may swap with a label near its old or new position (P_LOST).
  * which permutation fits the fish (size): the four fish differ in size, so each label's mask
    area is compared with every fish's typical area. Area is corrected for position first (fish
    look smaller along the walls), using a wall gain estimated from the video itself.

The typical areas and the wall gain are re-estimated from the decoded identities and the
decoding is repeated (N_ITER times). Each decoded episode is then tested against sizes measured
only in open water (classify_episodes, size_frames).

Output per frame: the decoded permutation; `wrong` = at least one label sits on another fish.
"""

import glob
import gzip
import itertools
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import mask_metrics as mm
import tracking_compare as tc

CACHE = os.path.join("figures", "identity_continuity", "cache")

P_CONTACT = 1e-6        # switch prior per contact frame
P_EXCHANGE = 0.95       # labels exchange positions frame-to-frame (physically impossible for real fish)
P_LOST = 0.1            # a label reappears after a gap or teleports
EXCHANGE_RATIO = 0.5    # crossed displacement < this x straight displacement ...
EXCHANGE_MIN = 0.25     # ... and straight displacement > this x mean body length
OBS_WEIGHT = 0.02       # weight of one frame's size evidence (frames are strongly correlated)
OUTLIER = 0.10          # share of size observations treated as outliers
SD_MIN = 0.05           # floor on the per-fish log-area spread
N_ITER = 2

GAIN_CLIP = (0.75, 1.25)
STACK_MIN_S = 5.0       # episodes this long with the two fish on top of each other are 'stacked'
STACK_FRAC = 0.25       # centroids closer than this x the two lengths summed = on top of each other
WALL_MARGIN = 120       # px from the image edge: closer, area is not used for the episode size test
IMAGE = 912


# --------------------------------------------------------------------------- inputs

def _shiv_table(path):
    """Per (frame, label 0-3): area, centroid, major axis; straight from the packed masks."""
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
    rows = []
    for k in sorted(d["frames"]):
        fr = d["frames"][k]
        for lab, box, packed, area in zip(fr["labels"], fr["boxes"], fr["masks"], fr["areas"]):
            if lab >= 4:
                continue
            crop = mm.unpack_crop(packed, box)
            ys, xs = np.nonzero(crop)
            if ys.size == 0:
                continue
            cy, cx = ys.mean(), xs.mean()
            dx, dy = xs - cx, ys - cy
            cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
            major = 4 * np.sqrt(max(np.linalg.eigvalsh(cov)[1], 0))
            rows.append((k, lab, area, cx + box[2], cy + box[0], major))
    df = pd.DataFrame(rows, columns=["frame", "label", "area", "cx", "cy", "major"])
    meta = dict(fps=d["fps"], n_frames=len(d["frames"]))
    return df, meta


def load_table(tracker, rec, path):
    """(df, meta) with columns frame, label, area, cx, cy, major (+ occluded for Kaden); cached."""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"{tracker}_{rec}.pkl" if tracker == "Shiv" else f"{tracker}_amodal_{rec}.pkl")
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)
    if tracker == "Shiv":
        out = _shiv_table(path)
    else:
        # Kaden: full-body (amodal) centroid and area from his output pickle. During occlusion
        # the visible mask breaks into pieces whose centroids hop around and can look like two
        # labels exchanging positions; the amodal mask is his estimate of the whole fish.
        with open(os.path.join(mm.OUT_DIR, "kaden", f"video{rec}", "metrics.pkl"), "rb") as f:
            df, _, meta = pickle.load(f)
        with open(path, "rb") as f:
            d = pickle.load(f)
        df = df[["frame", "label", "major", "occluded", "inferred"]].copy()
        k, lab = df.frame.to_numpy(), df.label.to_numpy()
        df["area"] = d["area_amodal"][k, lab].astype(float)
        df["cx"] = d["centroid_amodal"][k, lab, 0].astype(float)
        df["cy"] = d["centroid_amodal"][k, lab, 1].astype(float)
        out = (df, dict(fps=meta["fps"], n_frames=meta["n_frames"]))
    with open(cp, "wb") as f:
        pickle.dump(out, f)
    return out


def arrays(tracker, rec, path):
    """Frame x label arrays (NaN where untracked) and the usable-area mask."""
    df, meta = load_table(tracker, rec, path)
    fish = tc.FISH[tracker]
    n = meta["n_frames"]
    df = df[df.label.isin(fish)].sort_values("area").drop_duplicates(["frame", "label"], keep="last")
    out = {}
    for c in ["area", "cx", "cy", "major"]:
        out[c] = df.pivot(index="frame", columns="label", values=c).reindex(
            index=range(n), columns=fish).to_numpy(float)
    flags = tc.frame_flags(tc.load_metrics(tracker, rec, path)["metrics"])
    bad = np.stack([flags.xs(l, axis=1, level="fish")[tc.BAD_FLAGS].any(axis=1).values
                    for l in fish], 1)
    if tracker == "Kaden":   # amodal area is inferred, not measured, while occluded
        occ = df.pivot(index="frame", columns="label", values="occluded").reindex(
            index=range(n), columns=fish).astype(float).fillna(0).to_numpy(bool)
        bad |= occ
    out["usable"] = ~bad & np.isfinite(out["area"])
    out["fps"], out["n"], out["fish"] = meta["fps"], n, fish
    return out


# --------------------------------------------------------------------------- model pieces

PERMS = list(itertools.permutations(range(4)))          # perm[l] = true fish index of label l
PERM_IDX = {p: i for i, p in enumerate(PERMS)}
IDENTITY = PERM_IDX[(0, 1, 2, 3)]


def _swap_map(a, b):
    """state index -> state index after labels a and b exchange identities."""
    out = np.empty(len(PERMS), int)
    for i, p in enumerate(PERMS):
        q = list(p)
        q[a], q[b] = q[b], q[a]
        out[i] = PERM_IDX[tuple(q)]
    return out


SWAP = {(a, b): _swap_map(a, b) for a, b in itertools.combinations(range(4), 2)}


WALL_BINS = [0, 50, 100, 150]     # distance to the nearest image edge (px); beyond: no correction
SIDES = ["top", "bottom", "left", "right"]


def _wall_cell(x, y, size):
    d = np.stack([y, size - y, x, size - x])               # distance to top, bottom, left, right
    side = np.nanargmin(np.nan_to_num(d, nan=np.inf), axis=0)
    dist = np.nanmin(np.nan_to_num(d, nan=np.inf), axis=0)
    dbin = np.digitize(dist, WALL_BINS[1:])               # 0..len(WALL_BINS)-1; last = interior
    return side, dbin


def gain_map(x, y, rel, lab, size=IMAGE):
    """Apparent-size gain by nearest wall and distance to it, as a lookup function.

    Fish look smaller along the walls (and under the apparatus at the top). The gain of a
    (wall, distance) cell is the median relative area of every label there; a cell only gets
    a gain when at least two labels visited it (>= 50 usable frames each), so one fish's own
    size can't be absorbed into it. Gains are clipped to GAIN_CLIP; the interior is 1.
    """
    ok = np.isfinite(rel) & np.isfinite(x)
    side, dbin = _wall_cell(x[ok], y[ok], size)
    nd = len(WALL_BINS) - 1
    g = np.ones((len(SIDES), nd + 1))
    s = pd.Series(np.log(rel[ok]))
    med = s.groupby([side, dbin]).median()
    per_lab = pd.Series(1, index=s.index).groupby([side, dbin, lab[ok]]).size()
    n_labs = (per_lab >= 50).groupby(level=[0, 1]).sum()
    for (i, j), v in med.items():
        if j < nd and n_labs.get((i, j), 0) >= 2:
            g[i, j] = np.exp(np.clip(v, *np.log(GAIN_CLIP)))

    def f(xx, yy):
        sd_, db = _wall_cell(xx, yy, size)
        return g[sd_, db]
    f.grid = pd.DataFrame(g, index=SIDES,
                          columns=[f"{a}-{b}px" for a, b in zip(WALL_BINS, WALL_BINS[1:])] + ["interior"])
    return f


def size_frames(A, length, margin=None):
    """Frames whose area is a fair size measure: usable, off the walls, away from other fish.

    Used to test decoded episodes (classify_episodes). Along the walls fish look smaller in a
    fish-specific way the wall gain can't fully undo (a fish resting against the top wall can
    look like the next smaller fish for minutes), and in contact masks are shared or cut.
    Too strict for the decoder itself: without wall and contact frames most swaps have no size
    evidence left to end them.
    """
    margin = WALL_MARGIN if margin is None else margin
    x, y = A["cx"], A["cy"]
    wall = np.nan_to_num(np.minimum.reduce([x, IMAGE - x, y, IMAGE - y]), nan=0)
    ok = A["usable"] & (wall > margin)
    for a, b in itertools.combinations(range(4), 2):
        d = np.nan_to_num(np.hypot(x[:, a] - x[:, b], y[:, a] - y[:, b]), nan=np.inf)
        near = d < 0.5 * (length[a] + length[b])
        ok[near, a] = False
        ok[near, b] = False
    return ok


def size_loglik(la, usable, mu, sd):
    """LL[t, label, fish] of the log area under each fish's size model (0 where unusable)."""
    z = (la[:, :, None] - mu[None, None, :]) / sd[None, None, :]
    dens = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sd[None, None, :])
    ll = np.log((1 - OUTLIER) * dens + OUTLIER / 3.0)       # outliers uniform over 3 log units
    ll[~usable] = 0.0
    return OBS_WEIGHT * ll


def opportunities(A, length):
    """Per frame t (transition t-1 -> t): list of (label a, label b, log prior)."""
    cx, cy, n = A["cx"], A["cy"], A["n"]
    pres = np.isfinite(cx)
    L = np.nanmean(length)
    ops = {}

    def add(t, a, b, lp):
        a, b = min(a, b), max(a, b)
        d = ops.setdefault(t, {})
        d[(a, b)] = max(d.get((a, b), -np.inf), lp)

    explained = {}          # frame -> labels whose jump is explained by a position exchange
    # last known position for gap handling
    lastx, lasty = pd.DataFrame(cx).ffill().to_numpy(), pd.DataFrame(cy).ffill().to_numpy()
    for a, b in itertools.combinations(range(4), 2):
        dist = np.hypot(lastx[:, a] - lastx[:, b], lasty[:, a] - lasty[:, b])
        contact = dist < 0.5 * (length[a] + length[b])
        contact[1:] |= contact[:-1]
        # exchange of positions between t-1 and t
        s = (np.hypot(cx[1:, a] - cx[:-1, a], cy[1:, a] - cy[:-1, a]) +
             np.hypot(cx[1:, b] - cx[:-1, b], cy[1:, b] - cy[:-1, b]))
        x = (np.hypot(cx[1:, a] - cx[:-1, b], cy[1:, a] - cy[:-1, b]) +
             np.hypot(cx[1:, b] - cx[:-1, a], cy[1:, b] - cy[:-1, a]))
        exch = np.zeros(n, bool)
        exch[1:] = (x < EXCHANGE_RATIO * s) & (s > EXCHANGE_MIN * L)
        for t in np.flatnonzero(exch):
            add(t, a, b, np.log(P_EXCHANGE))
            explained.setdefault(t, set()).update((a, b))
        for t in np.flatnonzero(contact & ~exch):
            if t > 0:
                add(t, a, b, np.log(P_CONTACT))
    # reappearance after a gap / teleport: may swap with any other label
    speed = np.zeros_like(cx)
    speed[1:] = np.hypot(np.diff(cx, axis=0), np.diff(cy, axis=0)) * A["fps"]
    for a in range(4):
        reappear = np.zeros(n, bool)
        reappear[1:] = pres[1:, a] & ~pres[:-1, a]
        jump = speed[:, a] > tc.SPEED_JUMP
        for t in np.flatnonzero(reappear | jump):
            if a in explained.get(t, ()):
                continue
            # only with a label near where a was last seen or where it is now
            for b in range(4):
                if b == a or b in explained.get(t, ()):
                    continue
                near = 0.75 * (length[a] + length[b])
                d_new = np.hypot(cx[t, a] - lastx[t, b], cy[t, a] - lasty[t, b])
                d_old = np.hypot(lastx[t - 1, a] - lastx[t, b], lasty[t - 1, a] - lasty[t, b])
                if d_new < near or d_old < near:
                    add(t, a, b, np.log(P_LOST))
    return ops


def viterbi(LL, ops, n):
    """Most likely state per frame. LL: (n, 4, 4) label x fish log-likelihoods."""
    P = np.array(PERMS)                                     # (24, 4)
    E = LL[:, np.arange(4)[None, :], P].sum(-1)             # (n, 24)
    delta = np.full(len(PERMS), -50.0)
    delta[IDENTITY] = 0.0
    delta = delta + E[0]
    back = {}
    op_frames = sorted(ops)
    t = 1
    cum = np.cumsum(E, axis=0)
    for f in op_frames + [n]:
        if f > t:                                           # no switch possible in [t, f)
            delta = delta + cum[f - 1] - cum[t - 1]
            t = f
        if f == n:
            break
        # an exchange of positions is a label swap: not following it costs as much as an
        # unprompted switch at a contact (otherwise skipping it would always be cheaper than
        # entering the swap and leaving it again at a later contact)
        if any(lp == np.log(P_EXCHANGE) for lp in ops[f].values()):
            stay = np.log(P_CONTACT)
        else:
            stay = np.log(max(1 - sum(np.exp(lp) for lp in ops[f].values()), 0.05))
        cand = [delta + stay]
        for (a, b), lp in ops[f].items():
            m = SWAP[(a, b)]
            moved = np.full(len(PERMS), -np.inf)
            moved[m] = delta + lp                           # state s -> m[s]
            cand.append(moved)
        cand = np.stack(cand)                               # (k, 24)
        src = np.argmax(cand, 0)
        # remember the previous state of each new state
        prev = np.arange(len(PERMS))
        for k, (a, b) in enumerate(ops[f].keys(), start=1):
            inv = np.argsort(SWAP[(a, b)])                  # new state -> old state
            prev = np.where(src == k, inv, prev)
        back[f] = prev
        delta = cand.max(0) + E[f]
        t = f + 1
    state = np.empty(n, int)
    s = int(np.argmax(delta))
    for f in range(n - 1, -1, -1):
        state[f] = s
        if f in back:
            s = int(back[f][s])
    return state


def decode(tracker, rec, path, A=None, return_all=False):
    """Decoded state per frame plus the fitted size model."""
    if A is None:
        A = arrays(tracker, rec, path)
    n = A["n"]
    area, usable = A["area"], A["usable"]
    length = np.nanmedian(np.where(usable, A["major"], np.nan), 0)
    ops = opportunities(A, length)
    state = np.full(n, IDENTITY)
    for _ in range(N_ITER):
        true = np.array(PERMS)[state]                       # (n, 4): fish index of each label
        # typical area per true fish, then position gain from relative areas
        ref = np.array([np.nanmedian(np.where(usable & (true == k), area, np.nan)) for k in range(4)])
        rel = area / ref[true]
        g = gain_map(np.where(usable, A["cx"], np.nan).ravel(), A["cy"].ravel(),
                     np.where(usable, rel, np.nan).ravel(), np.tile(np.arange(4), n))
        la = np.log(area / g(A["cx"], A["cy"]))
        mu = np.array([np.nanmedian(np.where(usable & (true == k), la, np.nan)) for k in range(4)])
        sd = np.array([1.4826 * np.nanmedian(np.abs(np.where(usable & (true == k), la, np.nan) - mu[k]))
                       for k in range(4)])
        sd = np.maximum(sd, SD_MIN)
        LL = size_loglik(np.nan_to_num(la), usable & np.isfinite(la), mu, sd)
        state = viterbi(LL, ops, n)
    out = dict(state=state, wrong=state != IDENTITY, mu=mu, sd=sd, ref=np.exp(mu), gain=g.grid,
               n_ops=len(ops), fps=A["fps"], fish=A["fish"])
    if return_all:
        out.update(A=A, ops=ops, LL=LL, length=length)
    return out


# --------------------------------------------------------------------------- summaries

def episodes(state, fps, fish, min_s=0.0):
    """Runs of a non-identity state: start, end, labels on the wrong fish."""
    wrong = state != IDENTITY
    m = np.concatenate([[False], wrong, [False]]).astype(np.int8)
    st, en = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    rows = []
    for a, b in zip(st, en):
        if (b - a) / fps < min_s:
            continue
        perms = pd.Series(state[a:b]).value_counts()
        labs = sorted({fish[l] for s in perms.index for l in range(4) if PERMS[s][l] != l})
        rows.append(dict(start=a, end=b - 1, duration_s=(b - a) / fps,
                         labels="+".join(map(str, labs)),
                         main_perm=" ".join(f"{fish[l]}→{fish[PERMS[perms.index[0]][l]]}"
                                            for l in range(4) if PERMS[perms.index[0]][l] != l)))
    return pd.DataFrame(rows, columns=["start", "end", "duration_s", "labels", "main_perm"])


def classify_episodes(ep, state, A, ops, ref, length, guard=3):
    """Evidence behind each decoded episode.

    exchange   : two of the episode's labels exchange positions within `guard` frames of its
                 start or end (an abrupt, physically impossible swap: direct motion evidence)
    size_match : of the swapped labels with >= 10 size frames (size_frames) inside the episode, how many
                 are closer in size to the fish they were decoded onto than to their own fish
    evidence   : 'stacked' (>= STACK_MIN_S long, and two of its labels' centroids are within
                 STACK_FRAC x their summed lengths for most of it: fish lying on top of each
                 other, identity not checkable), 'motion' (exchange, and no label with size
                 data contradicts it),
                 'motion, size disagrees' (exchange, but every label with data still has its own
                 fish's size), 'size, both labels' (every label with data matches and
                 at least two have data), 'size, one label' (only one label deviates: also what
                 a fish that just looks smaller somewhere produces), 'none' (no size data)
    """
    if ep.empty:
        return ep.assign(exchange=[], size_match=[], evidence=[])
    exch_lp = np.log(P_EXCHANGE)
    area = np.where(size_frames(A, length), A["area"], np.nan)
    # open-water typical size per fish, from frames decoded as correct
    true = np.array(PERMS)[state]
    ref = np.array([np.nanmedian(np.where(true == k, area, np.nan)) for k in range(4)])
    out = []
    for e in ep.itertuples():
        perm = PERMS[pd.Series(state[e.start:e.end + 1]).value_counts().index[0]]
        moved = [l for l in range(4) if perm[l] != l]
        frames = list(range(e.start - guard, e.start + guard + 1)) + \
                 list(range(e.end + 1 - guard, e.end + 1 + guard + 1))
        exchange = any(lp == exch_lp and a in moved and b in moved
                       for f in frames for (a, b), lp in ops.get(f, {}).items())
        n_data, n_match = 0, 0
        for l in moved:
            v = area[e.start:e.end + 1, l]
            if np.isfinite(v).sum() < 10:
                continue
            n_data += 1
            a = np.nanmedian(v)
            n_match += abs(np.log(a / ref[perm[l]])) < abs(np.log(a / ref[l]))
        # two fish lying on top of each other for seconds: labels exchange positions as the
        # stack forms and breaks up, but which fish is which in between can't be checked
        pair_d = [np.hypot(A["cx"][e.start:e.end + 1, a] - A["cx"][e.start:e.end + 1, b],
                           A["cy"][e.start:e.end + 1, a] - A["cy"][e.start:e.end + 1, b])
                  < STACK_FRAC * (length[a] + length[b])
                  for a, b in itertools.combinations(moved, 2)]
        stacked = e.duration_s >= STACK_MIN_S and max(np.nanmean(d) for d in pair_d) > 0.5
        if stacked:
            ev = "stacked"
        elif exchange and n_data and n_match == 0:
            ev = "motion, size disagrees"     # e.g. a one-frame centroid glitch, then no exit until the next contact
        elif exchange:
            ev = "motion"
        elif n_data >= 2 and n_match == n_data:
            ev = "size, both labels"
        elif n_match >= 1:
            ev = "size, one label"
        else:
            ev = "none"
        out.append(dict(exchange=exchange, size_match=f"{n_match}/{n_data}" if n_data else "no data",
                        evidence=ev))
    return pd.concat([ep.reset_index(drop=True), pd.DataFrame(out)], axis=1)


CONFIRMED = ("motion", "size, both labels")


def plant_swaps(A, length, k=5, seed=0, min_s=5.0, max_s=60.0):
    """Copy of A with k synthetic swaps: two labels exchange all data between two contact frames.

    Returns (A_planted, truth) where truth[t] is True inside a planted swap.
    """
    rng = np.random.default_rng(seed)
    n, fps = A["n"], A["fps"]
    B = {c: (A[c].copy() if isinstance(A[c], np.ndarray) else A[c]) for c in A}
    truth = np.zeros(n, bool)
    cx, cy = A["cx"], A["cy"]
    pairs = list(itertools.combinations(range(4), 2))
    tries = 0
    planted = []
    while len(planted) < k and tries < 500:
        tries += 1
        a, b = pairs[rng.integers(len(pairs))]
        dist = np.hypot(cx[:, a] - cx[:, b], cy[:, a] - cy[:, b])
        contact = np.flatnonzero(dist < 0.5 * (length[a] + length[b]))
        if contact.size < 2:
            continue
        t1 = int(rng.choice(contact))
        later = contact[(contact > t1 + min_s * fps) & (contact < t1 + max_s * fps)]
        if later.size == 0 or truth[t1:later[-1]].any():
            continue
        t2 = int(rng.choice(later))
        if truth[max(t1 - int(5 * fps), 0):t2 + int(5 * fps)].any():
            continue
        for c in ["area", "cx", "cy", "major", "usable"]:
            B[c][t1:t2, [a, b]] = B[c][t1:t2, [b, a]]
        truth[t1:t2] = True
        planted.append((a, b, t1, t2))
    return B, truth, planted


def video_result(args):
    tracker, rec, path = args
    A = arrays(tracker, rec, path)
    res = decode(tracker, rec, path, A=A, return_all=True)
    length = res["length"]
    fps = A["fps"]
    # positive control: planted swaps
    B, truth, planted = plant_swaps(A, length, k=5, seed=1)
    pl = decode(tracker, rec, path, A=B)
    # planted swaps only count where the real decoding was clean (so real swaps don't inflate recall)
    clean = ~res["wrong"]
    recall = (pl["wrong"] & truth & clean).sum() / max((truth & clean).sum(), 1)
    per_planted = [pl["wrong"][t1:t2].mean() for _, _, t1, t2 in planted]
    ep = classify_episodes(episodes(res["state"], fps, A["fish"]), res["state"], A, res["ops"],
                           res["ref"], length)
    wrong_confirmed = np.zeros(A["n"], bool)
    for e in ep[ep.evidence.isin(CONFIRMED)].itertuples():
        wrong_confirmed[e.start:e.end + 1] = True
    return dict(tracker=tracker, recording=rec, state=res["state"], wrong=res["wrong"],
                wrong_confirmed=wrong_confirmed,
                episodes=ep, ref=res["ref"], sd=res["sd"], n_ops=res["n_ops"],
                planted=planted, planted_recall=recall, planted_detected=per_planted,
                planted_false=(pl["wrong"] & ~truth & clean).mean(), fps=fps, fish=A["fish"])


def run_all(workers=4):
    recs = tc.find_recordings()
    jobs = [(r.tracker, r.recording, r.path) for r in recs.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(video_result, jobs))
    out = os.path.join("figures", "identity_continuity", "results.pkl")
    with open(out, "wb") as f:
        pickle.dump(res, f)
    return res


def _cache_job(args):
    load_table(*args)
    return args[1]


if __name__ == "__main__":
    import sys
    recs = tc.find_recordings()
    jobs = [(r.tracker, r.recording, r.path) for r in recs.itertuples()]
    if sys.argv[1:] == ["cache"]:
        with ProcessPoolExecutor(4) as ex:
            for rec in ex.map(_cache_job, jobs):
                print("cached", rec, flush=True)
    else:
        run_all()
        print("done")
