"""Size-based identity-swap checks for one 4-fish tracking (built for Shiv's ZEUS output).

The four fish differ in size, so a label that ends up on another fish shows up as that label's
mask area moving towards the other fish's typical area - also when the swap is gradual (a mask
sliding from one fish onto another over a contact) rather than a one-frame jump.

Two complementary checks, both on area with unreliable frames removed (untracked, overlapping,
fragmented, area jump: the bad-frame checks of tracking_compare.py):

A. Continuous (anywhere in the video)
   s_i(t)        rolling median area of label i over SMOOTH_S seconds
   drift_i(t)    log(s_i / ref_i) / half the log gap to the neighbouring fish's typical size
                 on that side: 0 = own typical size, 1 = halfway to the neighbour (as close
                 to the neighbour's size as to its own), 2 = the neighbour's size
   swap_ij(t)    |log s_i/ref_i| + |log s_j/ref_j| - |log s_i/ref_j| - |log s_j/ref_i|
                 > 0 when the two labels' sizes fit better exchanged
   A swap episode is a run of swap_ij > 0 lasting >= MIN_EPISODE_S.

B. Encounters (before vs after every close approach)
   For every pair, an encounter is a run of frames where the centroids are closer than half
   the sum of the two fish's typical lengths (bodies can touch). The median area of both
   labels in the WINDOW_S seconds before and after the encounter gives
   enc_swap = |log ai'/ai| + |log aj'/aj| - |log ai'/aj| - |log aj'/ai|   (' = after)
   > 0 when the labels leave the encounter with each other's sizes, i.e. a persistent swap.

ref_i is the label's median area over the whole video, which assumes the labels are right for
most of it.
"""

import itertools

import numpy as np
import pandas as pd

SMOOTH_S = 2.0          # rolling-median window for the continuous size track
MIN_EPISODE_S = 1.0     # shortest swap episode reported
WINDOW_S = 10.0         # before / after window around an encounter
GUARD_S = 1.0           # gap between the encounter and those windows
MERGE_S = 0.5           # encounters closer than this are merged
MIN_VALID = 0.3         # share of valid frames needed in a before / after window


def valid_area(series, flags, fish):
    """Area per label (frames x fish), NaN on frames that fail any bad-frame check."""
    out = {}
    for lab in fish:
        bad = flags.xs(lab, axis=1, level="fish")[
            ["untracked", "centroid_jump", "area_jump", "fragmented", "overlap"]].any(axis=1)
        out[lab] = series[lab].area.where(~bad.values)
    return pd.DataFrame(out)


def references(area, series, fish):
    ref = area.median()
    length = pd.Series({lab: series[lab].major.median() for lab in fish})
    return ref, length


def drift_index(smooth, ref):
    """drift_i(t): signed distance from the own typical size, in half-gaps to the neighbour."""
    order = ref.sort_values().index.tolist()             # small -> large
    lr = np.log(ref)
    out = {}
    for k, lab in enumerate(order):
        d = np.log(smooth[lab]) - lr[lab]
        up = (lr[order[k + 1]] - lr[lab]) / 2 if k + 1 < len(order) else np.nan
        down = (lr[lab] - lr[order[k - 1]]) / 2 if k > 0 else np.nan
        # the largest / smallest fish have no neighbour on one side: use the other side's gap
        up = down if np.isnan(up) else up
        down = up if np.isnan(down) else down
        out[lab] = np.where(d >= 0, d / up, d / down)
    return pd.DataFrame(out, index=smooth.index)


def pair_swap_score(smooth, ref, fish):
    """swap_ij(t) for every pair, columns 'i-j'."""
    lr, ls = np.log(ref), np.log(smooth)
    out = {}
    for i, j in itertools.combinations(fish, 2):
        stay = (ls[i] - lr[i]).abs() + (ls[j] - lr[j]).abs()
        swap = (ls[i] - lr[j]).abs() + (ls[j] - lr[i]).abs()
        out[f"{i}-{j}"] = stay - swap
    return pd.DataFrame(out, index=smooth.index)


def runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e


def swap_episodes(score, smooth, ref, fps, min_s=MIN_EPISODE_S):
    """Runs of swap_ij > 0 lasting >= min_s, one row each, strongest first."""
    rows = []
    for pair in score:
        i, j = map(int, pair.split("-"))
        sc = score[pair].values
        for s, e in zip(*runs(sc > 0)):
            if e - s < min_s * fps:
                continue
            pk = s + int(np.nanargmax(sc[s:e]))
            rows.append(dict(pair=pair, start=s, end=e - 1, peak_frame=pk,
                             duration_s=(e - s) / fps, start_min=s / fps / 60,
                             peak_score=sc[pk], mean_score=np.nanmean(sc[s:e]),
                             **{f"area_{i}/ref_{i}": smooth[i].values[pk] / ref[i],
                                f"area_{i}/ref_{j}": smooth[i].values[pk] / ref[j],
                                f"area_{j}/ref_{j}": smooth[j].values[pk] / ref[j],
                                f"area_{j}/ref_{i}": smooth[j].values[pk] / ref[i]}))
    cols = ["pair", "start", "end", "peak_frame", "duration_s", "start_min", "peak_score",
            "mean_score"]
    df = pd.DataFrame(rows)
    return df[cols + [c for c in df.columns if c not in cols]].sort_values(
        "peak_score", ascending=False, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def encounters(series, length, fish, fps, merge_s=MERGE_S):
    """Close approaches per pair: (pair, start, end) frame ranges."""
    rows = []
    for i, j in itertools.combinations(fish, 2):
        d = np.hypot(series[i].cx - series[j].cx, series[i].cy - series[j].cy).values
        close = d < 0.5 * (length[i] + length[j])
        s, e = runs(close)
        # merge encounters separated by short gaps
        ms, me = [], []
        for a, b in zip(s, e):
            if ms and a - me[-1] < merge_s * fps:
                me[-1] = b
            else:
                ms.append(a)
                me.append(b)
        for a, b in zip(ms, me):
            rows.append(dict(pair=f"{i}-{j}", i=i, j=j, start=a, end=b - 1,
                             duration_s=(b - a) / fps,
                             min_dist=np.nanmin(d[a:b])))
    return pd.DataFrame(rows)


def encounter_swap(enc, area, fps, window_s=WINDOW_S, guard_s=GUARD_S, min_valid=MIN_VALID):
    """Before / after median areas and enc_swap for every encounter."""
    w, g, n = int(window_s * fps), int(guard_s * fps), len(area)
    out = []
    for r in enc.itertuples():
        b0, b1 = max(r.start - g - w, 0), max(r.start - g, 0)
        a0, a1 = min(r.end + 1 + g, n), min(r.end + 1 + g + w, n)
        vals = {}
        for lab in (r.i, r.j):
            pre, post = area[lab].values[b0:b1], area[lab].values[a0:a1]
            ok = (np.isfinite(pre).mean() >= min_valid if pre.size else False) and \
                 (np.isfinite(post).mean() >= min_valid if post.size else False)
            vals[lab] = (np.nanmedian(pre) if ok else np.nan, np.nanmedian(post) if ok else np.nan)
        (bi, ai), (bj, aj) = vals[r.i], vals[r.j]
        stay = abs(np.log(ai / bi)) + abs(np.log(aj / bj))
        swap = abs(np.log(ai / bj)) + abs(np.log(aj / bi))
        out.append(dict(r._asdict(), before_i=bi, after_i=ai, before_j=bj, after_j=aj,
                        enc_swap=stay - swap))
    df = pd.DataFrame(out).drop(columns="Index")
    df["start_min"] = df.start / fps / 60
    return df


# --------------------------------------------------------------------------- data / checks

def load_shiv_series(rec, path, mm_cache=None):
    """Per-label series (area, centroid, length, ...) of a Shiv recording, remnants dropped.

    Reuses mask_metrics.py's per-frame cache when given, else computes (slow, minutes).
    """
    import os
    import pickle

    import mask_metrics as mm
    import tracking_compare as tc
    if mm_cache and os.path.exists(mm_cache):
        with open(mm_cache, "rb") as f:
            df, _, meta = pickle.load(f)
    else:
        df, _, meta = mm.compute_metrics(path, tc.shiv_src())
    fish = tc.FISH["Shiv"]
    return mm.per_fish_series(df[df.label.isin(fish)], meta["n_frames"], meta["fps"], fish), meta


def inject_swap(area, series, i, j, start, ramp_s, fps):
    """Positive control: labels i and j exchange fish from `start` on, blended in over ramp_s.

    During the ramp each label's area moves linearly from its own fish to the other's,
    imitating a mask sliding from one fish onto the other.
    """
    a = area.copy()
    n = len(a)
    w = np.clip((np.arange(n) - start) / max(ramp_s * fps, 1), 0, 1)
    ai, aj = area[i].values, area[j].values
    a[i], a[j] = (1 - w) * ai + w * aj, (1 - w) * aj + w * ai
    return a


def contact_sheet(mp4, frames, out, fps, width=456):
    """Frames of an annotated video side by side in one png (via ffmpeg)."""
    import os
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp()
    parts = []
    for k, f in enumerate(frames):
        p = os.path.join(tmp, f"{k}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{f / fps:.4f}", "-i", mp4,
                        "-frames:v", "1", "-vf", f"scale={width}:-2", p], check=True)
        parts.append(p)
    ins = sum((["-i", p] for p in parts), [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex",
                    f"hstack=inputs={len(parts)}", out], check=True)
    return out


def analyse(rec, path, mm_cache=None):
    """Both checks for one Shiv recording; dict with every intermediate table."""
    import tracking_compare as tc
    series, meta = load_shiv_series(rec, path, mm_cache)
    fps, fish = meta["fps"], tc.FISH["Shiv"]
    flags = tc.frame_flags(tc.load_metrics("Shiv", rec, path)["metrics"])
    area = valid_area(series, flags, fish)
    ref, length = references(area, series, fish)
    w = int(SMOOTH_S * fps)
    smooth = area.rolling(w, center=True, min_periods=w // 2).median()
    score = pair_swap_score(smooth, ref, fish)
    enc = encounters(series, length, fish, fps)
    return dict(series=series, meta=meta, fps=fps, fish=fish, flags=flags, area=area, ref=ref,
                length=length, smooth=smooth, drift=drift_index(smooth, ref), score=score,
                episodes=swap_episodes(score, smooth, ref, fps), enc=enc,
                enc_swap=encounter_swap(enc, area, fps))


def mmss(frame, fps):
    """Video time of a frame as m:ss.s"""
    t = frame / fps
    return f"{int(t // 60)}:{t % 60:04.1f}"
