"""Per-frame tracking-quality flags for Shiv's (ZEUS) and Kaden's (efish4-track) 4-fish masks.

Used by Tracking_comparison_Shiv_Kaden.ipynb. Per-fish shape / area / centroid metrics come
from mask_metrics.py; this module turns them into one boolean flag table per recording and
caches the numeric per-fish metrics under figures/tracking_comparison/cache/; flags are derived
from them with adjustable thresholds (frame_flags).

A frame is BAD when, for any of the four fish:
  untracked      the fish has no mask
  centroid_jump  centroid speed > SPEED_JUMP px/s (vs the previous frame)
  area_jump      |relative area change| > AREA_JUMP (vs the previous frame)
  fragmented     >= FRAG_FRAC of the mask lies outside its largest connected component
                 (8-connectivity), so 1-10 px specks next to an intact mask don't count
  overlap        the mask shares pixels with another fish's mask
Kaden's own flags (occluded, inferred, needs_review) and confidence are kept for reporting
only and do not enter the good/bad decision.

Precompute every cache (slow part, a few minutes per recording) with
    ~/opt/anaconda3/envs/electric_fish/bin/python tracking_compare.py [--workers 3]
"""

import argparse
import glob
import gzip
import os
import pickle
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import mask_metrics as mm

DATA_DIR = "data"
OUT_DIR = os.path.join("figures", "tracking_comparison")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
CLIP_DIR = os.path.join(OUT_DIR, "clips")

AREA_JUMP = 0.30        # relative frame-to-frame area change
SPEED_JUMP = 1500.0     # centroid speed (px/s); 30 px/frame at 50 fps
FRAG_FRAC = 0.05        # share of the mask outside its largest connected component
BAD_FLAGS = ["untracked", "centroid_jump", "area_jump", "fragmented", "overlap"]
KADEN_FLAGS = ["occluded", "inferred", "needs_review"]

TRACKERS = ["Shiv", "Kaden"]
TRACKER_COLOR = {"Shiv": "#2a78d6", "Kaden": "#eda100"}   # CVD-validated pair
FISH = {"Shiv": [1, 2, 3, 0], "Kaden": [0, 1, 2, 3]}
FISH_NAME = {"Shiv": {1: "1 largest", 2: "2 second", 3: "3 third", 0: "0 smallest"},
             "Kaden": {i: f"fish {c}" for i, c in enumerate("ABCD")}}

FONT = "/System/Library/Fonts/Helvetica.ttc"


# --------------------------------------------------------------------------- sources

def load_shiv(path, mask_kind=None):
    """ZEUS tracking_final.pkl(.gz) as mask_metrics frame records, remnant labels (>= 4) dropped."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        d = pickle.load(f)
    frames = d["frames"]

    def gen():
        for k in sorted(frames):
            fr = frames[k]
            keep = [i for i, lab in enumerate(fr["labels"]) if lab < 4]
            yield dict(frame=k,
                       labels=[fr["labels"][i] for i in keep],
                       boxes=[fr["boxes"][i] for i in keep],
                       crops=[mm.unpack_crop(fr["masks"][i], fr["boxes"][i]) for i in keep],
                       areas=[fr["areas"][i] for i in keep],
                       obj_ids=[fr["obj_ids"][i] for i in keep],
                       extra=[{}] * len(keep), frame_extra={})

    meta = dict(fps=d["fps"], shape=d["shape"], n_frames=len(frames), video=d["video"])
    return gen(), meta


def shiv_src():
    return dict(name="shiv", load=load_shiv, mask_kind=None)


def kaden_src():
    return dict(name="kaden", load=mm.load_kaden, mask_kind="visible")


def recording_id(text):
    """'2024-02-20T21_56_26' from any path / video string, or None."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})[T_](\d{2}_\d{2}_\d{2})", text)
    return f"{m.group(1)}T{m.group(2)}" if m else None


def find_recordings():
    """DataFrame tracker, group, recording, path, annotated_mp4 (None when not on disk)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "track_Shiv", "Group_*",
                                           "*tracking_final.pkl.gz"))):
        group = os.path.basename(os.path.dirname(p))
        name = os.path.basename(p).replace("_tracking_final.pkl.gz", "")
        rec = recording_id(name)
        if rec is None:          # Group_01 names carry only the time; the day is 2024-02-20
            t = re.search(r"(\d{2}_\d{2}_\d{2})$", name).group(1)
            rec = f"2024-02-20T{t}"
        mp4 = os.path.join(DATA_DIR, p.split(os.sep)[-1].replace(".pkl.gz", ".mp4"))
        rows.append(dict(tracker="Shiv", group=group, recording=rec, path=p,
                         annotated_mp4=mp4 if os.path.exists(mp4) else None))
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "track_Kaden", "results_*", "*",
                                           "outputs", "*.pkl"))):
        if os.path.basename(p).startswith("._"):
            continue
        mp4 = p.replace(".pkl", "_annotated.mp4")
        rows.append(dict(tracker="Kaden", group="Group_01", recording=recording_id(p),
                         path=p, annotated_mp4=mp4 if os.path.exists(mp4) else None))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- flags

def fish_metrics(df, n_frames, fps, fish):
    """Per-frame numeric metrics, columns (metric, fish); NaN where the fish is untracked."""
    series = mm.per_fish_series(df, n_frames, fps, fish)
    keep = ["present", "speed", "area_change", "largest_frac", "overlap_px",
            *KADEN_FLAGS, "confidence", "overlap_px_amodal"]
    out = pd.concat({lab: series[lab][[c for c in keep if c in series[lab]]]
                     for lab in fish}, axis=1).swaplevel(axis=1).sort_index(axis=1)
    out.columns.names = ["metric", "fish"]
    return out


def frame_flags(metrics, area_jump=AREA_JUMP, speed_jump=SPEED_JUMP, frag_frac=FRAG_FRAC):
    """Per-frame flag table, columns (flag, fish) bool (plus Kaden's numeric confidence)."""
    cols = {}
    fish = list(dict.fromkeys(metrics.columns.get_level_values("fish")))
    have = set(metrics.columns.get_level_values("metric"))
    for lab in fish:
        m = metrics.xs(lab, axis=1, level="fish")
        cols[("untracked", lab)] = ~m.present.eq(True)
        cols[("centroid_jump", lab)] = m.speed > speed_jump
        cols[("area_jump", lab)] = m.area_change.abs() > area_jump
        cols[("fragmented", lab)] = (1 - m.largest_frac) >= frag_frac
        cols[("overlap", lab)] = m.overlap_px > 0
        for c in KADEN_FLAGS:
            if c in have:
                cols[(c, lab)] = m[c].eq(True)
        if "overlap_px_amodal" in have:
            cols[("overlap_amodal", lab)] = m.overlap_px_amodal > 0
        if "confidence" in have:
            cols[("confidence", lab)] = m.confidence
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["flag", "fish"])
    return out


def cache_path(tracker, rec):
    return os.path.join(CACHE_DIR, f"{tracker}_{rec}.pkl")


def load_metrics(tracker, rec, path, recompute=False):
    """Per-frame metrics table + meta for one recording, cached."""
    cp = cache_path(tracker, rec)
    if os.path.exists(cp) and not recompute:
        with open(cp, "rb") as f:
            return pickle.load(f)
    if tracker == "Kaden":
        # mask_metrics.py --source kaden already caches the identical per-frame metrics
        mcache = os.path.join(mm.OUT_DIR, "kaden", f"video{rec}", "metrics.pkl")
        if os.path.exists(mcache) and os.path.getmtime(mcache) > os.path.getmtime(path):
            with open(mcache, "rb") as f:
                df, _, meta = pickle.load(f)
        else:
            df, _, meta = mm.compute_metrics(path, kaden_src())
    else:
        df, _, meta = mm.compute_metrics(path, shiv_src())
    metrics = fish_metrics(df, meta["n_frames"], meta["fps"], FISH[tracker])
    out = dict(metrics=metrics, meta=dict(fps=meta["fps"], n_frames=meta["n_frames"],
                                      shape=tuple(meta["shape"]), video=meta["video"]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(out, f)
    print(f"done {tracker} {rec}", flush=True)
    return out


def bad_frames(flags, which=BAD_FLAGS):
    """Bool Series per frame: any fish has any of `which`."""
    return flags.loc[:, flags.columns.get_level_values("flag").isin(which)].any(axis=1)


def runs(mask):
    """(start, length) of every run of True in a bool array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


def recording_summary(tracker, group, rec, flags, fps):
    """One row of video-level metrics (all % are of the video's frames)."""
    bad = bad_frames(flags).values
    n = len(bad)
    row = dict(tracker=tracker, group=group, recording=rec, n_frames=n,
               minutes=n / fps / 60, pct_good=100 * (1 - bad.mean()))
    for fl in BAD_FLAGS:
        row[f"pct_{fl}"] = 100 * bad_frames(flags, [fl]).mean()
    _, good_len = runs(~bad)
    b_start, _ = runs(bad)
    row["longest_good_run_s"] = good_len.max() / fps if good_len.size else 0.0
    row["pct_in_good_runs_ge_2s"] = 100 * good_len[good_len >= 2 * fps].sum() / n
    row["bad_episodes_per_min"] = b_start.size / (n / fps / 60)
    fl_names = flags.columns.get_level_values("flag")
    for c in KADEN_FLAGS + ["overlap_amodal"]:
        row[f"pct_{c}"] = (100 * bad_frames(flags, [c]).mean()) if c in fl_names else np.nan
    row["confidence_median"] = (np.nanmedian(flags["confidence"].values)
                                if "confidence" in fl_names else np.nan)
    return row


def fish_summary(tracker, rec, flags):
    """Per-fish % good frames and % of frames per flag."""
    out = []
    for lab in FISH[tracker]:
        sub = flags.xs(lab, axis=1, level="fish")
        bad = sub[BAD_FLAGS].any(axis=1)
        row = dict(tracker=tracker, recording=rec, fish=FISH_NAME[tracker][lab],
                   pct_good=100 * (1 - bad.mean()))
        for fl in BAD_FLAGS:
            row[f"pct_{fl}"] = 100 * sub[fl].mean()
        out.append(row)
    return out


# --------------------------------------------------------------------------- snippets

def pick_windows(bad, fps, n, kind, length_s=4.0, seed=0, exclude=()):
    """Start frames of `n` non-overlapping windows of `length_s` seconds.

    kind='good': windows with no bad frame, drawn at random (seeded).
    kind='bad' : the windows with the most bad frames, greedily, centred on those frames.
    exclude: (start, stop) frame ranges to avoid (e.g. the first seconds of the video).
    """
    w = int(round(length_s * fps))
    b = np.asarray(bad, int)
    score = np.convolve(b, np.ones(w, int), "valid")       # bad frames in [s, s + w)
    ok = np.ones(score.size, bool)
    for a, z in exclude:
        ok[max(a - w + 1, 0):max(z, 0)] = False
    picks = []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        if kind == "good":
            cand = np.flatnonzero(ok & (score == 0))
            if cand.size == 0:
                break
            s = int(rng.choice(cand))
        else:
            cand = np.where(ok, score, -1)
            s = int(np.argmax(cand))
            if cand[s] <= 0:
                break
            # centre the window on its flagged frames (argmax alone puts a lone event last)
            hit = np.flatnonzero(b[s:s + w]) + s
            s = int(np.clip(round(hit.mean()) - w // 2, 0, score.size - 1))
        picks.append(s)
        ok[max(s - w + 1, 0):s + w] = False
    return picks


def make_clip(mp4, start, n_frames, fps, out, title, bad_mask, width=640):
    """Cut frames [start, start + n_frames) of an annotated mp4, re-encoded to h264.

    The title is burned in at the bottom; a red border is drawn on every frame flagged bad
    (bad_mask: bool array of length n_frames, relative to start).
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    starts, lens = runs(bad_mask)
    enable = "+".join(f"between(n,{s},{s + l - 1})" for s, l in zip(starts, lens)) or "0"
    txt = title.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
    vf = (f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.9:t=8:enable='{enable}',"
          f"drawtext=fontfile={FONT}:text='{txt}':x=12:y=h-th-14:fontsize=26:"
          f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
          f"scale={width}:-2")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start / fps:.4f}", "-i", mp4,
           "-frames:v", str(n_frames), "-vf", vf, "-an", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "26", "-preset", "veryfast",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return out


# --------------------------------------------------------------------------- main

def _job(args):
    tracker, rec, path = args
    load_metrics(tracker, rec, path)
    return tracker, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    recs = find_recordings()
    todo = [(r.tracker, r.recording, r.path) for r in recs.itertuples()
            if not os.path.exists(cache_path(r.tracker, r.recording))]
    print(f"{len(todo)} of {len(recs)} recordings to compute", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for tracker, rec in ex.map(_job, todo):
            pass


if __name__ == "__main__":
    main()
