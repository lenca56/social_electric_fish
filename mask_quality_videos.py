"""Which videos have the best and worst masks? The mask metrics of mask_metrics.py (stage_quality, stage_checks,
stage_summary) on every frame of each video's final efish4-track masks (read with io_utils.load_stage_masks(run, 'final')),
one table row per video, ranked within each group (videos of different groups come from different pipeline versions and
fish numbers, so they are not ranked together).

Groups:
  3fish   data/3_fish_silent/masks/<recording>                       (efish4-track 1.1.0)
  4fish   data/track_Kaden/results_<t>_full/video2024-02-20T<t>       (efish4-track 1.0.0; the 2-/3-fish controls in
                                                                      results_21_56_26_full are left out)

Ranking: % of fish-frames passing every QC rule of kaden_mask_qc.py except 'not touching other fish' (touching is where
the fish are, not how good the mask is), mean over the fish of the video. Also reported per video: % touching, % with no
mask, % unstable (centroid-aligned IoU with the previous frame < 0.6), % swap candidates, % fragmented, % with holes,
% needs_review, median aligned IoU.

    OMP_NUM_THREADS=1 ~/opt/anaconda3/envs/electric_fish/bin/python mask_quality_videos.py [--groups 3fish 4fish] [--workers 7]

Output in figures/mask_metrics/videos/: <group>/<recording>.csv.gz (per frame and fish, cached; --recompute redoes it),
videos.csv (one row per video and group, with rank), videos_<group>.png.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import mask_metrics as mm

OUT = os.path.join(mm.OUT_DIR, "videos")
GROUPS = {
    "3fish": "data/3_fish_silent/masks/*/outputs/*.pkl",
    "4fish": "data/track_Kaden/results_*_full/video2024-02-20T*/outputs/*.pkl",
}
KEEP = ["frame", "fish", "px", "cx", "cy", "major", "n_comp", "largest_frac", "solidity", "hole_frac", "contact",
        "overlap_px", "vis_in_amodal", "iou_prev", "iou_prev_aligned", "iou_prev_other", "occluded", "needs_review"]


def runs(group):
    """run folders of a group with a final pickle (io_utils.stage_mask_file(run, 'final'))"""
    from io_utils import stage_mask_file
    found = {os.path.dirname(os.path.dirname(p)) for p in glob.glob(GROUPS[group])}
    return sorted(r for r in found if stage_mask_file(r, "final"))


def per_frame(args):
    """per-frame metrics of one video's final masks, cached"""
    group, run, recompute = args
    rec = os.path.basename(run)
    cache = os.path.join(OUT, group, f"{rec}.csv.gz")
    if os.path.exists(cache) and not recompute:
        return cache
    t0 = time.time()
    df = mm.stage_quality(run, "final", None, "final")
    df[[c for c in KEEP if c in df]].round(4).to_csv(cache, index=False)
    print(f"  {group} {rec}: {df.frame.nunique()} frames, {(time.time() - t0) / 60:.1f} min", flush=True)
    return cache


def video_row(group, run, cache):
    df = pd.read_csv(cache)
    df.insert(0, "stage", "final")
    for c in ("iou_prev_stage", "iou_final"):
        df[c] = np.nan
    df, checks = mm.stage_checks(df)
    s = mm.stage_summary(df, checks)
    g = s.drop(columns=["stage", "fish"]).mean()                 # mean over the video's fish
    return dict(group=group, recording=os.path.basename(run), frames=df.frame.nunique(), fish=df.fish.nunique(),
                **{"% pass all but touching": g["% pass all but touching"],
                   "worst fish % pass all but touching": s["% pass all but touching"].min(),
                   "% touching": g["% fail: not touching other fish"],
                   "% no mask": g["% fail: has a mask"],
                   "% area off median": g["% fail: area within 20% of median"],
                   "% length off median": g["% fail: length 0.8-1.2 x median"],
                   "% fragmented": g["% fail: >= 99% in one piece"],
                   "% step > 15 px": g["% fail: step <= 15 px"],
                   "% area change > 15%": g["% fail: area change <= 15%"],
                   f"% aligned IoU < {mm.IOU_LOW}": g[f"% aligned temporal IoU < {mm.IOU_LOW}"],
                   "% swap candidate": g["% swap candidate"],
                   "% with holes": g["% with holes"],
                   "% needs_review": g.get("% needs_review flag", np.nan),
                   "median aligned IoU": g["median aligned temporal IoU"],
                   "track fragments": s["track fragments"].sum()})


def plot_group(v, group, path):
    """one dot row per video, sorted by % pass all but touching, with the other failure rates beside it"""
    import matplotlib.pyplot as plt
    v = v.sort_values("% pass all but touching").reset_index(drop=True)
    cols = ["% pass all but touching", "% touching", "% fragmented", "% step > 15 px", f"% aligned IoU < {mm.IOU_LOW}",
            "% swap candidate", "% no mask"]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.4 * len(cols), 0.22 * len(v) + 1.6), sharey=True)
    y = np.arange(len(v))
    for ax, c in zip(axes, cols):
        ax.hlines(y, 0, v[c], color=mm.GRID, lw=1)
        ax.plot(v[c], y, "o", color=mm.PALETTE[3] if c == cols[0] else mm.MUTED, ms=4)
        ax.set_title(c, loc="left", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[0].set_yticks(y, v.recording.str.replace("video2024-", ""), fontsize=6.5)
    fig.suptitle(f"{group}: final masks per video, worst (top) to best (bottom) by % fish-frames passing every QC rule "
                 "but touching", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    rows = []
    for group in a.groups:
        rs = runs(group)
        os.makedirs(os.path.join(OUT, group), exist_ok=True)
        print(f"{group}: {len(rs)} videos", flush=True)
        with ProcessPoolExecutor(min(a.workers, len(rs))) as ex:
            caches = list(ex.map(per_frame, [(group, r, a.recompute) for r in rs]))
        v = pd.DataFrame([video_row(group, r, c) for r, c in zip(rs, caches)])
        v.insert(1, "rank", v["% pass all but touching"].rank(ascending=False, method="min").astype(int))
        v = v.sort_values("rank")
        plot_group(v, group, os.path.join(OUT, f"videos_{group}.png"))
        rows.append(v)
    v = pd.concat(rows, ignore_index=True)
    path = os.path.join(OUT, "videos.csv")
    if os.path.exists(path):                                   # keep the groups not rerun this time
        old = pd.read_csv(path)
        v = pd.concat([old[~old.group.isin(a.groups)], v], ignore_index=True)
    v.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(v.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
