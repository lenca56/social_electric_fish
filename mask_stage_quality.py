"""Mask quality at every step of an efish4-track run: the stage pickles outputs/stages/01_segment.pkl ... 10_disturbance.pkl.

Masks are read with io_utils.load_stage_masks; every metric is computed in mask_metrics.py (stage_quality, stage_checks,
stage_summary); this script runs it over the stages in parallel and makes the plots (also used by
mask_quality_by_stage.ipynb). Per stage, frame and fish (visible masks):
  my QC rules (kaden_mask_qc.py)  has a mask, area within 20% / length 0.8-1.2 x the fish's median, >= 99% in one piece,
                                  step <= 15 px, area change <= 15% (to the previous and next frame), not within 5 px of
                                  another fish; 'all' = every rule passes
  shape                           solidity (area / convex hull), holes, number of pieces
  temporal                        IoU with the fish's own mask in the previous frame; identity-swap candidate when the
                                  mask overlaps another fish's previous mask more than its own
  exclusivity / amodal            pixels shared with another fish's visible mask (should be 0), fraction of the visible
                                  mask inside the fish's own full-body mask (should be 1)
  tracker flags                   occluded, inferred, needs_review
  stage change                    IoU with the same mask in the previous stage and in the final stage
Track fragments = number of separate runs of frames in which the fish has a mask (MOT 'Frag').

    ~/opt/anaconda3/envs/electric_fish/bin/python mask_stage_quality.py data/3_fish_silent/masks/video2024-10-15T15_30_54 [--recompute] \
        [--examples 50]

Works on finished runs (outputs/stages/NN_<stage>.pkl) and on running ones (work/<stage>/masks.json, stages done so far).
Output in figures/mask_metrics/stages/<recording>/:
  per_frame.csv.gz  one row per (stage, frame, fish): metrics and checks (cached; --recompute redoes it)
  summary.csv       one row per (stage, fish): % of frames per check and flag, metric medians
  flux.csv.gz       per (step, frame, fish): pixels the step adds, removes, takes from / gives to another fish (cached)
  step_changes.csv  one row per step: % changed, share of all change, pixels moved, distance to final, QC fixed / broken,
                    run time (mask_metrics.stage_change_table)
  *.png             the plots (06_step_changes.png: how much every step changes the masks)
With --examples N: results/Kaden_mask_quality/<recording>/ holds the N frames the pipeline redraws most (sum over steps of
1 - IoU with the previous step's mask, examples >= --min-gap frames apart), each as the video crop + the masks of every
step (rank<NN>_frame<k>_fish<X>.png), and examples.csv.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import mask_metrics as mm
from io_utils import STAGES, stage_mask_file

OUT = os.path.join(mm.OUT_DIR, "stages")
IOU_LOW = mm.IOU_LOW


def stages_done(run):
    """The pipeline stages of a run that have masks, in order."""
    return [s for s in STAGES if stage_mask_file(run, s)]


def compute(run, workers=5):
    """mask_metrics.stage_quality of every finished stage (in parallel), each compared with the previous and last stage."""
    stages = stages_done(run)
    if not stages:
        raise SystemExit(f"no stage masks in {run}")
    with ProcessPoolExecutor(min(workers, len(stages))) as ex:
        parts = list(ex.map(mm.stage_quality, [run] * len(stages), stages, [None] + stages[:-1],
                            [stages[-1]] * len(stages)))
    return pd.concat(parts, ignore_index=True)


def load_or_compute(run, recompute=False, workers=5):
    """(per-frame table with checks, summary table, rule names, output folder); per-frame metrics cached in the folder."""
    rec = os.path.basename(os.path.normpath(run))
    out = os.path.join(OUT, rec)
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "per_frame.csv.gz")
    if os.path.exists(cache) and not recompute:
        df = pd.read_csv(cache)
    else:
        print(f"{rec}: computing", flush=True)
        df = compute(run, workers)
    df, checks = mm.stage_checks(df)
    df.to_csv(cache, index=False)
    s = mm.stage_summary(df, checks)
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    return df, s, checks, out


def load_or_compute_changes(run, df, recompute=False, workers=5):
    """(step table, flux): mask_metrics.stage_flux of every step against the one before (cached as flux.csv.gz) and
    mask_metrics.stage_change_table with the run times of outputs/summary.json; the table is saved as step_changes.csv."""
    from io_utils import load_run_summary
    rec = os.path.basename(os.path.normpath(run))
    out = os.path.join(OUT, rec)
    cache = os.path.join(out, "flux.csv.gz")
    stages = stages_done(run)
    if os.path.exists(cache) and not recompute:
        flux = pd.read_csv(cache)
    else:
        with ProcessPoolExecutor(min(workers, len(stages) - 1)) as ex:
            flux = pd.concat(list(ex.map(mm.stage_flux, [run] * (len(stages) - 1), stages[1:], stages[:-1])),
                             ignore_index=True)
        flux.to_csv(cache, index=False)
    cost = load_run_summary(run)
    secs = None if cost is None else {mm.stage_label(s): v for s, v in zip(cost.stage, cost.seconds) if s in stages}
    t = mm.stage_change_table(df, flux, secs)
    t.to_csv(os.path.join(out, "step_changes.csv"), index=False)
    return t, flux


def plot_step_changes(t, df, rec, path=None):
    """How much each step changes the masks, how far they are from the final ones after it, what it does to the QC
    rules, and what it costs."""
    steps = list(t.step)
    x = np.arange(len(steps))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ax = axes[0, 0]                                      # pixels moved, stacked
    bottom = np.zeros(len(steps))
    for c, colr in (("% px added", mm.PALETTE[1]), ("% px removed", mm.PALETTE[0]),
                    ("% px moved between fish", mm.PALETTE[3])):
        v = t[c].fillna(0).to_numpy()
        ax.bar(x, v, 0.7, bottom=bottom, color=colr, label=c.replace("% px ", ""), edgecolor="white", lw=1)
        bottom += v
    ax.set_title("pixels changed, % of all mask pixels of the previous step", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[0, 1]                                      # share of all change + cumulative
    share = t["% of all change"].fillna(0).to_numpy()
    ax.bar(x, share, 0.7, color=mm.PALETTE[3], label="this step")
    ax.plot(x, np.cumsum(share), "-o", color=mm.INK, lw=1.5, ms=4, label="cumulative")
    ax.set_ylim(0, 105)
    ax.set_title("share of all mask change along the pipeline (%)", loc="left", fontsize=9)
    ax.legend(fontsize=8, loc="center left")
    ax = axes[0, 2]                                      # identical to final after each step
    d = df[df.iou_final.notna()]
    touch = d.contact.astype("boolean").fillna(False).to_numpy(bool)
    for name, sel, colr in (("fish touching another", touch, mm.PALETTE[3]), ("fish isolated", ~touch, mm.MUTED)):
        g = d[sel].groupby("stage", sort=False).iou_final
        ax.plot(x, g.apply(lambda s: 100 * (s >= 0.95).mean()).reindex(steps), "-o", color=colr, lw=2, ms=5,
                mec="white", label=f"{name}, IoU >= 0.95")
        ax.plot(x, g.apply(lambda s: 100 * (s == 1).mean()).reindex(steps), "--", color=colr, lw=1.2,
                label=f"{name}, identical")
    ax.set_ylim(0, 105)
    ax.set_title("% of fish-frames already (nearly) equal to the final mask", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1, 0]                                      # QC fixed / broken
    ax.bar(x, t["% QC fixed"].fillna(0), 0.7, color=mm.PALETTE[1], label="fixed (fail -> pass)")
    ax.bar(x, -t["% QC broken"].fillna(0), 0.7, color=mm.PALETTE[0], label="broken (pass -> fail)")
    ax.axhline(0, color=mm.MUTED, lw=0.8)
    ax.set_title("QC rules but touching: fish-frames this step fixes / breaks (%)", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(x, t["% pass all but touching"], "-o", color=mm.INK, lw=2, ms=5)
    ax.set_title("% of fish-frames passing every QC rule but touching, after the step", loc="left", fontsize=9)
    ax = axes[1, 2]
    if "run time (s)" in t:
        ax.bar(x, t["run time (s)"] / 60, 0.7, color=mm.MUTED)
        ax.set_yscale("log")
        for xi, v in zip(x, t["run time (s)"] / 60):
            ax.text(xi, v * 1.15, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", fontsize=7, color=mm.INK)
        ax.set_title("run time of the step (min, log scale)", loc="left", fontsize=9)
    for ax in axes.flat:
        stage_axis(ax, steps)
    fig.suptitle(f"{rec} - how much every pipeline step changes the visible masks", fontsize=10, x=0.01, ha="left")
    return finish(fig, path)


def run_fps(run):
    with open(os.path.join(run, "run.json")) as f:
        return json.load(f)["video_properties"]["fps"]


# --------------------------------------------------------------------------- plots
import matplotlib.pyplot as plt                          # noqa: E402  (mask_metrics sets the Agg backend and style)


def finish(fig, path):
    """tight layout, save if a path is given; returns the figure (the caller closes it)"""
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def fish_colors(fish):
    return dict(zip(fish, mm.PALETTE))


def stage_axis(ax, stages, labels=True):
    ax.set_xticks(range(len(stages)), stages if labels else [], rotation=40, ha="right")
    ax.set_xlim(-0.4, len(stages) - 0.6)


def plot_qc_heatmap(s, checks, rec, path=None):
    """% of fish-frames failing each of my QC rules, per stage (pooled over fish)."""
    stages = list(dict.fromkeys(s.stage))
    rows = [f"% fail: {c}" for c in checks]
    m = s.groupby("stage", sort=False)[rows].mean().T
    m.loc["% fail: any rule"] = 100 - s.groupby("stage", sort=False)["% pass all"].mean()
    m.loc["% fail: any rule but touching"] = 100 - s.groupby("stage", sort=False)["% pass all but touching"].mean()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.imshow(m.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(m.to_numpy().max(), 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 * m.to_numpy().max() else mm.INK)
    ax.set_yticks(range(m.shape[0]), [r.replace("% fail: ", "") for r in m.index])
    ax.axhline(len(rows) - 0.5, color="white", lw=3)
    stage_axis(ax, stages)
    ax.grid(False)
    ax.set_title(f"{rec} - % of fish-frames failing each QC rule of kaden_mask_qc.py, per step", loc="left")
    return finish(fig, path)


def plot_metrics_by_stage(s, checks, rec, path=None):
    """Small multiples: one metric per panel, x = stage, one line per fish."""
    stages = list(dict.fromkeys(s.stage))
    fish = list(dict.fromkeys(s.fish))
    col = fish_colors(fish)
    panels = ([f"% fail: {c}" for c in checks] + ["% pass all", "% pass all but touching",
              f"% aligned temporal IoU < {IOU_LOW}", "median aligned temporal IoU", "% swap candidate",
              "median solidity", "% with holes", "% changed vs previous stage", "mean IoU with final"])
    nc = 4
    nr = -(-len(panels) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(14, 2.5 * nr), sharex=True)
    for ax, p in zip(axes.flat, panels):
        for f in fish:
            v = s[s.fish == f].set_index("stage").loc[stages, p]
            ax.plot(range(len(stages)), v, "-o", color=col[f], lw=2, ms=4, mec="white", mew=0.8, label=f"fish {f}")
        ax.set_title(p.replace("% fail: ", "% fail: "), loc="left", fontsize=9)
        stage_axis(ax, stages)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{rec} - mask quality metrics at every pipeline step (one line per fish)",
                 fontsize=10, x=0.01, ha="left")
    return finish(fig, path)


def plot_stage_changes(df, rec, path=None):
    """What each step changes: % of masks changed vs the previous step, fish touching vs isolated, and their IoU."""
    stages = list(dict.fromkeys(df.stage))
    d = df[df.present]
    touch = d.contact.astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    x = np.arange(len(stages))
    for o, (name, sel, c) in zip((-0.2, 0.2), (("fish touching another", touch, mm.PALETTE[3]),
                                               ("fish isolated", ~touch, mm.MUTED))):
        pct = d[sel].groupby("stage", sort=False).changed.mean().reindex(stages) * 100
        axes[0].bar(x + o, pct, 0.38, color=c, label=name)
        iou = d[sel & d.changed].groupby("stage", sort=False).iou_prev_stage.median().reindex(stages)
        axes[1].plot(x + o, iou, "o", color=c, ms=8, mec="white", label=name)
    axes[0].set_title("% of masks changed from the previous step", loc="left")
    axes[1].set_title("median IoU with the previous step's mask (changed masks only)", loc="left")
    for ax in axes:
        stage_axis(ax, stages)
        ax.legend(fontsize=8)
    fig.suptitle(f"{rec} - what every pipeline step changes", fontsize=10, x=0.01, ha="left")
    return finish(fig, path)


def plot_bad_frames_timeline(df, rec, fps, path=None):
    """Frames where a fish fails a QC rule other than touching, one row per (stage, fish); top row = fish touching."""
    stages = list(dict.fromkeys(df.stage))
    fish = list(dict.fromkeys(df.fish))
    col = fish_colors(fish)
    fig, ax = plt.subplots(figsize=(13, 0.32 * (len(stages) + 1) * len(fish) + 1.2))
    y, ticks = 0, []
    touch = df[df.stage == stages[-1]].groupby("frame").contact.apply(lambda v: v.astype(bool).any())
    t = touch.index[touch.to_numpy()] / fps
    ax.plot(t, np.full(t.size, y), "|", color=mm.MUTED, ms=8, mew=0.6, rasterized=True)
    ticks.append((y, "any fish touching", f"{100 * touch.mean():.1f}%"))
    y += 1.5
    for st in stages:
        for f in fish:
            q = df[(df.stage == st) & (df.fish == f)]
            bad = ~q.all_but_contact.to_numpy(bool)
            t = q.frame.to_numpy()[bad] / fps
            ax.plot(t, np.full(t.size, y), "|", color=col[f], ms=7, mew=0.6, rasterized=True)
            ticks.append((y, f"{st}  {f}", f"{100 * bad.mean():.1f}%"))
            y += 1
        y += 0.6
    ax.set_yticks([v for v, _, _ in ticks], [n for _, n, _ in ticks], fontsize=7)
    for v, _, p in ticks:
        ax.text(1.005, v, p, transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=mm.INK)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, df.frame.max() / fps)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{rec} - frames failing a mask QC rule other than touching (ticks), per step and fish; "
                 "right = % of frames", loc="left")
    return finish(fig, path)


def plot_distributions_by_stage(df, rec, path=None):
    """Per-frame metric distributions at every step (box = IQR, whiskers = 1-99 percentile), pooled over fish."""
    stages = list(dict.fromkeys(df.stage))
    specs = [("iou_prev_aligned", "temporal IoU, centroid-aligned"),
             ("area_ratio", "area / fish's median area"),
             ("length_ratio", "length / fish's median length"),
             ("solidity", "solidity (area / convex hull)")]
    fig, axes = plt.subplots(1, len(specs), figsize=(15, 3.8))
    for ax, (c, name) in zip(axes, specs):
        data = [df.loc[(df.stage == st) & df[c].notna(), c].to_numpy() for st in stages]
        ax.boxplot(data, positions=range(len(stages)), whis=(1, 99), widths=0.55, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#dbe8f7", edgecolor=mm.PALETTE[3]),
                   medianprops=dict(color=mm.INK), whiskerprops=dict(color=mm.MUTED), capprops=dict(color=mm.MUTED))
        ax.set_title(name, loc="left", fontsize=9)
        stage_axis(ax, stages)
    fig.suptitle(f"{rec} - per-mask metric distributions at every step (box = IQR, whiskers = 1st-99th percentile, "
                 "all fish)", fontsize=10, x=0.01, ha="left")
    return finish(fig, path)


def plot_frame_through_stages(masks, df, k, fish, image=None, title="", path=None, pad=20):
    """Frame k at every step, in a square window around `fish` (letter) and the fish its mask comes near at any step: the
    video crop (first panel, if `image` is given) and each step's visible masks of every fish in the window (fill +
    outline, palette order A, B, C, ...).
    masks: {stage: Masks} from io_utils.load_stage_masks; df: stage_quality rows (with stage_checks columns)."""
    stages = list(masks)
    names = masks[stages[0]].names
    fi = names.index(fish)
    crops = {st: m.crops(k) for st, m in masks.items()}
    near = lambda a, b: not (a[1] + pad < b[0] or b[1] + pad < a[0] or a[3] + pad < b[2] or b[3] + pad < a[2])
    own = [c[fi][0] for c in crops.values() if fi in c]
    shown = sorted({f for c in crops.values() for f, (b, _) in c.items() if any(near(b, o) for o in own)})
    boxes = [c[f][0] for c in crops.values() for f in shown if f in c]
    H, W = masks[stages[0]].shape
    y0, y1 = min(b[0] for b in boxes) - pad, max(b[1] for b in boxes) + pad
    x0, x1 = min(b[2] for b in boxes) - pad, max(b[3] for b in boxes) + pad
    side = min(max(y1 - y0, x1 - x0), H, W)                 # square window around the fish, inside the frame
    y0 = int(np.clip((y0 + y1 - side) // 2, 0, H - side)); y1 = y0 + side
    x0 = int(np.clip((x0 + x1 - side) // 2, 0, W - side)); x1 = x0 + side
    win = (y0, y1, x0, x1)                                   # draw every fish whose mask reaches into the window
    shown = sorted({f for c in crops.values() for f, (b, _) in c.items()
                    if not (b[1] <= win[0] or win[1] <= b[0] or b[3] <= win[2] or win[3] <= b[2])})
    bg =(np.repeat(image[y0:y1, x0:x1, None] / 255, 3, axis=2) if image is not None
          else np.ones((y1 - y0, x1 - x0, 3)))
    col = [np.array(plt.matplotlib.colors.to_rgb(c)) for c in mm.PALETTE]
    panels = ([("video", None)] if image is not None else []) + [(st, crops[st]) for st in stages]
    nc = 6 if image is not None else 5
    nr = -(-len(panels) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 3.7 * nr + 0.7))
    rows = df[df.frame == k].set_index(["stage", "fish"])
    for ax, (st, c) in zip(axes.flat, panels):
        img = bg.copy()
        sub = {}
        for f in shown if c is not None else []:
            if f not in c:
                continue
            (b, m), s = c[f], np.zeros((y1 - y0, x1 - x0), bool)
            ys, xs = np.nonzero(m)
            ys, xs = ys + b[0] - y0, xs + b[2] - x0
            ok = (ys >= 0) & (ys < s.shape[0]) & (xs >= 0) & (xs < s.shape[1])
            s[ys[ok], xs[ok]] = True
            img[s] = 0.5 * img[s] + 0.5 * col[f]
            sub[f] = s
        ax.imshow(img, interpolation="nearest")
        for f, s in sub.items():
            ax.contour(s, levels=[0.5], colors=[col[f]], linewidths=0.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if st == "video":
            ax.set_title("video", loc="left", fontsize=9)
            continue
        lab = mm.stage_label(st)
        r = rows.loc[lab] if lab in rows.index.get_level_values(0) else None
        ch = r.loc[fish, "iou_prev_stage"] if r is not None and fish in r.index else np.nan
        changed = np.isfinite(ch) and ch < 1
        ax.set_title(lab + (f"   changed, IoU {ch:.3f}" if changed else ""), loc="left", fontsize=9,
                     color=mm.INK if changed else mm.MUTED, fontweight="bold" if changed else "normal")
        for sp in ax.spines.values():
            sp.set_visible(changed); sp.set_color(mm.INK); sp.set_linewidth(1.5)
        if r is not None:
            ax.set_xlabel("\n".join(
                f"{names[f]}: {int(r.loc[names[f], 'n_comp']) if r.loc[names[f], 'present'] else 0} piece(s), "
                f"IoU final {r.loc[names[f], 'iou_final']:.2f}" for f in shown if names[f] in r.index),
                fontsize=7, loc="left")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{title}frame {k}, zoomed on fish {fish} (fill colours: " +
                 ", ".join(f"{names[f]} = {n}" for f, n in zip(shown, [("pink", "green", "yellow", "blue")[f]
                                                                      for f in shown])) +
                 "); bold frame = the mask of fish " + fish + " changed at that step", x=0.01, ha="left", fontsize=10)
    return finish(fig, path)


def video_path(run):
    """The run's video next to its masks folder (data/<group>/<recording>.avi for data/<group>/masks/<recording>), or None."""
    run = os.path.normpath(run)
    p = os.path.join(os.path.dirname(os.path.dirname(run)), os.path.basename(run) + ".avi")
    return p if os.path.exists(p) else None


def write_examples(run, df, n=50, min_gap=25, out_root=os.path.join("results", "Kaden_mask_quality")):
    """The n frames the pipeline changes most (mask_metrics.most_changed_frames), each as a panel of every step, in
    results/Kaden_mask_quality/<recording>/ with examples.csv (rank, frame, time, fish, change score, IoU first-final)."""
    from io_utils import load_stage_masks, load_video_frames
    rec = os.path.basename(os.path.normpath(run))
    out = os.path.join(out_root, rec)
    os.makedirs(out, exist_ok=True)
    ex = mm.most_changed_frames(df, n, min_gap)
    fps = run_fps(run)
    ex.insert(2, "time_s", ex.frame / fps)
    ex.to_csv(os.path.join(out, "examples.csv"), index=False)
    masks = {st: load_stage_masks(run, st) for st in stages_done(run)}
    vid = video_path(run)
    imgs = load_video_frames(vid, ex.frame) if vid else {}
    for r in ex.itertuples(index=False):
        name = f"rank{r.rank:02d}_frame{r.frame:05d}_fish{r.fish}.png"
        plt.close(plot_frame_through_stages(
            masks, df, r.frame, r.fish, imgs.get(r.frame),
            title=f"{rec} - #{r.rank}: {r.time_s:.2f} s, change score {r.change:.2f}, "
                  f"IoU first-final {r.iou_first_final:.2f}\n", path=os.path.join(out, name)))
    print(f"{rec}: {len(ex)} examples in {out}" + ("" if vid else " (no video found: masks on white)"), flush=True)
    return ex, out


PLOTS = {"01_qc_rules_by_stage": lambda df, s, c, rec, fps, p: plot_qc_heatmap(s, c, rec, p),
         "02_metrics_by_stage": lambda df, s, c, rec, fps, p: plot_metrics_by_stage(s, c, rec, p),
         "03_stage_changes": lambda df, s, c, rec, fps, p: plot_stage_changes(df, rec, p),
         "04_bad_frames_timeline": lambda df, s, c, rec, fps, p: plot_bad_frames_timeline(df, rec, fps, p),
         "05_distributions_by_stage": lambda df, s, c, rec, fps, p: plot_distributions_by_stage(df, rec, p)}


def make_plots(df, s, checks, rec, fps, out):
    for name, plot in PLOTS.items():
        plt.close(plot(df, s, checks, rec, fps, os.path.join(out, f"{name}.png")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders (with outputs/stages/)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--examples", type=int, default=0, help="also write this many most-changed frames as panels of "
                    "every step to results/Kaden_mask_quality/<recording>/")
    ap.add_argument("--min-gap", type=int, default=25, help="examples at least this many frames apart")
    a = ap.parse_args()
    for run in a.runs:
        df, s, checks, out = load_or_compute(run, a.recompute, a.workers)
        rec = os.path.basename(os.path.normpath(run))
        make_plots(df, s, checks, rec, run_fps(run), out)
        t, _ = load_or_compute_changes(run, df, a.recompute, a.workers)
        plt.close(plot_step_changes(t, df, rec, os.path.join(out, "06_step_changes.png")))
        gross, net = mm.net_vs_gross_change(df)
        with pd.option_context("display.width", 250, "display.max_columns", 50):
            print(t.round(3).to_string(index=False))
        print(f"mean summed change per fish-frame {gross:.3f}, net (first vs final) {net:.3f}, net / gross {net / gross:.2f}")
        if a.examples:
            write_examples(run, df, a.examples, a.min_gap)


if __name__ == "__main__":
    main()
