"""Per-frame metrics and inspection plots for fish tracking masks.

Two tracking sources are supported (--source):

  zeus   data/*tracking_final.pkl (ZEUS): packed per-fish crops, identities
         1 largest, 2 second, 3 third, 0 smallest (4 = remnant).
  kaden  data/track_Kaden/results_*/<recording>/outputs/<recording>.pkl
         (efish4-track): COCO RLE masks for fish A-D, both the visible and the
         full-body (amodal) mask, plus occlusion / inference / review flags,
         confidence and a 4-point midline.

For every recording it computes one row per (frame, fish) with area, centroid,
shape and fragmentation metrics (plus the source's own flags), caches the
table, and writes inspection figures to <out>/<recording>/ plus a
cross-recording summary in <out>/, where <out> is figures/mask_metrics/ for
ZEUS and figures/mask_metrics/kaden[_amodal]/ for Kaden.

Needs numpy >= 2 to unpickle (e.g. the `electric_fish` conda env):
    ~/opt/anaconda3/envs/electric_fish/bin/python mask_metrics.py [--recompute]
    ~/opt/anaconda3/envs/electric_fish/bin/python mask_metrics.py --source kaden \\
        data/track_Kaden/results_12_32_58_full
Paths may be files or folders; without paths every recording of the source is
processed. Rerunning a subset updates only those rows of summary.csv.
"""

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

DATA_DIR = "data"
OUT_DIR = os.path.join("figures", "mask_metrics")

# CVD-validated palette shared by both sources. For ZEUS the identity order and
# colours follow the review reels' colour key (1 pink, 2 green, 3 yellow, 0 blue);
# Kaden's A-D identities are independent of ZEUS's and just take the palette in order.
PALETTE = ["#e87ba4", "#008300", "#eda100", "#2a78d6"]
INK = "#2b2b2b"
MUTED = "#8a8a85"
GRID = "#e6e6e3"

SMOOTH_S = 1.0          # rolling-median window for time series (s)
# Defaults; overridden per run by --area-jump / --speed-jump (set in process()/main())
AREA_JUMP = 0.30        # frame-to-frame relative area change flagged as a jump
SPEED_JUMP = 1500.0     # centroid speed (px/s) flagged as an implausible jump

STRUCT8 = np.ones((3, 3), bool)

# Per-fish boolean columns some sources provide, shown as extra quality-flag rows
EXTRA_FLAGS = [
    ("occluded", "flagged occluded by the tracker"),
    ("inferred", "position inferred (not detected)"),
    ("needs_review", "flagged for review by the tracker"),
    ("midline_held", "midline held from an earlier frame"),
]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})


# --------------------------------------------------------------------------- sources

def unpack_crop(packed, box):
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    return np.unpackbits(packed)[: h * w].reshape(h, w).astype(bool)


def rle_counts(s):
    """Run lengths of a compressed COCO RLE string (port of pycocotools rleFrString)."""
    if isinstance(s, str):
        s = s.encode()
    cnts, p = [], 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_crop(rle):
    """Bounding box (y0, y1, x0, x1) and boolean crop of a COCO RLE mask."""
    h = rle["size"][0]
    cnts = np.asarray(rle_counts(rle["counts"]), np.int64)
    starts = np.cumsum(cnts) - cnts
    on_s, on_l = starts[1::2], cnts[1::2]           # runs alternate 0s, 1s
    keep = on_l > 0
    on_s, on_l = on_s[keep], on_l[keep]
    if on_l.size == 0:
        return None, None
    offs = np.cumsum(on_l) - on_l
    idx = np.repeat(on_s - offs, on_l) + np.arange(on_l.sum())
    xs, ys = np.divmod(idx, h)                      # column-major order
    y0, x0 = ys.min(), xs.min()
    box = (int(y0), int(ys.max()) + 1, int(x0), int(xs.max()) + 1)
    crop = np.zeros((box[1] - y0, box[3] - x0), bool)
    crop[ys - y0, xs - x0] = True
    return box, crop


def load_zeus(path, mask_kind=None):
    """Frames of a ZEUS tracking_final.pkl as generic per-frame records."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    frames = d["frames"]

    def gen():
        for k in sorted(frames):
            fr = frames[k]
            crops = [unpack_crop(p, b) for p, b in zip(fr["masks"], fr["boxes"])]
            yield dict(frame=k, labels=list(fr["labels"]), boxes=list(fr["boxes"]),
                       crops=crops, areas=list(fr["areas"]),
                       obj_ids=list(fr["obj_ids"]), extra=[{}] * len(crops),
                       frame_extra=dict(n_remnant=sum(l == 4 for l in fr["labels"])))

    meta = dict(fps=d["fps"], shape=d["shape"], n_frames=len(frames), video=d["video"])
    return gen(), meta


def load_kaden(path, mask_kind="visible"):
    """Frames of an efish4-track output pickle as generic per-frame records."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    masks = d[f"mask_{mask_kind}"]
    ml = d["midline"]                                   # (n, fish, 4 points, xy)
    midline_len = np.linalg.norm(np.diff(ml, axis=2), axis=-1).sum(-1)  # NaN if any point missing

    def amodal_overlap(k):
        """Pixels each fish's full-body mask shares with any other fish's (dict i -> px).

        Visible masks are disjoint by construction, so shared pixels only show
        up in the amodal masks.
        """
        labs, boxes, crops = [], [], []
        for i, rle in enumerate(d["mask_amodal"][k]):
            box, crop = rle_crop(rle) if rle is not None else (None, None)
            if box is not None:
                labs.append(i)
                boxes.append(box)
                crops.append(crop)
        ov = dict.fromkeys(labs, 0)
        for (la, lb), n in pair_overlap(labs, boxes, crops).items():
            ov[la] += n
            ov[lb] += n
        return ov

    def gen():
        for k in range(d["n_frames"]):
            rec = dict(frame=k, labels=[], boxes=[], crops=[], areas=[], obj_ids=[],
                       extra=[], frame_extra={})
            ov_amodal = amodal_overlap(k)
            for i, rle in enumerate(masks[k]):
                if rle is None:
                    continue
                box, crop = rle_crop(rle)
                if box is None:
                    continue
                rec["labels"].append(i)
                rec["boxes"].append(box)
                rec["crops"].append(crop)
                rec["areas"].append(int(d[f"area_{mask_kind}"][k, i]))
                rec["obj_ids"].append(i)
                rec["extra"].append(dict(
                    area_visible=int(d["area_visible"][k, i]),
                    area_amodal=int(d["area_amodal"][k, i]),
                    occluded=bool(d["occluded"][k, i]),
                    inferred=bool(d["inferred"][k, i]),
                    needs_review=bool(d["needs_review"][k, i]),
                    confidence=float(d["confidence"][k, i]),
                    midline_len=float(midline_len[k, i]),
                    midline_held=bool(d["midline_source"][k, i] == 2),
                    overlap_px_amodal=ov_amodal.get(i, 0),
                ))
            yield rec

    meta = dict(fps=d["fps"], shape=tuple(d["image_size"]), n_frames=d["n_frames"],
                video=d["input_video"].get("path", d["recording"]))
    return gen(), meta


def get_source(name, mask_kind="visible"):
    if name == "zeus":
        return dict(
            name="zeus", load=load_zeus, mask_kind=None,
            fish=[1, 2, 3, 0],
            fish_name={1: "1 largest", 2: "2 second", 3: "3 third", 0: "0 smallest"},
            fish_color=dict(zip([1, 2, 3, 0], PALETTE)),
            pattern=os.path.join(DATA_DIR, "*tracking_final.pkl"),
            dir_pattern="*tracking_final.pkl",
            rec_name=lambda p: os.path.basename(p).replace("_tracking_final.pkl", ""),
            out=OUT_DIR,
            tag="",
        )
    if name == "kaden":
        return dict(
            name="kaden", load=load_kaden, mask_kind=mask_kind,
            fish=[0, 1, 2, 3],
            fish_name={i: f"fish {c}" for i, c in enumerate("ABCD")},
            fish_color=dict(zip(range(4), PALETTE)),
            pattern=os.path.join(DATA_DIR, "track_Kaden", "results_*", "*", "outputs",
                                 "*.pkl"),
            dir_pattern=os.path.join("**", "outputs", "*.pkl"),
            rec_name=lambda p: os.path.splitext(os.path.basename(p))[0],
            out=os.path.join(OUT_DIR, "kaden" if mask_kind == "visible"
                             else f"kaden_{mask_kind}"),
            tag=f" (Kaden, {mask_kind} masks)",
        )
    raise ValueError(f"unknown source {name!r}")


# --------------------------------------------------------------------------- metrics

def shape_metrics(crop, box):
    """Centroid (frame coords), principal axis lengths, orientation, components."""
    ys, xs = np.nonzero(crop)
    n = ys.size
    if n == 0:
        return dict(cy=np.nan, cx=np.nan, major=np.nan, minor=np.nan,
                    orient=np.nan, n_comp=0, largest_frac=np.nan, px=0)
    cy, cx = ys.mean(), xs.mean()
    dy, dx = ys - cy, xs - cx
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)],
                    [np.mean(dx * dy), np.mean(dy * dy)]])
    evals, evecs = np.linalg.eigh(cov)
    # Full axis length of the equivalent ellipse = 4 * sqrt(eigenvalue)
    minor, major = 4 * np.sqrt(np.maximum(evals, 0))
    vx, vy = evecs[:, 1]
    orient = np.degrees(np.arctan2(vy, vx)) % 180
    lab, n_comp = ndimage.label(crop, structure=STRUCT8)
    if n_comp > 1:
        sizes = np.bincount(lab.ravel())[1:]
        largest_frac = sizes.max() / n
    else:
        largest_frac = 1.0
    return dict(cy=cy + box[0], cx=cx + box[2], major=major, minor=minor,
                orient=orient, n_comp=n_comp, largest_frac=largest_frac, px=n)


def pair_overlap(labs, boxes, crops):
    """Pixels shared by each pair of fish masks in one frame (0 when disjoint)."""
    out = {}
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = boxes[i], boxes[j]
            y0, y1 = max(a[0], b[0]), min(a[1], b[1])
            x0, x1 = max(a[2], b[2]), min(a[3], b[3])
            if y0 >= y1 or x0 >= x1:
                continue
            ca = crops[i][y0 - a[0]:y1 - a[0], x0 - a[2]:x1 - a[2]]
            cb = crops[j][y0 - b[0]:y1 - b[0], x0 - b[2]:x1 - b[2]]
            inter = int(np.count_nonzero(ca & cb))
            if inter:
                out[(labs[i], labs[j])] = inter
    return out


def compute_metrics(path, src):
    frames, meta = src["load"](path, src["mask_kind"])
    rows, extras = [], []
    for fr in frames:
        labs = fr["labels"]
        overlaps = pair_overlap(labs, fr["boxes"], fr["crops"])
        ov_px = {lab: 0 for lab in labs}
        for (la, lb), n in overlaps.items():
            ov_px[la] += n
            ov_px[lb] += n
        counts = pd.Series(labs).value_counts()
        for lab, box, crop, area, oid, ex in zip(labs, fr["boxes"], fr["crops"],
                                                 fr["areas"], fr["obj_ids"], fr["extra"]):
            m = shape_metrics(crop, box)
            rows.append(dict(frame=fr["frame"], label=lab, obj_id=oid, area=area,
                             box_h=box[1] - box[0], box_w=box[3] - box[2],
                             overlap_px=ov_px[lab], n_same_label=int(counts[lab]),
                             **m, **ex))
        extras.append(dict(frame=fr["frame"], n_masks=len(labs), **fr["frame_extra"]))
    return pd.DataFrame(rows), pd.DataFrame(extras), meta


def load_or_compute(path, src, recompute=False):
    rec = src["rec_name"](path)
    cache_dir = os.path.join(src["out"], rec)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "metrics.pkl")
    if os.path.exists(cache) and not recompute:
        with open(cache, "rb") as f:
            return pickle.load(f)
    out = compute_metrics(path, src)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def per_fish_series(df, n_frames, fps, fish):
    """Dict label -> DataFrame indexed by every frame (NaN where untracked).

    Frames with duplicate rows for one fish keep the largest mask.
    """
    out = {}
    full = pd.RangeIndex(n_frames, name="frame")
    for lab in fish:
        s = (df[df.label == lab].sort_values("area")
             .drop_duplicates("frame", keep="last").set_index("frame")
             .reindex(full))
        step = np.hypot(s.cx.diff(), s.cy.diff())
        s["speed"] = step * fps                       # px/s, NaN across gaps
        s["area_change"] = s.area.pct_change(fill_method=None)
        # Exactly the same pixel count as the previous frame while the mask moves:
        # a real segmentation almost never repeats its area, so runs of this
        # mark masks that were filled in / painted at a fixed size.
        s["const_area"] = (s.area.diff() == 0) & (s.speed > 0)
        s["elong"] = s.major / s.minor
        s["present"] = s.area.notna()
        if "area_amodal" in s:
            s["vis_frac"] = s.area_visible / s.area_amodal
        out[lab] = s
    return out


# --------------------------------------------------------------------------- plotting

def smooth(s, fps):
    w = max(int(SMOOTH_S * fps), 1)
    return s.rolling(w, center=True, min_periods=max(w // 4, 1)).median()


def fish_label(ax, lab, src):
    ax.text(0.005, 0.95, src["fish_name"][lab], transform=ax.transAxes, va="top",
            ha="left", fontsize=9, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=src["fish_color"][lab], lw=1.5))


def timeseries_grid(series, src, col, ylabel, title, fps, path, ylim=None, logy=False,
                    raw=True):
    fish = src["fish"]
    fig, axes = plt.subplots(len(fish), 1, figsize=(12, 7), sharex=True, sharey=True)
    for ax, lab in zip(axes, fish):
        s = series[lab]
        t = s.index / fps / 60
        if raw:
            ax.plot(t, s[col], color=src["fish_color"][lab], lw=0.4, alpha=0.35,
                    rasterized=True)
        ax.plot(t, smooth(s[col], fps), color=src["fish_color"][lab], lw=1.2,
                rasterized=True)
        miss = ~s.present.values
        if miss.any():
            ax.plot(t[miss], np.zeros(miss.sum()), "|", color=INK, ms=6,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        fish_label(ax, lab, src)
        if logy:
            ax.set_yscale("log")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[-1].set_xlabel("time (min)")
    fig.supylabel(ylabel, fontsize=9)
    fig.suptitle(f"{title}\nthin = per frame, thick = {SMOOTH_S:g} s rolling median, "
                 "ticks on the x-axis = frames where the fish is untracked",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distributions(series, src, title, path):
    fish = src["fish"]
    specs = [("area", "area (px)"), ("major", "length, major axis (px)"),
             ("elong", "elongation (major / minor)"), ("speed", "speed (px/s)")]
    fig, axes = plt.subplots(1, len(specs), figsize=(14, 3.4))
    for ax, (col, xl) in zip(axes, specs):
        allv = pd.concat([series[l][col] for l in fish]).dropna()
        lo, hi = np.nanpercentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi, 80)
        for lab in fish:
            v = series[lab][col].dropna()
            ax.hist(v, bins=bins, histtype="step", lw=1.5, color=src["fish_color"][lab],
                    density=True, label=src["fish_name"][lab])
        ax.set_xlabel(xl)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0].set_ylabel("density")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title} - metric distributions per fish (0.5-99.5 percentile range)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def quality_problems(series):
    problems = [
        ("untracked", lambda s: ~s.present),
        ("fragmented (>1 component)", lambda s: s.n_comp > 1),
        ("overlaps another fish", lambda s: s.overlap_px > 0),
        *([("full-body (amodal) mask overlaps another fish",
            lambda s: s.overlap_px_amodal > 0)]
          if "overlap_px_amodal" in next(iter(series.values())) else []),
        (f"area jump >{AREA_JUMP:.0%}", lambda s: s.area_change.abs() > AREA_JUMP),
        (f"centroid jump >{SPEED_JUMP:g} px/s", lambda s: s.speed > SPEED_JUMP),
        ("constant area (same px count as previous frame)", lambda s: s.const_area),
        ("duplicate label in frame", lambda s: s.n_same_label > 1),
    ]
    some = next(iter(series.values()))
    for col, name in EXTRA_FLAGS:
        if col in some:
            problems.append((name, lambda s, c=col: s[c].eq(True)))
    return problems


def plot_quality(series, extras, src, fps, title, path):
    """Raster of tracking problems over time, one row per fish per problem."""
    fish = src["fish"]
    problems = quality_problems(series)
    fig, axes = plt.subplots(len(problems) + 1, 1,
                             figsize=(12, 1.4 * len(problems) + 0.7), sharex=True,
                             gridspec_kw=dict(height_ratios=[1] * len(problems) + [0.8]))
    for ax, (name, fn) in zip(axes, problems):
        for i, lab in enumerate(fish):
            s = series[lab]
            mask = fn(s).fillna(False).values.astype(bool)
            t = s.index.values[mask] / fps / 60
            ax.plot(t, np.full(t.size, i), "|", color=src["fish_color"][lab], ms=7,
                    mew=0.8, rasterized=True)
            pct = 100 * mask.mean()
            ax.text(1.005, i, f"{pct:.2f}%", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7, color=INK)
        ax.set_yticks(range(len(fish)), [src["fish_name"][l] for l in fish], fontsize=7)
        ax.set_ylim(-0.7, len(fish) - 0.3)
        ax.invert_yaxis()
        ax.set_title(name, loc="left", fontsize=9)
        ax.grid(axis="y", visible=False)
    ax = axes[-1]
    t = extras.frame / fps / 60
    ax.plot(t, extras.n_masks, color=INK, lw=0.6, rasterized=True)
    lo, hi = int(extras.n_masks.min()), int(extras.n_masks.max())
    ax.set_yticks(range(min(lo, len(fish) - 2), max(hi, len(fish) + 1) + 1))
    ax.set_ylabel("masks\nin frame", fontsize=8)
    what = "incl. remnants" if "n_remnant" in extras else "non-empty"
    ax.set_title(f"number of masks per frame ({what})", loc="left", fontsize=9)
    ax.set_xlabel("time (min)")
    fig.suptitle(f"{title} - tracking quality flags (right: % of frames)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_occupancy(series, src, shape, title, path):
    fish = src["fish"]
    fig, axes = plt.subplots(1, len(fish), figsize=(14, 3.9))
    bins = [np.linspace(0, shape[1], 92), np.linspace(0, shape[0], 92)]
    for ax, lab in zip(axes, fish):
        s = series[lab].dropna(subset=["cx"])
        h, _, _ = np.histogram2d(s.cx, s.cy, bins=bins)
        ax.imshow(np.log1p(h.T), origin="upper", cmap="Blues",
                  extent=[0, shape[1], shape[0], 0])
        ax.set_title(src["fish_name"][lab], color=INK)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(src["fish_color"][lab])
            sp.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle(f"{title} - centroid occupancy (log counts, image coordinates)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_area_vs_length(series, src, title, path):
    fish = src["fish"]
    fig, axes = plt.subplots(1, len(fish), figsize=(14, 3.6), sharex=True, sharey=True)
    for ax, lab in zip(axes, fish):
        s = series[lab].dropna(subset=["area", "major"])
        ax.hexbin(s.major, s.area, gridsize=60, bins="log", cmap="Blues", mincnt=1)
        ax.set_title(src["fish_name"][lab])
        ax.set_xlabel("length, major axis (px)")
    axes[0].set_ylabel("area (px)")
    fig.suptitle(f"{title} - area vs length (outliers off the main band suggest "
                 "merged / truncated masks)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise(rec, series, src):
    out = []
    for lab in src["fish"]:
        s = series[lab]
        row = dict(
            recording=rec, label=lab,
            area_median=s.area.median(),
            area_q25=s.area.quantile(0.25), area_q75=s.area.quantile(0.75),
            area_cv=s.area.std() / s.area.mean(),
            length_median=s.major.median(),
            speed_median=s.speed.median(),
            pct_untracked=100 * (~s.present).mean(),
            pct_tracked=100 * s.present.mean(),
            pct_fragmented=100 * (s.n_comp > 1).mean(),
            pct_overlap=100 * (s.overlap_px > 0).mean(),
            pct_area_jump=100 * (s.area_change.abs() > AREA_JUMP).mean(),
            pct_centroid_jump=100 * (s.speed > SPEED_JUMP).mean(),
            pct_const_area=100 * s.const_area.mean(),
        )
        if "overlap_px_amodal" in s:
            row["pct_overlap_amodal"] = 100 * (s.overlap_px_amodal > 0).mean()
        if "vis_frac" in s:
            row["vis_frac_median"] = s.vis_frac.median()
        if "midline_len" in s:
            row["midline_len_median"] = s.midline_len.median()
        for col, _ in EXTRA_FLAGS:
            if col in s:
                row[f"pct_{col}"] = 100 * s[col].eq(True).mean()
        out.append(row)
    return out


def plot_summary(summ, src, path):
    fish = src["fish"]
    recs = list(dict.fromkeys(summ.recording))
    specs = [("area_median", "median area (px)\nbars = IQR"),
             ("area_cv", "area coefficient of variation"),
             ("pct_untracked", "% frames untracked"),
             ("pct_fragmented", "% frames fragmented"),
             ("pct_overlap", "% frames overlapping\nanother fish"),
             ("pct_centroid_jump", f"% frames with centroid\njump >{SPEED_JUMP:g} px/s"),
             ("pct_const_area", "% frames with constant\narea (painted?)")]
    if "pct_occluded" in summ:
        specs.append(("pct_occluded", "% frames flagged\noccluded"))
    fig, axes = plt.subplots(1, len(specs), figsize=(2.6 * len(specs),
                                                     0.5 * len(recs) + 2.2),
                             sharey=True)
    y = np.arange(len(recs))
    off = np.linspace(-0.27, 0.27, len(fish))
    for ax, (col, xl) in zip(axes, specs):
        for o, lab in zip(off, fish):
            s = summ[summ.label == lab].set_index("recording").loc[recs]
            if col == "area_median":
                ax.hlines(y + o, s.area_q25, s.area_q75, color=src["fish_color"][lab],
                          lw=2)
            ax.plot(s[col], y + o, "o", ms=6, color=src["fish_color"][lab],
                    mec="white", mew=1, label=src["fish_name"][lab])
        ax.set_title(xl, fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, recs)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(f"Mask metrics across recordings{src['tag']}",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- report

def report_metrics(has_amodal):
    """(key, title, per-frame flag fn) for the report; % of all frames in the video."""
    out = [
        ("untracked", "% frames untracked", lambda s: ~s.present),
        ("tracked", "% frames tracked", lambda s: s.present),
        ("fragmented", "% frames with a fragmented mask (>1 connected component)",
         lambda s: s.n_comp > 1),
        ("area_jump", f"% frames with an area jump >{AREA_JUMP:.0%} from the previous frame",
         lambda s: s.area_change.abs() > AREA_JUMP),
        ("centroid_jump", f"% frames with a centroid jump >{SPEED_JUMP:g} px/s",
         lambda s: s.speed > SPEED_JUMP),
        ("overlap", "% frames in which a pixel belongs to this and another fish's mask",
         lambda s: s.overlap_px > 0),
    ]
    if has_amodal:
        out.append(("overlap_amodal",
                    "% frames in which a pixel belongs to this and another fish's "
                    "full-body (amodal) mask", lambda s: s.overlap_px_amodal > 0))
    return out


def write_report(series, src, fps, rec, title, out):
    """Per-frame flags (csv.gz), per-fish percentages (csv) and one raster per metric."""
    os.makedirs(out, exist_ok=True)
    metrics = report_metrics("overlap_px_amodal" in series[src["fish"][0]])
    flags = []
    for lab in src["fish"]:
        s = series[lab]
        f = pd.DataFrame({k: fn(s).fillna(False).astype(bool) for k, _, fn in metrics})
        f.insert(0, "fish", src["fish_name"][lab])
        flags.append(f.reset_index())
    flags = pd.concat(flags, ignore_index=True)
    flags.to_csv(os.path.join(out, "flags_per_frame.csv.gz"), index=False)
    pct = (flags.groupby("fish", sort=False)[[k for k, _, _ in metrics]].mean() * 100)
    pct.columns = [f"pct_{c}" for c in pct.columns]
    pct = pct.reset_index()
    pct.insert(0, "recording", rec)
    pct.to_csv(os.path.join(out, "metrics.csv"), index=False)

    fish = src["fish"]
    for key, name, fn in metrics:
        fig, ax = plt.subplots(figsize=(12, 2.2))
        for i, lab in enumerate(fish):
            s = series[lab]
            mask = fn(s).fillna(False).values.astype(bool)
            t = s.index.values[mask] / fps / 60
            ax.plot(t, np.full(t.size, i), "|", color=src["fish_color"][lab], ms=9,
                    mew=0.8, rasterized=True)
            ax.text(1.005, i, f"{100 * mask.mean():.2f}%",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                    color=INK)
        ax.set_yticks(range(len(fish)), [src["fish_name"][l] for l in fish])
        ax.set_ylim(-0.7, len(fish) - 0.3)
        ax.invert_yaxis()
        ax.set_xlim(0, len(next(iter(series.values()))) / fps / 60)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("time (min)")
        ax.set_title(f"{title}\n{name} (ticks = flagged frames, right = % of frames)",
                     loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{key}.png"), dpi=150)
        plt.close(fig)
    return pct


def plot_report_summary(table, src, out):
    """One across-recordings dot plot per report metric."""
    fish_names = [src["fish_name"][l] for l in src["fish"]]
    recs = list(dict.fromkeys(table.recording))
    y = np.arange(len(recs))
    off = np.linspace(-0.27, 0.27, len(fish_names))
    for key, name, _ in report_metrics("pct_overlap_amodal" in table):
        col = f"pct_{key}"
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(recs) + 1.4))
        for o, lab, fn in zip(off, src["fish"], fish_names):
            s = table[table.fish == fn].set_index("recording").reindex(recs)
            ax.plot(s[col], y + o, "o", ms=6, color=src["fish_color"][lab],
                    mec="white", mew=1, label=fn)
        ax.set_yticks(y, recs)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of frames")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.suptitle(f"{name}\nacross recordings{src['tag']}", fontsize=10,
                     x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{key}.png"), dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- main

def process(path, source="zeus", mask_kind="visible", recompute=False,
            area_jump=AREA_JUMP, speed_jump=SPEED_JUMP):
    global AREA_JUMP, SPEED_JUMP
    AREA_JUMP, SPEED_JUMP = area_jump, speed_jump      # this worker's thresholds
    src = get_source(source, mask_kind)
    rec = src["rec_name"](path)
    title = rec + src["tag"]
    df, extras, meta = load_or_compute(path, src, recompute)
    fps, n = meta["fps"], meta["n_frames"]
    series = per_fish_series(df, n, fps, src["fish"])
    out = os.path.join(src["out"], rec)
    timeseries_grid(series, src, "area", "mask area (px)", f"{title} - mask area over time",
                    fps, os.path.join(out, "01_area.png"))
    timeseries_grid(series, src, "major", "major-axis length (px)",
                    f"{title} - mask length over time", fps,
                    os.path.join(out, "02_length.png"))
    timeseries_grid(series, src, "elong", "major / minor axis",
                    f"{title} - mask elongation over time", fps,
                    os.path.join(out, "03_elongation.png"))
    hi = np.nanpercentile(pd.concat([series[l].speed for l in src["fish"]]), 99.5)
    timeseries_grid(series, src, "speed", "centroid speed (px/s)",
                    f"{title} - centroid speed over time", fps,
                    os.path.join(out, "04_speed.png"), ylim=(0, hi))
    timeseries_grid(series, src, "n_comp", "connected components",
                    f"{title} - mask fragmentation over time", fps,
                    os.path.join(out, "05_components.png"))
    plot_quality(series, extras, src, fps, title, os.path.join(out, "06_quality_flags.png"))
    plot_distributions(series, src, title, os.path.join(out, "07_distributions.png"))
    plot_area_vs_length(series, src, title, os.path.join(out, "08_area_vs_length.png"))
    plot_occupancy(series, src, meta["shape"], title, os.path.join(out, "09_occupancy.png"))
    some = series[src["fish"][0]]
    if "vis_frac" in some:
        timeseries_grid(series, src, "vis_frac", "visible / amodal area",
                        f"{title} - visible fraction of the full-body mask", fps,
                        os.path.join(out, "10_visible_fraction.png"), ylim=(0, 1.05))
    if "midline_len" in some:
        timeseries_grid(series, src, "midline_len", "midline length (px)",
                        f"{title} - midline length (mouth-head-middle-tail)", fps,
                        os.path.join(out, "11_midline_length.png"))
    report = write_report(series, src, fps, rec, title, os.path.join(out, "report"))
    print(f"done {rec}", flush=True)
    return summarise(rec, series, src), report


def resolve_paths(src, paths):
    if not paths:
        found = glob.glob(src["pattern"])
    else:
        found = []
        for p in paths:
            if os.path.isdir(p):
                found += glob.glob(os.path.join(p, src["dir_pattern"]), recursive=True)
            else:
                found.append(p)
    return sorted(p for p in found if not os.path.basename(p).startswith("._"))


def main():
    global AREA_JUMP, SPEED_JUMP
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*",
                    help="tracking files or folders (default: every recording of the source)")
    ap.add_argument("--source", choices=["zeus", "kaden"], default="zeus")
    ap.add_argument("--mask", choices=["visible", "amodal"], default="visible",
                    help="Kaden only: which mask the shape metrics use")
    ap.add_argument("--recompute", action="store_true", help="ignore cached metrics")
    ap.add_argument("--area-jump", type=float, default=AREA_JUMP,
                    help="relative frame-to-frame area change flagged as a jump")
    ap.add_argument("--speed-jump", type=float, default=SPEED_JUMP,
                    help="centroid speed (px/s) flagged as a jump")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    AREA_JUMP, SPEED_JUMP = args.area_jump, args.speed_jump
    src = get_source(args.source, args.mask)
    paths = resolve_paths(src, args.paths)
    if not paths:
        raise SystemExit(f"no {args.source} tracking files found")
    os.makedirs(src["out"], exist_ok=True)
    n = len(paths)
    with ProcessPoolExecutor(min(args.workers, n)) as ex:
        results = list(ex.map(process, paths, [args.source] * n, [args.mask] * n,
                              [args.recompute] * n, [args.area_jump] * n,
                              [args.speed_jump] * n))
    summ = pd.DataFrame([r for res, _ in results for r in res])
    report = pd.concat([rep for _, rep in results], ignore_index=True)
    summ_path = os.path.join(src["out"], "summary.csv")
    if args.paths and os.path.exists(summ_path):
        # Keep rows of recordings not rerun this time
        old = pd.read_csv(summ_path)
        old = old[~old.recording.isin(summ.recording)]
        summ = pd.concat([old, summ], ignore_index=True).sort_values(
            ["recording", "label"], key=lambda c: c.map(
                {l: i for i, l in enumerate(src["fish"])}) if c.name == "label" else c,
            kind="stable")
    summ.to_csv(summ_path, index=False)
    plot_summary(summ, src, os.path.join(src["out"], "summary.png"))
    report_dir = os.path.join(src["out"], "report")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "metrics_all_recordings.csv")
    if args.paths and os.path.exists(report_path):
        old = pd.read_csv(report_path)
        report = pd.concat([old[~old.recording.isin(report.recording)], report],
                           ignore_index=True).sort_values("recording", kind="stable")
    report.to_csv(report_path, index=False)
    with open(os.path.join(report_dir, "thresholds.txt"), "w") as f:
        f.write(f"area_jump={args.area_jump}\nspeed_jump_px_per_s={args.speed_jump}\n"
                f"mask={args.mask if args.source == 'kaden' else 'zeus'}\n")
    plot_report_summary(report, src, report_dir)
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
