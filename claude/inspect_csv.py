#!/usr/bin/env python3
"""
inspect_csv.py — memory-aware inspection of a large CSV (e.g. a 700 MB EOD table).

A 700 MB CSV typically expands to ~3-7 GB as a pandas DataFrame, so this script
does NOT blindly read the whole thing. It:
  1. counts rows without loading the file,
  2. peeks at a sample to learn columns / dtypes,
  3. streams the WHOLE file in chunks to get exact per-column stats
     (missingness, min/max, value counts for low-cardinality columns),
  4. estimates full in-memory size and prints how to load it efficiently.

Usage:
    python inspect_csv.py /path/to/data.csv
    python inspect_csv.py --demo          # generate + inspect a synthetic file
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

SEP = "=" * 72


def count_rows(path: str) -> int:
    """Count newlines without loading the file into memory."""
    n = 0
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            n += buf.count(b"\n")
    return n  # includes header line


def human_bytes(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:,.1f} {unit}"
        num /= 1024.0
    return f"{num:,.1f} PB"


def peek(path: str, nrows: int = 5000) -> pd.DataFrame:
    """Read a small sample to learn the schema."""
    return pd.read_csv(path, nrows=nrows)


def low_cardinality_cols(sample: pd.DataFrame, max_unique: int = 50):
    """Columns that look categorical (good candidates for the fish-ID column)."""
    cols = []
    for c in sample.columns:
        nu = sample[c].nunique(dropna=True)
        if nu <= max_unique:
            cols.append((c, nu))
    return cols


def stream_summary(path: str, chunksize: int = 500_000):
    """
    One pass over the full file. Accumulates, per column:
      - non-null count, null count
      - numeric min / max / sum / sumsq  (for mean & std)
      - value counts for low-cardinality columns
    Returns a dict keyed by column name.
    """
    sample = peek(path, 2000)
    numeric_cols = sample.select_dtypes("number").columns.tolist()
    cat_candidates = [c for c, nu in low_cardinality_cols(sample, 50)]

    stats = {
        c: dict(n=0, nulls=0, vmin=np.inf, vmax=-np.inf, s=0.0, ss=0.0)
        for c in sample.columns
    }
    vcounts = {c: {} for c in cat_candidates}
    total = 0

    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)
        for c in chunk.columns:
            col = chunk[c]
            stats[c]["nulls"] += int(col.isna().sum())
            stats[c]["n"] += int(col.notna().sum())
            if c in numeric_cols:
                v = pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")
                v = v[np.isfinite(v)]
                if v.size:
                    stats[c]["vmin"] = min(stats[c]["vmin"], float(v.min()))
                    stats[c]["vmax"] = max(stats[c]["vmax"], float(v.max()))
                    stats[c]["s"] += float(v.sum())
                    stats[c]["ss"] += float((v * v).sum())
            if c in vcounts:
                for k, n in col.value_counts(dropna=True).items():
                    vcounts[c][k] = vcounts[c].get(k, 0) + int(n)
        print(f"  ...scanned {total:,} rows", end="\r", flush=True)
    print(" " * 40, end="\r")
    return stats, vcounts, total, numeric_cols


def guess_roles(columns):
    """Heuristic guesses for the columns this analysis needs."""
    low = {c: c.lower() for c in columns}
    def find(keys):
        return [c for c in columns if any(k in low[c] for k in keys)]
    return {
        "time":     find(["time", "timestamp", "t_", "sec", "sample", "frame"]),
        "identity": find(["id", "fish", "ident", "label", "tag", "cluster"]),
        "x":        find(["x", "pos_x", "xpos", "longitude"]),
        "y":        find(["y", "pos_y", "ypos", "latitude"]),
        "waveform": find(["amp", "wave", "feat", "pc", "width", "peak", "eod"]),
    }


def report(path: str):
    size = os.path.getsize(path)
    print(SEP)
    print(f"FILE: {path}")
    print(f"On-disk size: {human_bytes(size)}")
    print(SEP)

    print("Counting rows (no load)...", flush=True)
    nrows = count_rows(path) - 1  # minus header
    print(f"Data rows: {nrows:,}")

    sample = peek(path, 5000)
    ncols = sample.shape[1]
    print(f"Columns: {ncols}")
    print()

    # Estimate full in-memory footprint from the sample.
    per_row_bytes = sample.memory_usage(deep=True).sum() / len(sample)
    est_mem = per_row_bytes * nrows
    print(f"Est. DataFrame memory if loaded naively: {human_bytes(est_mem)} "
          f"(~{est_mem/size:.1f}x the file)")
    print()

    print("--- dtypes & sample values " + "-" * 45)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        info = pd.DataFrame({
            "dtype": sample.dtypes.astype(str),
            "sample_nunique": [sample[c].nunique(dropna=True) for c in sample.columns],
            "example": [sample[c].dropna().iloc[0] if sample[c].notna().any() else None
                        for c in sample.columns],
        })
        print(info)
    print()

    print("--- head(5) " + "-" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(sample.head())
    print()

    print("Streaming full-file summary (one pass)...", flush=True)
    stats, vcounts, total, numeric_cols = stream_summary(path)

    print("--- per-column summary (FULL file) " + "-" * 36)
    rows = []
    for c, s in stats.items():
        n, nulls = s["n"], s["nulls"]
        mean = s["s"] / n if (c in numeric_cols and n) else None
        std = (np.sqrt(max(s["ss"] / n - mean ** 2, 0.0))
               if (c in numeric_cols and n) else None)
        rows.append({
            "column": c,
            "non_null": n,
            "nulls": nulls,
            "null_%": round(100 * nulls / max(n + nulls, 1), 2),
            "min": (s["vmin"] if c in numeric_cols and n else None),
            "max": (s["vmax"] if c in numeric_cols and n else None),
            "mean": (round(mean, 4) if mean is not None else None),
            "std": (round(std, 4) if std is not None else None),
        })
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))
    print()

    if vcounts:
        print("--- value counts for low-cardinality columns (FULL file) " + "-" * 15)
        for c, vc in vcounts.items():
            if 0 < len(vc) <= 50:
                vs = sorted(vc.items(), key=lambda kv: -kv[1])
                print(f"[{c}]  ({len(vc)} unique)")
                for k, n in vs[:20]:
                    print(f"    {k!r:>20}: {n:,}")
        print()

    print("--- column-role guesses (verify before analysis) " + "-" * 22)
    for role, cands in guess_roles(list(sample.columns)).items():
        print(f"  {role:9s}: {cands}")
    print()

    print("--- how to load efficiently " + "-" * 44)
    print("  # Fastest / lowest memory (pandas >= 2):")
    print("  df = pd.read_csv(PATH, engine='pyarrow')")
    print()
    print("  # Or downcast + categoricals (set ID col to category):")
    print("  df = pd.read_csv(PATH, dtype={ID_COL: 'category'})")
    print("  for c in df.select_dtypes('float').columns:")
    print("      df[c] = pd.to_numeric(df[c], downcast='float')")
    print()
    print("  # If it still won't fit, read only what you need:")
    print("  df = pd.read_csv(PATH, usecols=[TIME_COL, ID_COL, X_COL, Y_COL])")
    print(SEP)
    print("NOTE: 'inspect all contents' = structure + full-file summary above.")
    print("      Printing every one of "
          f"{nrows:,} rows is not useful; filter/slice instead.")
    print(SEP)


def make_demo(path: str, n_per_fish: int = 200_000):
    """Write a synthetic EOD-like CSV resembling a real export."""
    rng = np.random.default_rng(0)
    frames = []
    t0 = 0.0
    for fid in [f"fish_{i}" for i in range(1, 5)]:
        ipi = rng.gamma(shape=4.0, scale=40.0, size=n_per_fish) / 1000.0  # ~160 ms
        t = t0 + np.cumsum(ipi)
        frames.append(pd.DataFrame({
            "time_s": t,
            "fish_id": fid,
            "x": rng.normal(0, 10, n_per_fish),
            "y": rng.normal(0, 10, n_per_fish),
            "eod_amp": rng.normal(1.0, 0.1, n_per_fish),
            "eod_width_ms": rng.normal(0.9, 0.05, n_per_fish),
        }))
    df = pd.concat(frames).sort_values("time_s").reset_index(drop=True)
    df.loc[df.sample(frac=0.01, random_state=1).index, "eod_amp"] = np.nan  # some NaNs
    df.to_csv(path, index=False)
    print(f"Wrote demo file: {path} ({human_bytes(os.path.getsize(path))})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="CSV path")
    ap.add_argument("--demo", action="store_true", help="make + inspect a synthetic file")
    args = ap.parse_args()

    if args.demo:
        demo_path = "/home/claude/demo_eod.csv"
        make_demo(demo_path)
        report(demo_path)
    elif args.path:
        report(args.path)
    else:
        ap.error("provide a CSV path or use --demo")
