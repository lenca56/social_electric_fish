#!/usr/bin/env python3
"""
eod_entropy.py — single-fish information-theoretic measures for EOD data.

Implements the spec (all in bits):
  H1            marginal IPI entropy            (repertoire size)
  H_n           block entropy of order n
  h_n           conditional entropy curve        h_n = H_{n+1} - H_n
  h_mu          entropy rate                     (irreducible unpredictability)
  rho           redundancy = 1 - h_mu/log2(M)    (strength of syntax)
  E             excess entropy                   (memory depth)
  k_hat         estimated Markov order
All entropies use the Miller-Madow bias correction. A shuffled baseline is
computed so you can see structure as the gap between real and shuffled h_n.

Configure the three column names below, then:
    python eod_entropy.py /path/to/data.csv
    python eod_entropy.py --demo        # synthetic structured vs random fish
Figures are written to ./eod_figs/.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless-safe
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIG — set these to YOUR column names (run inspect_csv.py first)
TIME_COL = "time_s"      # EOD timestamps. UNITS MATTER: see note below.
ID_COL   = "fish_id"     # fish identity
TIME_UNIT_TO_S = 1.0     # multiply TIME_COL by this to get SECONDS
                         #   seconds -> 1.0 ; milliseconds -> 1e-3
                         #   sample index -> 1/sampling_rate_hz
M       = 8              # IPI alphabet size (try 5 and 8)
MAX_N   = 6              # max block order for h_n curve
OUTDIR  = "eod_figs"
# ----------------------------------------------------------------------

LOG2 = np.log(2.0)


# ============================ core estimators =========================
def shannon_entropy_bits(symbols, miller_madow=True) -> float:
    """H(X) in bits with optional Miller-Madow bias correction."""
    symbols = np.asarray(symbols)
    N = symbols.size
    if N == 0:
        return np.nan
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / N
    H = float(-(p * np.log2(p)).sum())
    if miller_madow:
        m = counts.size                       # number of OBSERVED symbols
        H += (m - 1) / (2 * N * LOG2)
    return H


def _words_as_int(symbols, n, M):
    """Encode each length-n window as a base-M integer (fast block counting)."""
    s = np.asarray(symbols, dtype=np.int64)
    L = s.size - n + 1
    if L <= 0:
        return np.empty(0, dtype=np.int64)
    w = np.zeros(L, dtype=np.int64)
    for k in range(n):
        w = w * M + s[k:k + L]
    return w


def block_entropy_bits(symbols, n, M, miller_madow=True):
    """H_n in bits, and the number of length-n words used."""
    w = _words_as_int(symbols, n, M)
    return shannon_entropy_bits(w, miller_madow), w.size


def conditional_entropy_curve(symbols, M, max_n, miller_madow=True):
    """
    Returns:
      H : array H[0..max_n+1] of block entropies (H[0]=0, H[1]=H1, ...)
      h : array h[0..max_n]   conditional entropies, h[n]=H[n+1]-H[n]
      reliable : bool array over n flagging enough data (N_words >= 5 * M^n)
    """
    H = [0.0]
    counts = [np.inf]
    for n in range(1, max_n + 2):
        Hn, Lw = block_entropy_bits(symbols, n, M, miller_madow)
        H.append(Hn)
        counts.append(Lw)
    H = np.array(H)
    h = H[1:] - H[:-1]                          # h[n] for n=0..max_n
    # reliability of h_n needs H_{n+1}, i.e. (n+1)-grams:
    reliable = np.array([counts[n + 1] >= 5 * (M ** (n + 1))
                         for n in range(len(h))])
    return H, h, reliable


def summarize_fish(symbols, M, max_n):
    """Compute all scalar quantities for one fish's symbol sequence."""
    H, h, reliable = conditional_entropy_curve(symbols, M, max_n)
    H1 = H[1]
    # entropy rate: smallest reliable h_n (it decreases then plateaus)
    rel_idx = np.where(reliable)[0]
    if rel_idx.size:
        h_mu = float(h[rel_idx].min())
        n_used = int(rel_idx.max())
    else:
        h_mu, n_used = float(h[0]), 0
    rho = 1.0 - h_mu / np.log2(M)
    # excess entropy E = sum_n (h_n - h_mu), summed over reliable n
    E = float(np.sum((h[rel_idx] - h_mu))) if rel_idx.size else 0.0
    # Markov order: first n where the drop h_{n}-h_{n+1} < tol (within reliable n)
    tol = 0.02
    k_hat = n_used
    for n in range(len(h) - 1):
        if reliable[n] and reliable[n + 1] and (h[n] - h[n + 1]) < tol:
            k_hat = n
            break
    return dict(H1=H1, h_mu=h_mu, rho=rho, E=E, k_hat=k_hat,
                n_reliable=n_used, H=H, h=h, reliable=reliable)


# ============================ data plumbing ===========================
def per_fish_ipis(df, time_col, id_col, unit_to_s=1.0):
    """{fish_id: ipi_array_in_seconds}. Sorts by time within each fish."""
    out = {}
    for fid, g in df.groupby(id_col, observed=True):
        t = np.sort(g[time_col].to_numpy(dtype="float64")) * unit_to_s
        ipi = np.diff(t)
        out[fid] = ipi[ipi > 0]
    return out


def pooled_edges(ipis_by_fish, M):
    """Equiprobable (quantile) bin edges over the POOLED IPI distribution."""
    allipi = np.concatenate(list(ipis_by_fish.values()))
    edges = np.quantile(allipi, np.linspace(0, 1, M + 1))
    edges = np.unique(edges)                    # guard against ties
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def symbolize(ipi, edges):
    """Map IPIs to symbols 0..M-1 using shared edges."""
    return np.digitize(ipi, edges[1:-1]).astype(np.int64)


# ============================== plotting ==============================
def plot_all(ipis_by_fish, edges, results, M, max_n, outdir):
    os.makedirs(outdir, exist_ok=True)
    fish = list(ipis_by_fish.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fish), 3)))

    # 1) IPI histograms (log-x)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, fid in zip(colors, fish):
        ipi_ms = ipis_by_fish[fid] * 1000.0
        ax.hist(ipi_ms, bins=np.logspace(0, 3, 80), histtype="step",
                lw=1.6, color=c, label=str(fid))
    for e in edges[1:-1]:
        ax.axvline(e * 1000.0, color="0.7", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("IPI (ms)"); ax.set_ylabel("count")
    ax.set_title("IPI distributions (dotted = symbol bin edges)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/01_ipi_hist.png", dpi=130); plt.close(fig)

    # 2) Conditional-entropy curve h_n with shuffled baseline
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for c, fid in zip(colors, fish):
        sym = symbolize(ipis_by_fish[fid], edges)
        _, h, rel = conditional_entropy_curve(sym, M, max_n)
        ns = np.arange(len(h))
        ax.plot(ns, h, "-o", color=c, ms=4, label=str(fid))
        ax.plot(ns[~rel], h[~rel], "x", color=c, ms=7)   # mark unreliable
        sh = sym.copy(); rng.shuffle(sh)
        _, h0, _ = conditional_entropy_curve(sh, M, max_n)
        ax.plot(ns, h0, "--", color=c, lw=0.8, alpha=0.6)
    ax.axhline(np.log2(M), color="k", lw=0.8, ls=":")
    ax.text(0.02, np.log2(M) + 0.02, "log2(M) = max", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel("history length n"); ax.set_ylabel(r"$h_n = H_{n+1}-H_n$ (bits)")
    ax.set_title("Conditional entropy (solid=real, dashed=shuffled, x=low-data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/02_entropy_rate.png", dpi=130); plt.close(fig)

    # 3) Block entropy H_n scaling
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, fid in zip(colors, fish):
        ax.plot(np.arange(len(results[fid]["H"])), results[fid]["H"],
                "-o", color=c, ms=4, label=str(fid))
    ax.set_xlabel("block length n"); ax.set_ylabel(r"$H_n$ (bits)")
    ax.set_title("Block entropy scaling")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/03_block_entropy.png", dpi=130); plt.close(fig)

    # 4) IPI return maps  (tau_k vs tau_{k+1})
    ncol = len(fish)
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.2), squeeze=False)
    for ax, c, fid in zip(axes[0], colors, fish):
        ipi_ms = ipis_by_fish[fid] * 1000.0
        ax.plot(ipi_ms[:-1], ipi_ms[1:], ".", ms=1, alpha=0.15, color=c)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\tau_k$ (ms)"); ax.set_ylabel(r"$\tau_{k+1}$ (ms)")
        ax.set_title(str(fid), fontsize=9)
    fig.suptitle("IPI return maps (structure = off-diagonal clustering)")
    fig.tight_layout(); fig.savefig(f"{outdir}/04_return_maps.png", dpi=130); plt.close(fig)

    print(f"  figures -> {outdir}/01..04_*.png")


# =============================== driver ===============================
def run(df):
    ipis = per_fish_ipis(df, TIME_COL, ID_COL, TIME_UNIT_TO_S)
    ipis = {k: v for k, v in ipis.items() if v.size > 100}   # need data
    if not ipis:
        raise SystemExit("No fish with >100 IPIs. Check TIME_COL/ID_COL/units.")
    edges = pooled_edges(ipis, M)

    results = {}
    for fid, ipi in ipis.items():
        sym = symbolize(ipi, edges)
        results[fid] = summarize_fish(sym, M, MAX_N)
        results[fid]["N_ipi"] = ipi.size
        results[fid]["median_ipi_ms"] = float(np.median(ipi) * 1000)

    # scalar table
    tab = pd.DataFrame({
        fid: {
            "N_IPI":          r["N_ipi"],
            "median_IPI_ms":  round(r["median_ipi_ms"], 1),
            "H1_bits":        round(r["H1"], 3),
            "h_mu_bits":      round(r["h_mu"], 3),
            "redundancy_rho": round(r["rho"], 3),
            "excess_E_bits":  round(r["E"], 3),
            "Markov_order":   r["k_hat"],
            "n_reliable":     r["n_reliable"],
        } for fid, r in results.items()
    }).T
    print("\n=== single-fish information measures (bits) ===")
    print(tab.to_string())
    print(f"\nAlphabet size M = {M}  ->  max possible entropy = log2(M) = "
          f"{np.log2(M):.3f} bits")
    print("Interpretation: low h_mu / high rho / large E  =>  strong temporal "
          "syntax (a 'code' worth decoding).")

    plot_all(ipis, edges, results, M, MAX_N, OUTDIR)
    return tab, results


def make_demo_df(n=120_000):
    """Two structured fish (periodic + bursty) and two near-random fish."""
    rng = np.random.default_rng(1)
    frames = []

    # fish_1: strong period-3 motif in IPI -> low entropy rate
    motif = np.array([60, 60, 180]) / 1000.0
    ipi1 = np.tile(motif, n // 3) * rng.normal(1.0, 0.04, (n // 3) * 3)
    # fish_2: bursty (two regimes) with memory
    state = (rng.random(n) < 0.5)
    ipi2 = np.where(state, rng.normal(0.025, 0.003, n), rng.normal(0.25, 0.03, n))
    ipi2 = np.abs(ipi2)
    # fish_3, fish_4: renewal (gamma) -> entropy rate ~ H1
    ipi3 = rng.gamma(4.0, 0.04, n)
    ipi4 = rng.gamma(2.5, 0.06, n)

    for fid, ipi in [("fish_1", ipi1), ("fish_2", ipi2),
                     ("fish_3", ipi3), ("fish_4", ipi4)]:
        t = np.cumsum(ipi)
        frames.append(pd.DataFrame({TIME_COL: t, ID_COL: fid}))
    return pd.concat(frames).sort_values(TIME_COL).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        df = make_demo_df()
    elif args.path:
        # memory-friendly: only load the two needed columns
        df = pd.read_csv(args.path, usecols=[TIME_COL, ID_COL])
    else:
        ap.error("provide a CSV path or use --demo")
    run(df)
