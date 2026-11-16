import ast

SKIP_MARKERS = [
    "res = solve_source_fish(fish_mask, N, source_fish, Ihat, om, keep, n_t,",
    "def sample_phi_t(phi_t, p, h, N):",
    "from matplotlib.gridspec import GridSpec",
    'render_eod_cycle_video(mask_path, df_eod, pairs, N, out_path="eod_cycle.mp4"',
    "window_s_test = 10",
    "window_min = 2",
]

BUILD_NB_PATH = "/private/tmp/claude-501/-Volumes-Lenca-SSD-github-social-electric-fish/864876be-15ee-4109-9472-7f6a341c8f4b/scratchpad/build_nb.py"

def extract_code_cells(path):
    tree = ast.parse(open(path).read())
    codes = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "C":
                arg = call.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    codes.append(arg.value)
    return codes

ns = {"__name__": "__diag__", "delta_h": 0.5}
codes = extract_code_cells(BUILD_NB_PATH)
print(f"found {len(codes)} code cells total")
kept = 0
for code in codes:
    if any(m in code for m in SKIP_MARKERS):
        continue
    exec(compile(code, "<cell>", "exec"), ns)
    kept += 1
print(f"executed {kept} definition/data cells")

np = ns["np"]; pd = ns["pd"]; plt = ns["plt"]
df_eod = ns["df_eod"]; pairs = ns["pairs"]; N = ns["N"]; n_poles = ns["n_poles"]
delta_h = ns["delta_h"]; MAT = ns["MAT"]; mgz_file = ns["mgz_file"]
_event_v_sim = ns["_event_v_sim"]

# dense cache of the first 1001 real frames -- no streaming needed for this quick check
all_masks_dense = np.load(f"{mgz_file}_masks_first_1000_frames.npy", mmap_mode="r")

# pick a handful of representative real EOD events within the cached frame range
candidates_df = df_eod[df_eod.Frame <= 1000]
idxs = np.linspace(0, len(candidates_df) - 1, 6).astype(int)
chosen_rows = candidates_df.iloc[idxs]
print(f"chosen df_eod rows (labels): {list(chosen_rows.index)}, Frames: {list(chosen_rows.Frame)}")

pair_ids = [pid for pid, _, _ in pairs]

def peaknorm_attribution(row, frame_masks):
    # recorded full waveform per pair (25 samples), peak-normalized by the OVERALL max
    V = np.stack([row[[f"electrode_{pid}_time_{i}" for i in range(1, 26)]].to_numpy(dtype=float)
                 for pid in pair_ids])
    V = V - np.nanmean(V, axis=1, keepdims=True)
    overall_peak = np.nanmax(np.abs(V))
    V_norm = V / overall_peak
    t_star = np.unravel_index(np.nanargmax(np.abs(V)), V.shape)[1]
    dominant_pair = np.unravel_index(np.nanargmax(np.abs(V)), V.shape)[0]
    # temporal envelope of the pulse, from its OWN strongest channel: drive(t_star) = 1 by
    # construction, |drive(t)| <= 1 everywhere else (t_star is the global argmax of all of V)
    drive = V[dominant_pair] / V[dominant_pair, t_star]
    v_rec_norm = V_norm[:, t_star]                         # recorded snapshot, peak-normalized

    best = None
    preds = {}                                             # f -> full (npair, 25) predicted curve
    for f in range(frame_masks.shape[0]):
        if not frame_masks[f].any():
            continue
        v_sim, phi_with_u, pos, cur = _event_v_sim(frame_masks, f, N, delta_h, n_poles, pairs, **MAT)
        v_sim_norm = v_sim / np.max(np.abs(v_sim))          # candidate's OWN peak -> 1 (at t_star)
        resid = np.linalg.norm(v_sim_norm - v_rec_norm)
        # resistive/quasi-static model: spatial pattern is time-invariant, so the predicted
        # curve is just that pattern scaled by the pulse's own temporal envelope
        preds[f] = np.outer(v_sim_norm, drive)              # (npair, 25)
        if best is None or resid < best[0]:
            best = (resid, f)
    return V_norm, t_star, preds, (best[1] if best else None)

def plot_event(row, frame_masks, title):
    V_norm, t_star, preds, winner = peaknorm_attribution(row, frame_masks)
    npair = len(pairs); ncol = 2; nrow = int(np.ceil(npair / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), sharex=True)
    colors_fish = plt.cm.tab10.colors
    t_ms = np.arange(25) * (1.0 / 20000.0) * 1e3
    for k, pid in enumerate(pair_ids):
        ax = axes.ravel()[k]
        ax.plot(t_ms, V_norm[k], color="black", lw=2.0, label="recorded (peak-norm)", zorder=4)
        ax.axvline(t_ms[t_star], color="0.8", lw=0.7, ls=":")
        for f, pred_curve in preds.items():
            is_winner = (f == winner)
            ax.plot(t_ms, pred_curve[k], color=colors_fish[f],
                    lw=2.2 if is_winner else 1.2, alpha=1.0 if is_winner else 0.65,
                    ls="-" if is_winner else "--", zorder=5 if is_winner else 3,
                    label=f"fish {f}" + (" (winner)" if is_winner else ""))
        ax.axhline(0, color="0.85", lw=0.5)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(f"pair {pid}", fontsize=9)
        if k == 0: ax.legend(fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"/tmp/peaknorm_event_{row.name}.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname, "winner:", winner)
    return fname

saved = []
for lbl, row in chosen_rows.iterrows():
    frame_masks = np.asarray(all_masks_dense[:, int(row.Frame) - 1])
    fname = plot_event(row, frame_masks,
                       title=f"df_eod row {lbl}, Frame={int(row.Frame)} -- peak-normalized comparison")
    saved.append(fname)

print("ALL_SAVED:", saved)
