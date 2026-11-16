import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_PATH = "/Volumes/Lenca_SSD/github/social_electric_fish/eod_two_layer_pipeline_compare_electrode.ipynb"
SKIP_INDICES = {20, 23, 28, 32, 36}   # expensive, already-validated action cells from earlier work; skip for speed

nb = nbformat.read(NB_PATH, as_version=4)

# --- fix cell 5: remove the body-length amplitude scaling from multipole_sources (same fix
#     applied to the main pipeline) -- otherwise this file's calibration would double-count size ---
src5 = "".join(nb["cells"][5]["source"])
old_tail = (
    "    length = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()\n"
    "    m = np.full(n + 1, 1.0 / n); m[0] = -1.0\n"
    "    return pos, m * I0 * length"
)
new_tail = (
    "    m = np.full(n + 1, 1.0 / n); m[0] = -1.0\n"
    "    return pos, m * I0"
)
assert old_tail in src5, "multipole_sources body text not found as expected in cell 5"
nb["cells"][5]["source"] = src5.replace(old_tail, new_tail)
nb["cells"][5]["outputs"] = []
nb["cells"][5]["execution_count"] = None

# --- new cells appended at the end ---
new_cells = []

def M(src):
    new_cells.append(nbformat.v4.new_markdown_cell(src))

def C(src):
    new_cells.append(nbformat.v4.new_code_cell(src))

M("""## 13. How good is the fish attribution? C_GLOBAL-scaled prediction vs. recorded voltage

Everything above this point was this notebook's state before two follow-up changes made in the
main `eod_two_layer_pipeline_clean.ipynb` pipeline: (1) `multipole_sources` no longer scales
current by body length (fixed in cell 5 above, just before this section, since an explicit
recorded/calibrated amplitude already reflects whatever that individual fish produced -- an
extra length multiplier would double-count it), and (2) fish attribution uses one calibrated
global scalar instead of a per-event free fit (below).

Attribution needs to answer "which fish produced this recorded EOD?" using the model, but the
model only knows field patterns in arbitrary unit-current terms, and the electrodes record in
real voltage units. Converting between them should be ONE constant of the recording setup, not
something that varies event to event -- but that constant can't be fit directly without already
knowing which fish is which (the very thing attribution is for). This is solved with
**alternating minimization**, over a batch of calibration events sampled across the whole
recording: (1) fix the scalar, assign each event to whichever candidate fish it currently
matches best; (2) fix those assignments, re-fit the one scalar from every assigned event pooled
together; repeat until assignments stop changing.

This section calibrates that scalar (`C_GLOBAL`), then plots a handful of calibration events to
directly see how good the resulting fit actually is: the full recorded waveform (black) per
electrode pair, with each candidate fish's `C_GLOBAL`-scaled prediction marked as a point at the
instant being compared -- the attributed (winning) fish's point is a larger gold star.""")

C("""def sample_phi_point(phi_grid, p, h, N):
    # Bilinear sample of a single (N,N) field at tank-frame point p=(x,y).
    fi, fj = p[0] / h, p[1] / h
    i, j = int(fi), int(fj); a, c = fi - i, fj - j
    def w(ii, jj): return phi_grid[min(max(ii, 0), N - 1), min(max(jj, 0), N - 1)]
    return (1-a)*(1-c)*w(i,j) + a*(1-c)*w(i+1,j) + (1-a)*c*w(i,j+1) + a*c*w(i+1,j+1)

def _event_v_rec(row, pairs):
    # Demeaned recorded electrode-pair voltages for one df_eod row, plus the sample index
    # (shared across all pairs) where the recording's amplitude peaks.
    pair_ids = [pid for pid, _, _ in pairs]
    V = np.stack([row[[f'electrode_{pid}_time_{i}' for i in range(1, 26)]].to_numpy(dtype=float)
                 for pid in pair_ids])
    V = V - np.nanmean(V, axis=1, keepdims=True)
    t_star = np.unravel_index(np.nanargmax(np.abs(V)), V.shape)[1]
    return V[:, t_star], t_star

def _event_v_sim(fish_mask_frame, f, N, delta_h, n_poles, pairs, **mat):
    # Unit-current (I0=1) electrode-pair pattern candidate fish f would produce this frame.
    h = 1.0 / (N - 1)
    phi_u, pos, cur = solve_instant_field(fish_mask_frame, N, f, 1.0, delta_h, n_poles, **mat)
    return np.array([sample_phi_point(phi_u, pp, h, N) - sample_phi_point(phi_u, pm, h, N)
                     for _, pp, pm in pairs])""")

C("""def calibrate_global_scalar(mgz_path, df_eod, pairs, N, delta_h=0.5, n_poles=100,
                           n_events=30, n_iter=10, **mat):
    # Jointly estimate ONE current-to-voltage scalar shared by every fish and every event
    # (see markdown above). n_events calibration rows are sampled evenly across the ENTIRE
    # df_eod (not just this notebook's one cached frame range); their body masks are pulled
    # from the full .mgz wherever in the recording they fall. Returns (c_global, events, assign)
    # -- events/assign are kept so later cells can re-plot the exact calibration sample.
    idxs = np.linspace(0, len(df_eod) - 1, n_events).astype(int)
    rows = df_eod.iloc[idxs]
    frames_needed = sorted(set((rows['Frame'] - 1).astype(int)))
    frame_masks_by_frame = load_frames_from_mgz(mgz_path, frames_needed)

    events = []
    for lbl, row in rows.iterrows():
        frame_masks = frame_masks_by_frame.get(int(row['Frame']) - 1)
        if frame_masks is None:
            continue
        v_sim_by_fish = {}
        for f in range(frame_masks.shape[0]):
            if frame_masks[f].any():
                v_sim_by_fish[f] = _event_v_sim(frame_masks, f, N, delta_h, n_poles, pairs, **mat)
        if v_sim_by_fish:
            v_rec, t_star = _event_v_rec(row, pairs)
            events.append(dict(label=lbl, row=row, v_rec=v_rec, t_star=t_star,
                              v_sim_by_fish=v_sim_by_fish))

    c, assign = 1.0, [None] * len(events)
    for it in range(n_iter):
        changed = 0
        for k, ev in enumerate(events):
            best_f = min(ev["v_sim_by_fish"],
                        key=lambda f: np.linalg.norm(c * ev["v_sim_by_fish"][f] - ev["v_rec"]))
            changed += (best_f != assign[k]); assign[k] = best_f
        num = sum(events[k]["v_sim_by_fish"][assign[k]] @ events[k]["v_rec"] for k in range(len(events)))
        den = sum(events[k]["v_sim_by_fish"][assign[k]] @ events[k]["v_sim_by_fish"][assign[k]]
                 for k in range(len(events)))
        c = num / den if den > 1e-30 else c
        print(f"  iter {it}: c_global={c:.4g}, {changed}/{len(events)} assignments changed")
        if changed == 0 and it > 0:
            break
    return c, events, assign""")

M("Run the calibration (small `n_events` to check convergence quickly).")

C("""C_GLOBAL, cal_events, cal_assign = calibrate_global_scalar(
    mgz_file, df_eod, pairs, N, delta_h=0.5, n_poles=n_poles, n_events=30, **MAT)
print(f"C_GLOBAL = {C_GLOBAL:.4g} (from {len(cal_events)} calibration events)")""")

M("""Plot a handful of calibration events: recorded waveform (black) per electrode pair, with
each candidate fish's `C_GLOBAL`-scaled unit-current prediction marked at the comparison instant
(the attributed/winning fish as a larger gold star).""")

C("""def plot_eod_assignment_check(ev, assigned_fish, pairs, C_GLOBAL, dt, title=""):
    npair = len(pairs); ncol = 2; nrow = int(np.ceil(npair / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow), sharex=True)
    colors_fish = plt.cm.tab10.colors
    t_ms = np.arange(25) * dt * 1e3
    for k, (pid, _, _) in enumerate(pairs):
        ax = axes.ravel()[k]
        cols = [f'electrode_{pid}_time_{i}' for i in range(1, 26)]
        v_full = ev['row'][cols].to_numpy(dtype=float); v_full = v_full - v_full.mean()
        ax.plot(t_ms, v_full, color='black', lw=1.5, label='recorded')
        ax.axvline(t_ms[ev['t_star']], color='0.75', lw=0.8, ls=':')
        for f, v_sim in ev['v_sim_by_fish'].items():
            pred = C_GLOBAL * v_sim[k]
            is_winner = (f == assigned_fish)
            ax.scatter(t_ms[ev['t_star']], pred, color=colors_fish[f],
                      s=110 if is_winner else 45, marker='*' if is_winner else 'o',
                      ec='k' if is_winner else 'none', lw=1.2, zorder=5,
                      label=f"fish {f}" + (" (assigned)" if is_winner else ""))
        ax.axhline(0, color='0.85', lw=0.5)
        ax.set_title(f"pair {pid}", fontsize=9)
        if k == 0: ax.legend(fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.show()

for k in range(min(4, len(cal_events))):
    ev = cal_events[k]
    plot_eod_assignment_check(ev, cal_assign[k], pairs, C_GLOBAL, dt,
                              title=f"calibration event {k} (df_eod row {ev['label']}, "
                                    f"Frame={int(ev['row']['Frame'])}) -- assigned fish {cal_assign[k]}")""")

for c in new_cells:
    nb["cells"].append(c)

print(f"notebook now has {len(nb['cells'])} cells; executing (skipping {sorted(SKIP_INDICES)})...")

client = NotebookClient(nb, timeout=1800, kernel_name="python3")
with client.setup_kernel():
    # bootstrap: cell 20 (skipped) normally sets delta_h as a side effect of its demo solve
    bootstrap = nbformat.v4.new_code_cell("delta_h = 0.5")
    client.execute_cell(bootstrap, 0)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if i in SKIP_INDICES:
            continue
        print(f"executing cell {i}...")
        try:
            client.execute_cell(cell, i)
        except CellExecutionError as e:
            print(f"ERROR in cell {i}:\\n{e}")
            raise

nbformat.write(nb, NB_PATH)
print("saved", NB_PATH)
