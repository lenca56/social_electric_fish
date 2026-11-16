import ast

SKIP_MARKERS = [
    "res = solve_source_fish(fish_mask, N, source_fish, Ihat, om, keep, n_t,",   # section 8 demo (18 solves)
    "def sample_phi_t(phi_t, p, h, N):",                                        # section 9 defs+loop
    "from matplotlib.gridspec import GridSpec",                                 # section 9 plot
    'render_eod_cycle_video(mask_path, df_eod, pairs, N, out_path="eod_cycle.mp4"',  # section 10 action
    "window_min = 2",                                                          # section 12's 2-MINUTE job -- skip for now
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

ns = {"__name__": "__quick_test__", "delta_h": 0.5}   # normally set in the skipped section-8 demo cell
codes = extract_code_cells(BUILD_NB_PATH)
print(f"found {len(codes)} code cells total")
kept = 0
for code in codes:
    if any(m in code for m in SKIP_MARKERS):
        continue
    exec(compile(code, "<cell>", "exec"), ns)
    kept += 1
print(f"executed {kept} cells (this runs the real section-11/12 10s-test cell as written)")
print("done")
