"""Why do fish bodies hurt, and what keeps the wrongly assigned clean-good EODs wrong? 2- and 3-fish silent controls.

EODs: clean good (mask QC + length check) that are assigned wrongly by ANY of four full runs (N150 square with / without bodies,
N200 tank with / without bodies; shape rule), plus a control sample of 300 clean-good EODs that all four runs get right (seed 0),
so that a variant that fixes wrong EODs but breaks right ones shows up.

Model for this test: N 150 in the measured tank (--geometry tank: tank_outline_updated, updated electrodes, 14.2 px/cm), unit current
along the mask midline (saved per-video midlines), sink at the tail tip; shape rule (L2 of MODEL_SIGN u / max|u| vs the peak-sample
pattern / max|r|) and the cosine rule. Only the treatment of the OTHER fish changes:
  skin + interior        as in every run so far: interior 100 x water inside a 0.005 cm skin as resistive as 0.1 cm of water. For a
                         body of half-width a ~ 0.5 cm this acts like a conductor of about a / 0.1 cm ~ 5 x water (the skin limits
                         how much current the very conductive interior can carry);
  no bodies              the other fish are water;
  homogeneous sigma s    the other fish are a uniform medium of s x water, no skin: s < 1 resistive, s > 1 conductive; a fish that
                         fills only part of the water depth acts, depth-averaged, like a weaker contrast (closer to 1);
  skin + interior, eroded  as the first, with each other fish's mask eroded by ERODE_PX px first (removes thin fins).
Diagnostics of the true emitter: its L2 with the midline reversed (head <-> tail; would a head-tail error explain the miss?),
the gap to the nearest other fish (eod_geometry_synch_fix_newest.csv), distance and heading to the nearest wall
(emitter_wall_geometry.csv).

    OMP_NUM_THREADS=1 python debug_bodies_wrong_eods.py [--workers 7]
Output in results/eod_assignment_<n>_fish_silent/Nov_10_2026/bodies_debug/: per_eod.csv, summary.csv, bodies_debug.png.
"""
import argparse, os
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eod_examples as ex
import eod_assignment_control_groups as g
from io_utils import load_assignment
from compute_electric_field import (build_multipole_b, multipole_sources, occupancy_field, build_A_shell, solve_potential,
                                    unflatten_to_grid, confine_sources)
from eod_assign import pair_voltages, compare_patterns, MODEL_SIGN

N, ERODE_PX, N_CONTROL = 150, 4, 300
SIGMAS = [0.05, 0.3, 0.7, 1.5, 3.0, 6.0, 20.0]
VARIANTS = ['skin + interior (current)', 'no bodies'] + [f'homogeneous sigma {s:g}' for s in SIGMAS] + \
           [f'skin + interior, masks eroded {ERODE_PX} px']
REC = {'2-fish': dict(res='results/eod_ground_truth', emitter=1), '3-fish': dict(res='results/eod_ground_truth_3fish', emitter=0)}
RUNS = {'N150 square': '{res}/conductivities/my_model_shape_synch_fix_newest.csv', 'N150 square, no bodies': '{out}/N150_image_nobodies_none/eod_assignment_shape.csv',
        'N200 tank': '{out}/N200_tank_none/eod_assignment_shape.csv', 'N200 tank, no bodies': '{out}/N200_tank_nobodies_none/eod_assignment_shape.csv'}
CTX = None


def variant_matrix(occ, v):
    """build_A_shell for one treatment of the other fish (occ: their occupancy)."""
    m = CTX['mat']; inside = CTX['inside']
    if v == 'no bodies': return build_A_shell(np.zeros_like(occ), N, **m, inside=inside)
    if v.startswith('homogeneous'):
        s = float(v.split()[-1]); return build_A_shell(occ, N, m['delta_h'], 1.0, s, s, inside=inside)   # skin = interior: no skin
    return build_A_shell(occ, N, **m, inside=inside)


def solve_eod(k):
    """{variant: (n_fish, 10) u}, and the true emitter's u with its midline reversed (current bodies, no bodies)."""
    fm = CTX['masks'][k]; M = g.frame_midlines(k, CTX); nf = len(CTX['fish']); em = CTX['em']
    fe = np.array([binary_erosion(m, iterations=ERODE_PX) if m.any() else m for m in fm])
    out = {v: np.full((nf, 10), np.nan) for v in VARIANTS}; flip = {}
    for f in range(nf):
        if not np.isfinite(M[f]).all(): continue
        occ, occ_e = occupancy_field(fm, N, skip=f), occupancy_field(fe, N, skip=f)
        mids = [(M[f], False)] + ([(M[f][::-1], True)] if f == em else [])
        for mid, flipped in mids:
            b = confine_sources(build_multipole_b(*multipole_sources(mid, I0=1.0, sink_frac=g.SINK), 1.0 / (N - 1), N), CTX['inside'])
            for v in (VARIANTS if not flipped else ['skin + interior (current)', 'no bodies']):
                A = variant_matrix(occ_e if 'eroded' in v else occ, v)
                u = pair_voltages(unflatten_to_grid(solve_potential(A, b), N).real, CTX['pairs'], N)
                if flipped: flip[v] = u
                else: out[v][f] = u
    return int(k), out, flip


def picks(U, r):
    d = np.array([compare_patterns(u, r)[0] if np.isfinite(u).all() else np.inf for u in U])
    cos = np.array([(MODEL_SIGN * u) @ r / (np.linalg.norm(u) * np.linalg.norm(r)) if np.isfinite(u).all() else -np.inf for u in U])
    return d, int(np.argmin(d)), int(np.argmax(cos))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=7); a = ap.parse_args()
    summary = []
    for rec, r in REC.items():
        c = g.REC[rec]; em = r['emitter']; out = f"{c['out']}/bodies_debug"; os.makedirs(out, exist_ok=True)
        q = pd.read_csv(f"{r['res']}/eod_clean_good_synch_fix_newest.csv"); clean = q.clean_good.to_numpy(bool)
        runs = {k: load_assignment(p.format(res=r['res'], out=c['out'])).pick for k, p in RUNS.items()}
        wrong = clean & np.logical_or.reduce([p != em for p in runs.values()])
        right_all = clean & np.logical_and.reduce([p == em for p in runs.values()])
        ctrl = np.sort(np.random.default_rng(0).choice(np.flatnonzero(right_all), N_CONTROL, replace=False))
        rows_i = np.r_[np.flatnonzero(wrong), ctrl]; is_wrong = np.r_[np.ones(wrong.sum(), bool), np.zeros(N_CONTROL, bool)]
        base = load_assignment(RUNS['N200 tank'].format(res=r['res'], out=c['out'])); assert base.pair_ids == list(range(1, 11)); R = base.recorded
        frames = q.Frame.to_numpy()[rows_i]
        CTX = g.context(rec, N, [None], 'tank'); CTX['mids'] = g.load_midlines(rec); CTX['em'] = em
        print(f'{rec}: {wrong.sum()} clean-good EODs wrong in at least one of {list(RUNS)}; {N_CONTROL} controls right in all; '
              f'{len(np.unique(frames))} frames to solve', flush=True)
        with mp.get_context('fork').Pool(a.workers) as pool:
            res = dict((k, (o, fl)) for k, o, fl in pool.imap_unordered(solve_eod, np.unique(frames).tolist(), chunksize=2))
        geo = pd.read_csv(f"{r['res']}/eod_geometry_synch_fix_newest.csv").set_index('eod_row')
        wall = pd.read_csv(f"{c['out']}/emitter_wall_geometry.csv").set_index('eod_row')
        rows = []
        for i, k, w in zip(rows_i, frames, is_wrong):
            o, fl = res[int(k)]; row = dict(eod_row=int(i), Frame=int(k), set='wrong' if w else 'control',
                                           gap_emitter_cm=geo.gap_emitter_cm.get(i, np.nan), wall_cm=wall.wall_cm.get(i, np.nan),
                                           heading_deg=wall.heading_deg.get(i, np.nan), **{f'run | {k_}': int(p[i]) for k_, p in runs.items()})
            for v in VARIANTS:
                d, pl, pc = picks(o[v], R[i]); row[f'pick L2 | {v}'] = pl; row[f'pick cosine | {v}'] = pc
                row[f'L2 emitter | {v}'] = d[em]; row[f'L2 best other | {v}'] = np.min(np.delete(d, em))
            for v, u in fl.items(): row[f'L2 emitter reversed | {v}'] = compare_patterns(u, R[i])[0]
            rows.append(row)
        T = pd.DataFrame(rows); T.to_csv(f'{out}/per_eod.csv', index=False)
        for v in VARIANTS:
            for rule in ('L2', 'cosine'):
                ok = T[f'pick {rule} | {v}'].to_numpy() == em; cur = T[f'pick {rule} | skin + interior (current)'].to_numpy() == em
                Wm = T.set.to_numpy() == 'wrong'
                summary.append(dict(recording=rec, variant=v, rule=rule, n_wrong_set=int(Wm.sum()), correct_in_wrong_set=int(ok[Wm].sum()),
                                    correct_in_controls=int(ok[~Wm].sum()), n_controls=int((~Wm).sum()),
                                    fixed_vs_current=int((ok & ~cur).sum()), broken_vs_current=int((~ok & cur).sum())))
        W = T[T.set == 'wrong']; cur = 'skin + interior (current)'
        rev_better = (W[f'L2 emitter reversed | {cur}'] < W[f'L2 emitter | {cur}']).mean() * 100
        rev_wins = (W[f'L2 emitter reversed | {cur}'] < W[f'L2 best other | {cur}']).mean() * 100
        print(f'{rec}: in the wrong set the reversed emitter (head <-> tail) fits better than the normal one in {rev_better:.0f}% '
              f'and would beat every other fish in {rev_wins:.0f}%', flush=True)
    S = pd.DataFrame(summary)
    for rec in REC: S[S.recording == rec].to_csv(f"{g.REC[rec]['out']}/bodies_debug/summary.csv", index=False)
    with pd.option_context('display.width', 220):
        print(S.to_string(index=False))

    # figure: correct in the wrong set and in the controls, per variant (shape L2 rule and cosine)
    fig, axs = plt.subplots(2, 2, figsize=(18, 9.5))
    for row, rec in enumerate(REC):
        for col, rule in enumerate(('L2', 'cosine')):
            a = axs[row, col]; s = S[(S.recording == rec) & (S.rule == rule)].set_index('variant').reindex(VARIANTS); x = np.arange(len(VARIANTS))
            a.bar(x - 0.2, s.correct_in_wrong_set, 0.38, color='#2a78d6', label=f'wrong set: EODs now right (of {s.n_wrong_set.iloc[0]})')
            a.bar(x + 0.2, s.n_controls - s.correct_in_controls, 0.38, color='#eb6834', label=f'controls: EODs now wrong (of {s.n_controls.iloc[0]})')
            for xi, v1, v2 in zip(x, s.correct_in_wrong_set, s.n_controls - s.correct_in_controls):
                a.text(xi - 0.2, v1, str(int(v1)), ha='center', va='bottom', fontsize=8); a.text(xi + 0.2, v2, str(int(v2)), ha='center', va='bottom', fontsize=8)
            a.set_xticks(x, [v.replace('homogeneous sigma', 'σ').replace('skin + interior', 'skin+int.').replace(', masks eroded', ',\neroded') for v in VARIANTS],
                         fontsize=8, rotation=30, ha='right')
            a.set_ylabel('number of EODs'); a.set_title(f'{rec}, shape rule: {rule} (N 150 tank; only the other fish change)', fontsize=10.5)
            a.legend(fontsize=8.5, frameon=False); a.grid(axis='y', alpha=0.3); a.set_axisbelow(True)
            for s_ in ('top', 'right'): a.spines[s_].set_visible(False)
    fig.suptitle('Treatment of the other fish: how many wrongly assigned clean-good EODs become right (blue), and how many of 300 right '
                 'controls become wrong (orange)', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for rec in REC: fig.savefig(f"{g.REC[rec]['out']}/bodies_debug/bodies_debug.png", dpi=120)
