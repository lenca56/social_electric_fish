"""My model (numerical, shape only) on the *_synch_fix_newest.csv EOD tables, every EOD of the 2- or 3-fish silent control,
for several model versions in one pass over the frames.

Model (compute_electric_field.py): 2D resistive field on an N x N grid over the video frame, BEM materials (materials(N): interior
100 x water, 0.005 cm skin as resistive as 0.1 cm of water), point currents along the mask midline of the emitting fish
(multipole_sources, unit current, sink at the tail tip), every other fish a body (build_A_shell); each fish in turn is the emitter.
Masks: ZEUS (*_Shiv_tracking_final.pkl) of the EOD's frame in the newest table (no offset).

Geometry (--geometry):
  tank   (default) the non-conductive tank walls along data/electrodes/tank_outline_updated_<n>-fish.csv (60 x 60 cm tank, 2 cm
         square cut out at the top left; tank_outline_electrodes.py), electrodes at electrode_locations_updated_<n>-fish.csv,
         model scale 14.2 px/cm (compute_electric_field.PX_PER_CM);
  tank_enlarged  as tank, but every wall moved outwards by its typical fish-mask overshoot and the electrodes right at the new
         walls / corners (tank_outline_enlarged_<n>-fish.csv, electrode_locations_enlarged_<n>-fish.csv; tank_outline_electrodes.py
         --variant enlarged);
  image  walls at the edges of the video frame, the original electrode positions (electrodes_signed_px_coord_<n>-fish.csv) and
         Denis' 15.2 px/cm: the model of the 2026-11-05 run (BASELINE), used by --check.
Versions (--versions, comma separated): 'none' (no electrode discs) or '<radius cm>:<sigma x water>' (conducting discs at the
electrodes, electrode_occupancy). An electrode's voltage is the potential at its centre (pair_voltages).

Each video frame is read once: masks, midlines, occupancy and sources are built once per frame and fish, then every version is
solved. Comparison (eod_assign.compare_patterns): recorded peak pattern r = the 10 pair voltages at the sample where |V| peaks
(traces demeaned per pair); model pattern MODEL_SIGN x u (-u); both scaled to max |v| = 1 over the pairs; d = L2 distance; the
fish with the smallest d gets the EOD (a fish without midline cannot; pick = -1 when no fish has a model).

Output, per version, in <out>/<tag>/ (tag e.g. N200_tank_none, N200_tank_r0.5_s1e7; --eods n adds _random<n>):
  eod_assignment_shape.csv  one row per EOD: eod_row, Frame (video frame of the EOD), r_raw_p<q> (recorded peak pattern, V),
                            r_norm = max_q |r_raw|, and per fish f: u<f>_raw_p<q> (model unit-current pair voltages V(+) - V(-),
                            before MODEL_SIGN), u<f>_norm = max_q |u<f>_raw|, d<f> (L2, NaN without a model); pick.
  config.json               the model settings of that version
and <out>/model_frames_<run>.pkl: {frame: (n_versions, n_fish, n_pairs) u} with the version tags under '__versions__', the
checkpoint (resumable).

Mask midlines, once per video for every run (MIDLINES, e.g. results/eod_assignment_2_fish_silent/mask_midlines_by_frame_newest.npz):
frames (every video frame of an EOD in the newest table), midlines_tank (n_frames, n_fish, 101, 2: (row, col) / (P - 1), what the
model uses) and midlines_px (same points as video px (x = column, y = row)), tail -> head, NaN where a fish has no midline
(eod_examples.fish_midlines: fish_midline_with_head on the ZEUS mask, area rule for the head). A run reads the midlines from this
file and fits (and adds) only the frames that are missing; --midlines-only fills the file without solving.

    OMP_NUM_THREADS=1 python eod_assignment_control_groups.py --rec 3-fish --N 200 --versions none,0.25:1e7,0.5:1e7 [--workers 7]
    OMP_NUM_THREADS=1 python eod_assignment_control_groups.py --rec 3-fish --check 50   # image geometry, N 150: must match BASELINE
    OMP_NUM_THREADS=1 python eod_assignment_control_groups.py --rec 3-fish --midlines-only [--workers 7]

--masks kaden (2026-11-10): Kaden's efish4-track masks (data/track_Kaden/<n>fish/*.pkl, visible masks) instead of ZEUS. His frame k is
video frame k (checked: median IoU with the ZEUS masks 0.83-0.93 at offset 0, lower at +-1) and his letters are mapped to the
ZEUS fish numbers (2-fish A = 1, B = 0; 3-fish A = 1, B = 2, C = 0), so emitters and fish indices stay the same. Midlines then
come from results/eod_assignment_<n>_fish_silent/with_Kaden_tracking/mask_midlines_by_frame_newest.npz and results go to that
folder by default.
--body-sigmas s1,s2,...: the other fish as a uniform medium of s x water without skin (instead of skin + interior), one version
per value, all solved in one pass over the frames (tag N<N>_<geometry>_body<s>_none; needs --versions none).
--interior-sigmas s1,s2,...: the other fish keep the skin (same resistance: as resistive as SKIN_LAMBDA_CM of water) and get an
interior of s x water; s = 100 is the original model (tag N<N>_<geometry>_int<s>_none; needs --versions none).
--skin-lambdas l1,l2,... (with --interior-sigmas): the skin as resistive as l cm of water instead of SKIN_LAMBDA_CM = 0.1, one version
per interior x skin (tag ..._int<s>_skin<l>_none); l = 0 is no skin.
"""
import argparse, json, os, re, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from io_utils import load_masks
import eod_examples as ex
from compute_electric_field import (multipole_sources, build_multipole_b, occupancy_field, build_A_shell, solve_potential,
                                    unflatten_to_grid, materials, electrode_occupancy, electrode_points, tank_inside,
                                    confine_sources, cell_cm, SIGMA_WATER, SIGMA_INTERIOR, SIGMA_SKIN, SKIN_CM, SKIN_LAMBDA_CM,
                                    PX_PER_CM, PX_PER_CM_DENIS, potential_solver)
from eod_assign import pair_voltages, load_electrode_pairs, peak_pattern, compare_patterns, cosine_similarity, MODEL_SIGN

SINK = 0.0                                                          # sink at the tail tip (beat 5% in every control run)
REC = {
    '2-fish': dict(fish=(0, 1), emitter=1, res='results/eod_ground_truth', out='results/eod_assignment_2_fish_silent/Nov_10_2026',
                   track='data/2_fish_silent/elephantnose_2024-02-13T14-33-21_Shiv_tracking_final.pkl',
                   newest='data/2_fish_silent/Data_2fish_2024-02-13T14_33_21_synch_fix_newest.csv',
                   electrodes='data/electrodes/electrodes_signed_px_coord_2-fish.csv',
                   electrodes_updated='data/electrodes/electrode_locations_updated_2-fish.csv',
                   outline='data/electrodes/tank_outline_updated_2-fish.csv',
                   midlines='results/eod_assignment_2_fish_silent/mask_midlines_by_frame_newest.npz',
                   kaden='data/track_Kaden/2fish/video2024-02-13T14_33_21.pkl', kaden_order=(1, 0),
                   out_kaden='results/eod_assignment_2_fish_silent/with_Kaden_tracking'),
    '3-fish': dict(fish=(0, 1, 2), emitter=0, res='results/eod_ground_truth_3fish', out='results/eod_assignment_3_fish_silent/Nov_10_2026',
                   track='data/3_fish_silent/elephantnose_2024-10-15T14-12-25_Shiv_tracking_final.pkl',
                   newest='data/3_fish_silent/Data_3fish_2024-10-15T14_12_25_synch_fix_newest.csv',
                   electrodes='data/electrodes/electrodes_signed_px_coord_3-fish.csv',
                   electrodes_updated='data/electrodes/electrode_locations_updated_3-fish.csv',
                   outline='data/electrodes/tank_outline_updated_3-fish.csv',
                   midlines='results/eod_assignment_3_fish_silent/mask_midlines_by_frame_newest.npz',
                   kaden='data/track_Kaden/3fish/video2024-10-15T14_12_25.pkl', kaden_order=(2, 0, 1),
                   out_kaden='results/eod_assignment_3_fish_silent/with_Kaden_tracking'),
}
BASELINE = '{res}/conductivities/my_model_shape_synch_fix_newest.csv'  # N 150, image geometry, no discs (run of 2026-11-05)
C = None                                                            # run context, set before the pool forks


def parse_versions(text):
    """'none,0.25:1e7' -> [None, (0.25, 1e7)]."""
    return [None if v == 'none' else tuple(float(x) for x in v.split(':')) for v in text.split(',')]


def version_tag(N, geometry, v, bodies=True, body_sigma=None):
    bs = '' if body_sigma is None else (f'_int{body_sigma[1]:g}' + (f'_skin{body_sigma[2]:g}' if len(body_sigma) > 2 else '')
                                        if isinstance(body_sigma, tuple) else f'_body{body_sigma:g}')
    return (f'N{N}_{geometry}{"" if bodies else "_nobodies"}{bs}_' +
            ('none' if v is None else f'r{v[0]:g}_s{v[1]:.0e}'.replace('e+0', 'e').replace('e+', 'e')))


def midlines_path(rec, masks='zeus'):
    return REC[rec]['midlines'] if masks == 'zeus' else f"{REC[rec]['out_kaden']}/mask_midlines_by_frame_newest.npz"


def context(rec, N, versions=(None,), geometry='tank', bodies=True, masks='zeus', body_sigmas=None):
    """Everything solve_frame needs: masks, fish, electrode pairs, grid, materials, tank walls and the discs of every version.
       bodies=False: the other fish are not modelled (each candidate a bare current source in the tank, as in Denis' model); the
       medium is then the same for every fish and frame, so each version's matrix is factorised once (ctx['solvers'])."""
    c = REC[rec]
    video = (load_masks(c['track'], 'shiv', fish_order=c['fish']) if masks == 'zeus' else
             load_masks(c['kaden'], fish_order=c['kaden_order']))                          # video[k]: (n_fish, P, P), ZEUS fish order
    P = video.shape[0]
    if body_sigmas is not None:
        assert list(versions) == [None], '--body-sigmas needs --versions none'; versions = [None] * len(body_sigmas)
    if geometry in ('tank', 'tank_enlarged'):
        variant = 'updated' if geometry == 'tank' else 'enlarged'
        px_cm, el_file = PX_PER_CM, c['electrodes_updated'].replace('updated', variant)
        inside = tank_inside(pd.read_csv(c['outline'].replace('updated', variant))[['x', 'y']].to_numpy(), N, P)
    else:
        px_cm, el_file, inside = PX_PER_CM_DENIS, c['electrodes'], None
    pairs = load_electrode_pairs(el_file, P); pts = electrode_points(pairs)
    elec = [None if v is None else (electrode_occupancy(pts, N, v[0], P, px_cm), v[1]) for v in versions]
    ctx = dict(masks=video, tracking=masks, fish=c['fish'], P=P, pairs=pairs, N=N, mat=materials(N, P, px_cm), inside=inside,
               elec=elec, px_per_cm=px_cm, electrodes_file=el_file, geometry=geometry, bodies=bodies,
               body=list(body_sigmas) if body_sigmas is not None else [None] * len(elec))
    if not bodies:
        ctx['solvers'] = [potential_solver(build_A_shell(np.zeros((N, N)), N, **ctx['mat'], electrodes=e, inside=inside)) for e in elec]
    return ctx


def load_midlines(rec, masks='zeus'):
    """{frame: (n_fish, 101, 2) midlines in tank coordinates, NaN where none} from the video's MIDLINES file (empty if none)."""
    p = midlines_path(rec, masks)
    if not os.path.exists(p): return {}
    z = np.load(p); return dict(zip(z['frames'].tolist(), z['midlines_tank']))


def save_midlines(rec, mids, P, masks='zeus'):
    p = midlines_path(rec, masks); os.makedirs(os.path.dirname(p), exist_ok=True)
    frames = np.array(sorted(mids)); M = np.stack([mids[k] for k in frames])
    np.savez_compressed(p + '.tmp.npz', frames=frames, midlines_tank=M, midlines_px=M[..., ::-1] * (P - 1),
                        n_ctrl=ex.n_poles + 1, mask_source=REC[rec]['track'] if masks == 'zeus' else REC[rec]['kaden'],
                        eod_table=REC[rec]['newest'],
                        method=f'eod_examples.fish_midlines: fish_midline_with_head on the {masks} mask (area rule head), tail -> head')
    os.replace(p + '.tmp.npz', p)


def frame_midlines(k, ctx=None):
    """(n_fish, 101, 2) midlines of video frame k (tank coordinates, NaN where none): from the video's midline file if there,
       else fitted on the masks."""
    ctx = ctx or C; m = ctx.get('mids', {}).get(int(k))
    if m is not None: return m
    out = np.full((len(ctx['fish']), ex.n_poles + 1, 2), np.nan)
    for f, mid in enumerate(ex.fish_midlines(ctx['masks'][k])):
        if mid is not None: out[f] = mid
    return out


def midlines_only(k):
    return int(k), frame_midlines(k)


def solve_frame(k, ctx=None):
    """Unit-current pair voltages (n_versions, n_fish, n_pairs) of video frame k, each fish in turn emitting (NaN: no midline),
       and the frame's midlines. Masks, midlines, occupancy and sources are built once; each version only rebuilds the matrix
       and solves."""
    ctx = ctx or C; N = ctx['N']; inside = ctx['inside']; bodies = ctx.get('bodies', True)
    fm = ctx['masks'][k] if bodies else None; M = frame_midlines(k, ctx)
    mids = [M[f] if np.isfinite(M[f]).all() else None for f in range(len(M))]
    u = np.full((len(ctx['elec']), len(ctx['fish']), len(ctx['pairs'])), np.nan)
    for f in ctx['fish']:
        if mids[f] is None: continue
        b = build_multipole_b(*multipole_sources(mids[f], I0=1.0, sink_frac=SINK), 1.0 / (N - 1), N)
        if inside is not None: b = confine_sources(b, inside)
        if not bodies:
            for j, solve in enumerate(ctx['solvers']):
                u[j, f] = pair_voltages(unflatten_to_grid(solve(b), N).real, ctx['pairs'], N)
            continue
        occ = occupancy_field(fm, N, skip=f)
        for j, (elec, s) in enumerate(zip(ctx['elec'], ctx['body'])):
            if s is None: mat = ctx['mat']
            elif isinstance(s, tuple):                                 # ('int', s[, lam]): skin kept (lam cm of water), interior s
                lam = s[2] if len(s) > 2 else SKIN_LAMBDA_CM
                mat = dict(ctx['mat'], g_interior=s[1], g_skin=1 / (lam / SKIN_CM + 1 / s[1]))
            else: mat = dict(delta_h=ctx['mat']['delta_h'], g_water=1.0, g_skin=s, g_interior=s)                         # uniform body
            A = build_A_shell(occ, N, **mat, electrodes=elec, inside=inside)
            u[j, f] = pair_voltages(unflatten_to_grid(solve_potential(A, b), N).real, ctx['pairs'], N)
    return int(k), u, M


def recorded_patterns(path, pair_ids):
    """Recorded peak pattern (n_eods, n_pairs) of an EOD table: traces demeaned per pair, values at the sample where |V| peaks."""
    head = pd.read_csv(path, nrows=0).columns
    pat = re.compile(r'electrode_(\d+)_t(?:ime)?_([mp]?)(\d+)$'); cols = {}
    for col in head:
        m = pat.match(col)
        if m: cols.setdefault(int(m[1]), []).append(((-1 if m[2] == 'm' else 1) * int(m[3]), col))
    order = {p: [col for _, col in sorted(v)] for p, v in cols.items()}
    d = pd.read_csv(path, usecols=['Frame'] + [col for p in pair_ids for col in order[p]])
    out = []
    for _, row in d.iterrows():
        V = np.stack([row[order[p]].to_numpy(float) for p in pair_ids]); V = V - np.nanmean(V, axis=1, keepdims=True)
        out.append(peak_pattern(V)[0])
    return d.Frame.to_numpy(), np.array(out)


def shape_assignment(U, R):
    """Per EOD and fish (NaN where a fish has no model): L2 distance (compare_patterns) and cosine similarity (cosine_similarity),
       and the pick of each rule: smallest L2 (pick) and largest cosine (pick_cosine); -1 when no fish has a model."""
    D, Cs = np.full(U.shape[:2], np.nan), np.full(U.shape[:2], np.nan)
    for i in range(len(R)):
        for f in range(U.shape[1]):
            if np.isfinite(U[i, f]).all(): D[i, f] = compare_patterns(U[i, f], R[i])[0]; Cs[i, f] = cosine_similarity(U[i, f], R[i])
    ok = np.isfinite(D).any(axis=1)
    pick = np.where(ok, np.argmin(np.where(np.isfinite(D), D, np.inf), axis=1), -1)
    pick_cos = np.where(ok, np.argmax(np.where(np.isfinite(Cs), Cs, -np.inf), axis=1), -1)
    return D, pick, Cs, pick_cos


def result_table(frames, R, U, D, pick, pair_ids, fish, rows, Cs=None, pick_cos=None):
    res = {'eod_row': rows, 'Frame': frames}
    res.update({f'r_raw_p{q}': R[:, j] for j, q in enumerate(pair_ids)}); res['r_norm'] = np.abs(R).max(axis=1)
    for f in fish:
        res.update({f'u{f}_raw_p{q}': U[:, f, j] for j, q in enumerate(pair_ids)})
        res[f'u{f}_norm'] = np.abs(U[:, f]).max(axis=1); res[f'd{f}'] = D[:, f]
        if Cs is not None: res[f'cos{f}'] = Cs[:, f]
    res['pick'] = pick
    if pick_cos is not None: res['pick_cosine'] = pick_cos
    return pd.DataFrame(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rec', required=True, choices=list(REC))
    ap.add_argument('--N', type=int, default=200); ap.add_argument('--geometry', choices=['tank', 'tank_enlarged', 'image'], default='tank')
    ap.add_argument('--versions', default='none', help="comma separated: none or <radius cm>:<sigma x water>")
    ap.add_argument('--workers', type=int, default=7); ap.add_argument('--out', default=None)
    ap.add_argument('--eods', type=int, default=0, help='only n random EODs (seed 0), for tests')
    ap.add_argument('--check', type=int, default=0, help='image geometry, N 150, no discs: n EODs must match BASELINE')
    ap.add_argument('--midlines-only', action='store_true', help="fill the video's midline file, no solves")
    ap.add_argument('--bodies', choices=['yes', 'no'], default='yes', help='no: the other fish are not modelled')
    ap.add_argument('--masks', choices=['zeus', 'kaden'], default='zeus')
    ap.add_argument('--body-sigmas', default=None, help='comma separated: other fish as uniform s x water (no skin), one version each')
    ap.add_argument('--interior-sigmas', default=None, help='comma separated: other fish with skin and an interior of s x water')
    ap.add_argument('--skin-lambdas', default=None, help='with --interior-sigmas: comma separated skin resistances (cm of water), '
                                                         'one version per interior x skin')
    a = ap.parse_args(); c = REC[a.rec]

    if a.check:                                                     # reproduce the baseline run (same model before the refactor)
        C = context(a.rec, 150, [None], 'image'); pair_ids = [pid for pid, _, _ in C['pairs']]
        frames, R = recorded_patterns(c['newest'], pair_ids); rows = np.arange(len(frames))
        base = pd.read_csv(BASELINE.format(res=c['res'])); assert (base.Frame.to_numpy() == frames).all()
        sel = rows[::len(rows) // a.check][:a.check]
        U = np.stack([solve_frame(frames[i])[1][0] for i in sel]); D, pick, _, _ = shape_assignment(U, R[sel])
        ref_d = base.loc[sel, [f'd{f}' for f in c['fish']]].to_numpy()
        ref_u = base.loc[sel, [f'u{f}_raw_p{q}' for f in c['fish'] for q in pair_ids]].to_numpy().reshape(U.shape)
        ok = np.isfinite(ref_d)
        print(f'{a.rec} check on {len(sel)} EODs: max |u - baseline u| / max|u| = '
              f'{np.nanmax(np.abs(U - ref_u) / np.abs(ref_u).max(axis=-1, keepdims=True)):.1e}; max |d - baseline d| = '
              f'{np.abs(D[ok] - ref_d[ok]).max():.1e}; same NaNs {(np.isfinite(D) == ok).all()}; '
              f"same pick {(pick == base.pick.to_numpy()[sel]).sum()} / {len(sel)}", flush=True)
        raise SystemExit

    versions = parse_versions(a.versions); bsig = None if a.body_sigmas is None else [float(x) for x in a.body_sigmas.split(',')]
    if a.interior_sigmas is not None:
        bsig = [('int', float(x)) for x in a.interior_sigmas.split(',')]
        if a.skin_lambdas is not None: bsig = [b + (float(l),) for b in bsig for l in a.skin_lambdas.split(',')]
    C = context(a.rec, a.N, versions, a.geometry, a.bodies == 'yes', a.masks, bsig); pair_ids = [pid for pid, _, _ in C['pairs']]
    if bsig is not None: versions = [None] * len(bsig)
    C['mids'] = load_midlines(a.rec, a.masks); n_mids0 = len(C['mids'])
    if a.midlines_only:
        frames = np.unique(pd.read_csv(c['newest'], usecols=['Frame']).Frame.to_numpy())
        todo = [int(k) for k in frames if int(k) not in C['mids']]; t0 = time.time()
        print(f'{a.rec}: {len(frames)} EOD frames, {n_mids0} midline frames on file, {len(todo)} to fit', flush=True)
        with mp.get_context('fork').Pool(a.workers) as pool:
            for k, M in pool.imap_unordered(midlines_only, todo, chunksize=16): C['mids'][k] = M
        save_midlines(a.rec, C['mids'], C['P'], a.masks)
        n_none = sum((~np.isfinite(C['mids'][int(k)]).all(axis=(1, 2))).sum() for k in frames)
        print(f"{a.rec}: midlines of {len(frames)} frames in {midlines_path(a.rec, a.masks)} ({n_none} fish-frames without a midline), "
              f'{(time.time() - t0) / 60:.1f} min', flush=True)
        raise SystemExit
    frames, R = recorded_patterns(c['newest'], pair_ids); rows = np.arange(len(frames))
    if a.eods:
        sel = np.sort(np.random.default_rng(0).choice(len(frames), a.eods, replace=False))
        frames, R, rows = frames[sel], R[sel], rows[sel]
    suffix = f'_random{a.eods}' if a.eods else ''
    tags = [version_tag(a.N, a.geometry, v, a.bodies == 'yes', s) + suffix for v, s in zip(versions, C['body'])]
    out = a.out or (c['out'] if a.masks == 'zeus' else c['out_kaden']); os.makedirs(out, exist_ok=True)
    cache_path = f'{out}/model_frames_N{a.N}_{a.geometry}_{"+".join(t.split("_", 2)[2] for t in tags)}.pkl'
    done = pd.read_pickle(cache_path) if os.path.exists(cache_path) else {'__versions__': tags}
    assert done['__versions__'] == tags, f'{cache_path} holds versions {done["__versions__"]}'
    todo = [int(k) for k in np.unique(frames) if int(k) not in done]
    print(f'{a.rec}: {len(frames)} EODs on {len(np.unique(frames))} frames, versions {tags}; {len(done) - 1} frames cached, '
          f'{len(todo)} to solve ({a.workers} workers)', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(a.workers) as pool:
        for n, (k, u, M) in enumerate(pool.imap_unordered(solve_frame, todo, chunksize=4), 1):
            done[k] = u; C['mids'][k] = M
            if n % 500 == 0 or n == len(todo):
                pd.to_pickle(done, cache_path + '.tmp'); os.replace(cache_path + '.tmp', cache_path); el = time.time() - t0
                print(f'  {n}/{len(todo)} frames, {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left', flush=True)
    if len(C['mids']) > n_mids0: save_midlines(a.rec, C['mids'], C['P'], a.masks)
    UU = np.stack([done[int(k)] for k in frames], axis=1)            # (n_versions, n_eods, n_fish, n_pairs)
    for j, (v, tag, s_body) in enumerate(zip(versions, tags, C['body'])):
        U = UU[j]; D, pick, Cs, pick_cos = shape_assignment(U, R); d = f'{out}/{tag}'; os.makedirs(d, exist_ok=True)
        result_table(frames, R, U, D, pick, pair_ids, c['fish'], rows, Cs, pick_cos).to_csv(f'{d}/eod_assignment_shape.csv', index=False)
        cfg = dict(rec=a.rec, eod_table=c['newest'], geometry=a.geometry,
                   tank_outline=None if a.geometry == 'image' else c['outline'].replace('updated', 'enlarged' if a.geometry == 'tank_enlarged' else 'updated'),
                   electrodes_file=C['electrodes_file'], px_per_cm=C['px_per_cm'], N=a.N, cell_cm=cell_cm(a.N, C['P'], C['px_per_cm']),
                   sigma_water=SIGMA_WATER, sigma_interior=SIGMA_INTERIOR, sigma_skin=SIGMA_SKIN, skin_cm=SKIN_CM,
                   skin_lambda_cm=SKIN_LAMBDA_CM, sink_frac=SINK, model_sign=MODEL_SIGN,
                   electrode_radius_cm=None if v is None else v[0], electrode_sigma=None if v is None else v[1],
                   electrode_readout='potential at the electrode centre', n_eods=len(frames), random_eods_seed0=a.eods or None,
                   rule='shape only: pick = smallest L2 of MODEL_SIGN u / max|u| vs r / max|r|; pick_cosine = largest cosine of '
                        'MODEL_SIGN u vs r', other_fish_bodies=a.bodies == 'yes', masks=a.masks,
                   other_fish_body='none' if a.bodies == 'no' else ('skin + interior 100' if s_body is None else
                       (f'skin ({s_body[2] if len(s_body) > 2 else SKIN_LAMBDA_CM:g} cm of water) + interior {s_body[1]:g} x water'
                        if isinstance(s_body, tuple) else f'uniform {s_body:g} x water, no skin')),
                   n_no_model={f'fish {f}': int((~np.isfinite(U[:, f]).all(axis=1)).sum()) for f in c['fish']})
        with open(f'{d}/config.json', 'w') as fh: json.dump(cfg, fh, indent=1)
        print(f"{a.rec} {tag}: % correct {(pick == c['emitter']).mean() * 100:.2f} (L2), {(pick_cos == c['emitter']).mean() * 100:.2f} "
              f"(cosine) on {len(pick)} EODs (no pick counts as wrong)",
              flush=True)
    print(f'{a.rec}: done, {(time.time() - t0) / 60:.1f} min', flush=True)
