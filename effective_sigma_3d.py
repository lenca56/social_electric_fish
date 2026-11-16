"""Effective 2D conductivity of a non-emitting fish: what uniform conductivity should a fish column have in the depth-uniform 2D
model so that it disturbs the electrode voltages as a real 3D fish (interior 100 x water + resistive skin) does, given the water
depth and the height difference between the emitter and that fish.

3D: rectangular tank 60 x 60 cm, water depth H (argument, default 20 cm; the real water level is 12-15 cm), insulating walls, floor and surface; graded finite-volume grid, fine around
the fish. Other fish: ellipsoid (length L, width W, body depth D) centred at mid-depth, axis along x, interior SIGMA_INT x water, skin
as resistive as `lam` cm of water (series resistance on every face that crosses the body surface). Emitter: transparent, current +1
spread along its front 90 % and -1 at its tail tip (as the 2D model's sources), at height z_fish + dz, at lateral distance d, either
parallel to the other fish (side by side: field along the other fish's axis) or perpendicular, its head pointing at the other fish's
middle (field across it). Readout: depth-averaged potential at 20 points 2 cm inside the walls (vertical electrodes).
2D: the same x-y grid, sheet of thickness H, the fish as its top-view ellipse (L x W) with a uniform conductivity s, no skin.
For each 3D case, s_eff = the s whose 2D electrode perturbation (fish minus no fish, mean removed) is closest (least squares) to the
3D one (s scanned 0.02-100); also the relative misfit at s_eff.
The 3D electrode readouts are cached in perturbations_3d_H<H>.npz (delete it after changing the 3D setup).

    python effective_sigma_3d.py
    python effective_sigma_3d.py 12        (water depth in cm)
Output in results/effective_sigma_3d/: effective_sigma_H<H>.csv, effective_sigma_H<H>.png and the 3D cache perturbations_3d_H<H>.npz.
Height differences dz only up to 1 cm below the surface (fish at mid-depth).
"""
import sys, time, itertools, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
sys.path.append('/private/tmp/claude-501/-Volumes-Lenca-SSD-github-social-electric-fish/df83191d-1e3e-4c5e-939e-988975f8b577/scratchpad/pylib')
import pyamg

TANK = 60.0                                # cm
H = float(sys.argv[1]) if len(sys.argv) > 1 and __name__ == '__main__' else 20.0   # water depth (cm)
SIGMA_INT = 100.0
L, W = 14.5, 1.6                           # typical Kaden-mask length; top-view max width (4/pi x mean mask width 1.0-1.4 cm)
DEPTH_FRAC = 0.25                          # body depth / length (Gnathonemus petersii 22-29 % SL)
C0 = np.array([TANK / 2, TANK / 2, H / 2])  # other fish centre
OUT = 'results/effective_sigma_3d'; SUF = f'_H{H:g}'


def axis(lo, hi, fine_lo, fine_hi, h_fine, h_max, growth=0.12):
    """Cell edges on [lo, hi]: h_fine inside [fine_lo, fine_hi], growing linearly with distance outside, at most h_max."""
    def step(x): return min(h_max, h_fine + growth * max(0.0, fine_lo - x, x - fine_hi))
    e = [fine_lo]; x = fine_lo
    while x < hi - 1e-9: x = min(hi, x + step(x)); e.append(x)
    x = fine_lo; left = []
    while x > lo + 1e-9: x = max(lo, x - step(x)); left.append(x)
    e = np.array(left[::-1] + e)
    return e


def grid(nz=True):
    ex = axis(0, TANK, C0[0] - 12, C0[0] + 12, 0.3, 1.0)
    ey = axis(0, TANK, C0[1] - 1.5, C0[1] + 1.5, 0.12, 1.0, growth=0.08)
    ey = np.unique(np.r_[ey[ey < C0[1] - 1.5], np.arange(C0[1] - 1.5, C0[1] + 1.5, 0.12), ey[ey >= C0[1] + 1.5]])
    ez = axis(0, H, C0[2] - 2.5, C0[2] + 2.5, 0.25, 1.0) if nz else np.array([0.0, H])
    return ex, ey, ez


def laplacian(ex, ey, ez, sig, inside, lam):
    """Finite-volume conductance matrix; sig, inside: (nx, ny, nz) cell arrays; lam: skin (cm of water) on faces crossing the body."""
    dx, dy, dz = np.diff(ex), np.diff(ey), np.diff(ez); n = sig.shape; idx = np.arange(sig.size).reshape(n)
    rows, cols, vals = [], [], []
    for ax, d in enumerate((dx, dy, dz)):
        if n[ax] < 2: continue
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3; sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None); sl0, sl1 = tuple(sl0), tuple(sl1)
        shp = [1, 1, 1]; shp[ax] = -1
        h0, h1 = d[:-1].reshape(shp), d[1:].reshape(shp)
        others = [dd for a_, dd in enumerate((dx, dy, dz)) if a_ != ax]
        area = np.multiply.outer(others[0], others[1])
        area = np.expand_dims(area, ax)
        R = h0 / (2 * sig[sl0]) + h1 / (2 * sig[sl1]) + lam * (inside[sl0] != inside[sl1])
        G = (area / R).ravel(); i0, i1 = idx[sl0].ravel(), idx[sl1].ravel()
        rows += [i0, i1, i0, i1]; cols += [i1, i0, i0, i1]; vals += [-G, -G, G, G]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(sig.size, sig.size))
    A = A + sp.csr_matrix(([A.diagonal().mean() * 1e-6], ([0], [0])), shape=A.shape)      # ground one cell (insulated tank)
    return A


def cell_index(e, x): return np.clip(np.searchsorted(e, x) - 1, 0, len(e) - 2)


def emitter_rhs(ex, ey, ez, orient, d, dz, n_src=60):
    """Right-hand side: +1 along the emitter's front 90 %, -1 at its tail tip, at height C0z + dz (2D: ez has one cell)."""
    s = np.linspace(0, 1, n_src)                                # tail (0) -> head (1)
    if orient == 'parallel':   # side by side, same direction, lateral distance d
        p = np.c_[C0[0] - L / 2 + s * L, np.full(n_src, C0[1] + d)]
    else:                      # perpendicular, head pointing at the other fish's middle from distance d
        p = np.c_[np.full(n_src, C0[0]), C0[1] + d + (1 - s) * L]
    I = np.where(s > 0.1, 1.0 / (s > 0.1).sum(), 0.0); I[0] = -1.0
    b = np.zeros((len(ex) - 1, len(ey) - 1, len(ez) - 1))
    k = cell_index(ez, C0[2] + dz)
    np.add.at(b, (cell_index(ex, p[:, 0]), cell_index(ey, p[:, 1]), k), I)
    return b.ravel()


def electrodes(ex, ey):
    t = np.linspace(0, 4, 20, endpoint=False); m = 2.0; a = TANK - 2 * m
    xy = np.array([(m + a * f, m) if s == 0 else (TANK - m, m + a * f) if s == 1 else (TANK - m - a * f, TANK - m) if s == 2
                   else (m, TANK - m - a * f) for s, f in zip(t.astype(int), t % 1)])
    return cell_index(ex, xy[:, 0]), cell_index(ey, xy[:, 1])


def readout(phi, shape, ez, eidx):
    """Depth-averaged potential at the electrodes, mean removed."""
    P = phi.reshape(shape); w = np.diff(ez) / H
    v = (P[eidx[0], eidx[1], :] * w).sum(-1); return v - v.mean()


def solve_many(A, B, tol=1e-10):
    ml = pyamg.smoothed_aggregation_solver(A, symmetry='hermitian', max_coarse=500)
    return [ml.solve(b, tol=tol, accel='cg', maxiter=400) for b in B]


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); t0 = time.time()
    D = DEPTH_FRAC * L
    ex, ey, ez = grid(); cx, cy, cz = [(e[1:] + e[:-1]) / 2 for e in (ex, ey, ez)]
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij'); shape3 = X.shape
    ell = ((X - C0[0]) / (L / 2)) ** 2 + ((Y - C0[1]) / (W / 2)) ** 2 + ((Z - C0[2]) / (D / 2)) ** 2 <= 1
    eidx = electrodes(ex, ey)
    print(f'3D grid {shape3} = {X.size:,} cells; fish L {L} W {W} D {D:.2f} cm, {ell.sum()} cells', flush=True)
    cases = [(o, d, dz) for o in ('parallel', 'perpendicular') for d in (4.0, 8.0, 15.0) for dz in (0.0, 2.0, 4.0, 6.0, 8.0)
             if C0[2] + dz <= H - 1.0]                       # emitter at least 1 cm below the surface
    B3 = [emitter_rhs(ex, ey, ez, o, d, dz) for o, d, dz in cases]
    LAMS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0); cache = f'{OUT}/perturbations_3d{SUF}.npz'
    if os.path.exists(cache):                                   # 3D electrode readouts of an earlier run with the same setup
        z = np.load(cache, allow_pickle=True); V0 = list(z['V0']); dV3 = z['dV3'].item()
        assert list(map(tuple, z['cases'])) == [tuple(map(str, c)) for c in cases] and tuple(z['lams']) == LAMS
    else:
        V0 = [readout(p, shape3, ez, eidx) for p in solve_many(laplacian(ex, ey, ez, np.ones(shape3), np.zeros(shape3, bool), 0.0), B3)]
        print(f'  no-fish 3D done {(time.time() - t0) / 60:.1f} min', flush=True)
        dV3 = {}
        for lam in LAMS:
            sig = np.where(ell, SIGMA_INT, 1.0)
            for c, p in zip(cases, solve_many(laplacian(ex, ey, ez, sig, ell, lam), B3)):
                dV3[(lam,) + c] = readout(p, shape3, ez, eidx) - V0[cases.index(c)]
            print(f'  3D skin {lam} done {(time.time() - t0) / 60:.1f} min', flush=True)
        np.savez(cache, V0=np.array(V0), dV3=np.array(dV3, dtype=object), cases=np.array([tuple(map(str, c)) for c in cases]), lams=np.array(LAMS))

    # 2D: sheet of thickness H on the same x-y grid, fish = top-view ellipse, uniform s, no skin
    ez2 = np.array([0.0, H]); X2, Y2 = np.meshgrid(cx, cy, indexing='ij'); shape2 = X2.shape + (1,)
    ell2 = (((X2 - C0[0]) / (L / 2)) ** 2 + ((Y2 - C0[1]) / (W / 2)) ** 2 <= 1)[..., None]
    pl = sorted(set((o, d) for o, d, _ in cases)); B2 = {od: emitter_rhs(ex, ey, ez2, od[0], od[1], 0.0) for od in pl}
    lu0 = spla.splu(laplacian(ex, ey, ez2, np.ones(shape2), np.zeros(shape2, bool), 0.0).tocsc())
    V20 = {od: readout(lu0.solve(b), shape2, ez2, eidx) for od, b in B2.items()}
    S = np.geomspace(0.02, 100.0, 181)
    dV2 = {od: [] for od in pl}
    for s in S:
        lu = spla.splu(laplacian(ex, ey, ez2, np.where(ell2, s, 1.0), ell2, 0.0).tocsc())
        for od, b in B2.items(): dV2[od].append(readout(lu.solve(b), shape2, ez2, eidx) - V20[od])
    # check: 3D no fish vs 2D no fish at the electrodes
    for od in pl:
        v3 = V0[cases.index(od + (0.0,))]; print(f'  check {od}: 3D vs 2D no-fish electrode potentials, rel. diff '
                                                 f'{np.linalg.norm(v3 - V20[od]) / np.linalg.norm(V20[od]):.3f}', flush=True)

    rows = []
    for (lam, o, d, dz), v3 in dV3.items():
        M = np.array(dV2[(o, d)]); err = np.linalg.norm(M - v3, axis=1); j = int(np.argmin(err))
        # refine on a fine log grid by linear interpolation of the 2D perturbation between grid points
        ss = np.geomspace(S[max(j - 1, 0)], S[min(j + 1, len(S) - 1)], 41); ls = np.log(S)
        Mi = np.array([[np.interp(np.log(x), ls, M[:, e]) for e in range(M.shape[1])] for x in ss])
        e2 = np.linalg.norm(Mi - v3, axis=1); k = int(np.argmin(e2))
        rows.append(dict(skin_lambda_cm=lam, orientation=o, distance_cm=d, dz_cm=dz, sigma_eff=ss[k],
                         misfit_rel=e2[k] / np.linalg.norm(v3), fish_effect_rel=np.linalg.norm(v3) / np.linalg.norm(V0[cases.index((o, d, dz))]),
                         at_s100_rel=err[-1] / np.linalg.norm(v3)))
    T = pd.DataFrame(rows); T.to_csv(f'{OUT}/effective_sigma{SUF}.csv', index=False)
    with pd.option_context('display.width', 200):
        print(T.pivot_table(index=['orientation', 'distance_cm', 'dz_cm'], columns='skin_lambda_cm', values='sigma_eff').round(2).to_string())
        print(T.pivot_table(index=['orientation', 'distance_cm', 'dz_cm'], columns='skin_lambda_cm', values='misfit_rel').round(2).to_string())
        print('fish effect: ||dV|| / ||V|| at the electrodes (%)')
        print((100 * T.pivot_table(index=['orientation', 'distance_cm', 'dz_cm'], columns='skin_lambda_cm', values='fish_effect_rel')).round(2).to_string())
    print(f'done {(time.time() - t0) / 60:.1f} min')

    # figure: s_eff vs height difference, one panel per orientation x distance, one line per skin
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True)
    cols = dict(zip(LAMS, ['#1b2a49', '#2a78d6', '#3fa34d', '#eda100', '#eb6834', '#b0306a']))
    for r_, o in enumerate(('parallel', 'perpendicular')):
        for c_, d in enumerate((4.0, 8.0, 15.0)):
            a = axs[r_, c_]
            for lam in LAMS:
                s_ = T[(T.orientation == o) & (T.distance_cm == d) & (T.skin_lambda_cm == lam)].sort_values('dz_cm')
                a.plot(s_.dz_cm, s_.sigma_eff, 'o-', color=cols[lam], ms=4, label=f'skin {lam:g} cm of water')
            e0 = T[(T.orientation == o) & (T.distance_cm == d) & (T.skin_lambda_cm == 0.1) & (T.dz_cm == 0)].iloc[0]
            a.axhline(1, color='0.6', lw=0.8, ls=':')
            a.set_title(f"{o}, emitter {d:g} cm away\nfish changes the electrode voltages by {100 * e0.fish_effect_rel:.2g} % "
                        f"(skin 0.1, same height)", fontsize=9)
            a.set_yscale('log'); a.grid(alpha=0.3, which='both')
            if c_ == 0: a.set_ylabel('effective 2D conductivity (x water)')
            if r_ == 1: a.set_xlabel('height difference emitter - other fish (cm)')
            for s__ in ('top', 'right'): a.spines[s__].set_visible(False)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'3D fish (ellipsoid {L} x {W} x {D:.1f} cm, interior {SIGMA_INT:g} x water, skin) in {H:g} cm of water -> uniform 2D '
                 'column (top-view ellipse, no skin) giving the same depth-averaged electrode perturbation', fontsize=10.5)
    fig.text(0.5, 0.005, 'parallel: emitter side by side with the other fish (field along its body); perpendicular: emitter head pointing at '
             "its middle (field across it). Dotted: water.", ha='center', fontsize=8.5, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(f'{OUT}/effective_sigma{SUF}.png', dpi=130); plt.close(fig)
