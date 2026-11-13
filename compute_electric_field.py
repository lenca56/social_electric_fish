import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.ndimage import zoom
from scipy.interpolate import UnivariateSpline

# grid utils

def flatten_2d_grid(i, j, N):
    """
    Flatten a 2D grid coordinate to its 1D index (row-major), for the length-N*N vectors
       the PDE solver works on (build_A_shell, solve_potential).
       i, j: int, grid row/col, 0 <= i,j < N.  N: int, grid resolution (tank is N x N).
       Returns: int in [0, N*N) -- the flat index i*N + j.
    """
    if not (0 <= i < N and 0 <= j < N):
        raise IndexError(f"idx: i={i}, j={j} out of range for N={N}")
    return i * N + j

def unflatten_to_grid(v, N):
    """
    Inverse of idx: reshape a flat length-N*N vector back into an (N,N) grid for plotting
       or further array ops (e.g. np.gradient for the E-field).
       v: array-like, length N*N (e.g. the potential vector returned by solve_potential).
       N: int, grid resolution.
       Returns: np.ndarray, shape (N,N).
    """
    return np.asarray(v).reshape(N, N)

def mask_to_grid(m, N):
    """
    reshaping from video tracking mask to grid space
    takes in (P,P) bool mask m (corresponding to a fish's body in pixels of video) 
    -> (N,N) bool on the tank grid (nearest-neighbour, stays binary).
    """
    P = m.shape[0]
    if P == N: return m.astype(bool)
    return zoom(m.astype(float), N / P, order=0)[:N, :N] > 0.5

# electric field

SINK_FRAC = 0.05                   # sink position along the midline, fraction of body length from the tail tip

# materials (relative to water) and grid geometry, the single source for every run that imports them

VIDEO_PX = 912                     # video frames are VIDEO_PX x VIDEO_PX pixels
PX_PER_CM_DENIS = VIDEO_PX / 60    # Denis' pixel2cm (whole frame = 60 cm); still used by the reporting code (eod_examples)
PX_PER_CM = 14.2                   # model scale from 2026-11-10: the 60 x 60 cm tank spans ~852 px inside its walls
                                   # (tank_outline_updated_*.csv: 14.21 px/cm 2-fish, 14.12 3-fish)
# BEM conductivities, from the user's G. petersii BEM files: CondMedio 1e-4 S/cm (water), CondCuerpo 0.01 S/cm (interior),
# CondPiel 1e-3 S/cm^2 (skin, per area).
SIGMA_WATER = 1.0
SIGMA_INTERIOR = 100.0             # CondCuerpo / CondMedio
SKIN_CM = 0.005                    # skin thickness; only delta_h / sigma_skin matters (thin sheet)
SKIN_LAMBDA_CM = 0.1               # CondMedio / CondPiel: the skin is as resistive as 0.1 cm of water
SIGMA_SKIN = 1 / (SKIN_LAMBDA_CM / SKIN_CM + 1 / SIGMA_INTERIOR)        # 0.04998: sheet of SKIN_CM replacing interior

def cell_cm(N, P=VIDEO_PX, px_per_cm=PX_PER_CM):
    """Grid spacing in cm of an N x N grid spanning the P x P video frame."""
    return (P - 1) / (N - 1) / px_per_cm

def materials(N, P=VIDEO_PX, px_per_cm=PX_PER_CM):
    """Keyword arguments of build_A_shell for the BEM materials on an N x N grid: g_water, g_skin, g_interior, delta_h (cells)."""
    return dict(g_water=SIGMA_WATER, g_skin=SIGMA_SKIN, g_interior=SIGMA_INTERIOR, delta_h=SKIN_CM / cell_cm(N, P, px_per_cm))

def tank_inside(outline_px, N, P=VIDEO_PX):
    """(N, N) bool: grid nodes inside the tank outline (vertices (x, y) in video px, x = column, y = row; e.g.
       data/electrodes/tank_outline_updated_*.csv). Nodes outside are cut off by the non-conductive walls (staircase on the grid)."""
    from matplotlib.path import Path as MPath
    g = np.arange(N) / (N - 1) * (P - 1); R, Cc = np.meshgrid(g, g, indexing='ij')
    return MPath(np.asarray(outline_px, float)).contains_points(np.c_[Cc.ravel(), R.ravel()]).reshape(N, N)

def confine_sources(b, inside):
    """Move injected current that falls on nodes outside the tank (a midline point or its bilinear spread past a wall) to the
       nearest inside node, so no current is lost into the walls."""
    from scipy.ndimage import distance_transform_edt
    out = ~inside & (b != 0)
    if not out.any(): return b
    _, (ii, jj) = distance_transform_edt(~inside, return_indices=True)
    b = b.copy(); np.add.at(b, (ii[out], jj[out]), b[out]); b[out] = 0
    return b

def multipole_sources(midline, I0=1.0, exp_length_scale=0, L=1, sink_frac=SINK_FRAC):
    """Point currents along a fish midline.

    midline : (n+1, 2) midline points, tail -> head, evenly spaced by arc length. Use the slice-spline midline
              with n_ctrl = n + 1:
                  body = fish_midline_with_head(mask, n_ctrl=n + 1)          # area rule decides the head
                  midline = oriented_midline(body, head_is_a)                # head side after heads_with_continuity
              (or body['midline'] when the area rule alone decides). The points are used as they are, so the
              poles are body_length / n apart and a longer fish's currents cover more of the tank.
    Currents: the sink of -A sits on the midline point sink_frac of the body length from the tail tip (point
              round(sink_frac * n), 5% by default, about where the SLEAP tail keypoint lies); each of the m points between
              it and the head is a source of +A/m, and the points behind the sink carry no current, so the currents sum
              to zero. A = I0 * L**exp_length_scale. sink_frac = 0 puts the sink on the tail tip (the earlier model).
    L : body length for the amplitude scaling. Default None = the midline's own arc length, in the midline's
        units ([0, 1] tank coordinates for the functions above). With exp_length_scale = 0 (default) the
        amplitude does not depend on L; use this when I0 is a recorded voltage, which already contains any
        real size dependence.
    Returns pos (n+1, 2) and currents (n+1,)."""
    pos = np.asarray(midline, float)
    n = len(pos) - 1
    k = int(round(sink_frac * n))
    m = np.zeros(n + 1); m[k] = -1.0; m[k + 1:] = 1.0 / (n - k)
    return pos, m * I0 * L ** exp_length_scale


# TO CHECK THE FOLLOWING FUNCTIONS ARE CORRECT !!!!


def occupancy_field(masks, N, skip=None):
    """(n_fish,P,P) bool -> smooth NxN occupancy in [0,1]; occ > 0.5 = inside a fish.
       Linear interpolation keeps fractional values at the boundary. `skip` excludes one fish
       (the emitter) so it's transparent to water."""
    occ = np.zeros((N, N))
    for f, m in enumerate(masks):
        if f == skip or not np.any(m): continue
        P = m.shape[0]
        occ = np.maximum(occ, zoom(m.astype(float), N / P, order=1)[:N, :N])
    return occ

def theta_outer(occ, level=0.5):
    """Fractional position of the water|fish boundary on each grid edge (NaN = uncut)."""
    def cross(a, b):
        cut = (a > level) != (b > level)                    # same side test as gnode (occ > level)
        d = a - b
        th = np.divide(a - level, d, out=np.full_like(a, np.nan), where=np.abs(d) > 1e-12)
        return np.where(cut, np.clip(th, 1e-6, 1 - 1e-6), np.nan)
    return cross(occ[:-1, :], occ[1:, :]), cross(occ[:, :-1], occ[:, 1:])

def _edge_weight(gC, gM, thc, delta_h, g_skin, g_interior):
    """Effective admittivity across one grid edge: C-side medium | skin sheet | M-side medium in SERIES.
       thc = water|fish crossing measured from node C; delta_h = skin thickness in cell units.
       The skin is a thin sheet at the crossing that replaces delta_h of interior (contact impedance
       delta_h*(1/g_skin - 1/g_interior)), so it is never cut short by the next grid node, whatever
       thc is, and g_skin = g_interior = g_water gives plain water exactly."""
    if np.isnan(thc):
        return gC                                           # uncut edge: both nodes in the same medium
    return 1.0 / (thc / gC + (1 - thc) / gM + delta_h * (1 / g_skin - 1 / g_interior))

def electrode_occupancy(points, N, radius_cm, P=VIDEO_PX, px_per_cm=PX_PER_CM):
    """Recording electrodes as conducting discs of radius radius_cm on the N x N grid (2D: a rod through the water column).
       points: (n, 2) electrode centres in tank coordinates (row, col) / (P - 1), e.g. electrode_points(pairs).
       Returns a level set occ_e = 0.5 + (R - d) / (4 h), clipped to [0, 1] (max over the electrodes), with d the distance to the
       nearest centre: occ_e > 0.5 inside a disc, and occ_e is linear in d across every grid edge that the rim cuts, so
       theta_outer(occ_e) puts the water|electrode crossing where d = R (sub-cell, like the fish outline)."""
    h = 1.0 / (N - 1); R = radius_cm * px_per_cm / (P - 1)
    g = np.arange(N) * h; X, Y = np.meshgrid(g, g, indexing='ij')
    occ_e = np.zeros((N, N))
    for p in np.asarray(points, float):
        occ_e = np.maximum(occ_e, np.clip(0.5 + (R - np.hypot(X - p[0], Y - p[1])) / (4 * h), 0, 1))
    return occ_e

def electrode_points(pairs):
    """(2 n_pairs, 2) electrode centres (tank coordinates) of load_electrode_pairs output, + and - of each pair."""
    return np.array([p for _, pp, pm in pairs for p in (pp, pm)])

def build_A_shell(occ, N, delta_h, g_water, g_skin, g_interior, electrodes=None, inside=None):
    """Complex system matrix for div(gamma grad phi) = -s, with the skin as a thin sheet at the actual occupancy boundary (not
       snapped to the grid). Vectorised; each edge weight is _edge_weight (the C-side | skin sheet | M-side series rule).
       electrodes: None, or (occ_e, g_electrode) with occ_e from electrode_occupancy. Nodes inside a disc get g_electrode
       (they win over a fish mask); an edge the disc rim cuts is water (or fish) | electrode in series at the crossing, with no
       skin or contact layer; g_electrode = g_water gives the matrix without electrodes (where no fish overlaps a disc).
       inside: None (walls at the grid edge), or (N, N) bool from tank_inside: every edge touching a node outside the tank gets
       zero conductance (non-conductive walls); outside nodes are left with empty rows (solve_potential handles them)."""
    gnode = np.where(occ > 0.5, g_interior, g_water).astype(complex)
    tx, ty = theta_outer(occ)
    if electrodes is not None:
        occ_e, g_e = electrodes
        ins_e = occ_e > 0.5
        gnode = np.where(ins_e, g_e, gnode)
        ex_, ey_ = theta_outer(occ_e)
    skin = delta_h * (1 / g_skin - 1 / g_interior)
    k = np.arange(N * N).reshape(N, N); r, c, v = [], [], []
    for d, (a, b) in enumerate(((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:]))):
        th = (tx, ty)[d]; gA, gB = gnode[a], gnode[b]                    # th: crossing measured from node a
        with np.errstate(invalid='ignore'):
            w = np.where(np.isnan(th), gA, 1.0 / (th / gA + (1 - th) / gB + skin))
            if electrodes is not None:
                the = (ex_, ey_)[d]
                w = np.where(ins_e[a] & ins_e[b], gA, w)                 # inside one disc
                w = np.where(np.isnan(the), w, 1.0 / (the / gA + (1 - the) / gB))
        if inside is not None: w = np.where(inside[a] & inside[b], w, 0)  # walls: no current through them
        r += [k[a].ravel(), k[b].ravel()]; c += [k[b].ravel(), k[a].ravel()]; v += [w.ravel()] * 2
    A = sp.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(N * N, N * N), dtype=complex).tocsr()
    return (A - sp.diags(np.asarray(A.sum(axis=1)).ravel())).tocsr()

def potential_solver(A, pin=0):
    """Factorise A once and return solve(b) -> phi, for many current distributions b on the same medium (e.g. every fish and frame
       when the other fish are not modelled). Same result as solve_potential(A, b, pin)."""
    M = (-A).tocsr()
    cut = M.diagonal() == 0
    if cut.any():
        if cut[pin]: pin = int(np.flatnonzero(~cut)[0])
        M = M + sp.diags(cut.astype(float))                               # identity rows: phi = 0 there, removed below
    real = not np.any(M.data.imag)
    if real: M = M.real
    keep = np.ones(M.shape[0]); keep[pin] = 0
    lu = splu((sp.diags(keep) @ M + sp.csr_matrix(([1.0], ([pin], [pin])), shape=M.shape)).tocsc())
    def solve(b):
        rhs = np.asarray(b).ravel()
        if cut.any(): assert not np.any(rhs[cut]), 'current injected on nodes cut off by the walls (use confine_sources)'
        if real and np.any(np.imag(rhs)):                                 # complex currents on a real medium: solve both parts
            phi = solve(np.real(rhs)) + 1j * solve(np.imag(rhs)); return phi
        rhs = (np.real(rhs).astype(float) if real else rhs.astype(complex)).copy(); rhs[pin] = 0
        phi = lu.solve(rhs)
        if cut.any():
            phi = phi - phi[~cut].mean(); phi[cut] = np.nan; return phi
        return phi - phi.mean()
    return solve

def solve_potential(A, b, pin=0):
    """Solve A phi = -b for injected currents b (singular Neumann system), so phi is highest at current sources; pin one node
       to fix the gauge; zero-mean result. Needs sum(b) == 0. Real arithmetic when A and b have no imaginary part (omega = 0).
       Nodes cut off by walls (empty rows of A, see build_A_shell inside=) get phi = NaN and must carry no current; the pin then
       moves to the first connected node and the mean is taken over the connected nodes."""
    return potential_solver(A, pin)(b)
# materials, sources, sampling

def gammas(omega, **mat):
    """Complex admittivity gamma = sigma + i*omega*eps for water, skin, interior.
       mat: sigma_water, eps_water, sigma_skin, eps_skin, sigma_interior, eps_interior."""
    gw = mat['sigma_water']    + 1j*omega*mat['eps_water']
    gs = mat['sigma_skin']     + 1j*omega*mat['eps_skin']
    gi = mat['sigma_interior'] + 1j*omega*mat['eps_interior']
    return gw, gs, gi

def place_current_on_grid(b, x, y, I, h, N):
    """Add point current I at tank point (x, y) to the (N,N) source grid b, spread over the
       4 surrounding nodes by bilinear weights."""
    fi, fj = x/h, y/h; i, j = int(fi), int(fj); a, c = fi-i, fj-j
    for di, wx in ((0, 1-a), (1, a)):
        for dj, wy in ((0, 1-c), (1, c)):
            ii, jj = i+di, j+dj
            if 0 <= ii < N and 0 <= jj < N: b[ii, jj] += I*wx*wy

def build_multipole_b(pos, cur, h, N):
    """(N,N) injected-current grid for point currents cur at positions pos (from multipole_sources)."""
    b = np.zeros((N, N))
    for (x, y), Ii in zip(pos, cur): place_current_on_grid(b, x, y, Ii, h, N)
    return b - b.mean()            # enforce net current = 0 (required for the Neumann solve)

def sample_phi_point(phi_grid, p, h, N):
    """Bilinear sample of an (N,N) field (or (..., N, N) stack) at tank point p = (x, y)."""
    fi, fj = p[0] / h, p[1] / h
    i, j = int(fi), int(fj); a, c = fi - i, fj - j
    def w(ii, jj): return phi_grid[..., min(max(ii, 0), N - 1), min(max(jj, 0), N - 1)]
    return (1-a)*(1-c)*w(i,j) + a*(1-c)*w(i+1,j) + (1-a)*c*w(i,j+1) + a*c*w(i+1,j+1)
