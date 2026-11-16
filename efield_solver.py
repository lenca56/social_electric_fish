import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import inspect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- grid bookkeeping ----------------
def idx(i, j, N):  return i * N + j
def to_grid(v, N): return v.reshape(N, N)

# ---------------- operator: Neumann (insulating-wall) Laplacian ----------------
def build_A(N):
    """N^2 x N^2 sparse Laplacian, diag -(#neighbours): interior -4, edge -3,
       corner -2; +1 per real neighbour. Vectorised (== the explicit builder),
       so it scales to N = 1000 (1e6 unknowns)."""
    e = np.ones(N); main = -2*e.copy(); main[0] = -1; main[-1] = -1
    D1 = sp.diags([e[:-1], main, e[:-1]], [-1, 0, 1])
    I  = sp.identity(N)
    return (sp.kron(D1, I) + sp.kron(I, D1)).tocsr()

# ---------------- source deposition (cell -> <=4 nodes, bilinear) -------------
def place_current_on_grid(b, x, y, I, h, N):
    fi, fj = x/h, y/h; i, j = int(fi), int(fj); a, c = fi-i, fj-j
    for di, wx in ((0, 1-a), (1, a)):
        for dj, wy in ((0, 1-c), (1, c)):
            ii, jj = i+di, j+dj
            if 0 <= ii < N and 0 <= jj < N:
                b[ii, jj] += I * wx * wy

def sample_fish(rng, l):
    s = rng.uniform(0, 1, size=2)
    while True:
        th = rng.uniform(0, 2*np.pi)
        k = s + l*np.array([np.cos(th), np.sin(th)])
        if (k >= 0).all() and (k <= 1).all():
            return s, k

# ---------------- CG that works across scipy versions ----------------
def solve_potential(A, b, maxiter=20000):
    """Solve A phi = b for the singular Neumann system via CG on (-A) phi = -b."""
    rhs = -(b.ravel())
    it = [0]
    cb = lambda xk: it.__setitem__(0, it[0] + 1)
    kw = dict(maxiter=maxiter, callback=cb)
    if "rtol" in inspect.signature(cg).parameters:      # scipy >= 1.12
        kw.update(rtol=1e-8, atol=0.0)
    else:                                               # older scipy
        kw.update(tol=1e-8)
    phi, info = cg(-A, rhs, **kw)
    phi -= phi.mean()                                   # fix the constant (gauge)
    return phi, info, it[0]

# ---------------- plotting ----------------
def plot_potential(phi, source, sink, N, path):
    g = np.linspace(0, 1, N)
    fig, ax = plt.subplots(figsize=(6, 5))
    # first axis is x -> transpose so x is horizontal; origin lower so y points up
    im = ax.imshow(phi.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r")
    ax.contour(g, g, phi.T, levels=16, colors="k", linewidths=0.4, alpha=0.45)
    ax.plot(*source, "o", mfc="#C0392B", mec="k", ms=8, label="+ source (head)")
    ax.plot(*sink,   "s", mfc="#2C6FBB", mec="k", ms=8, label="- sink (tail)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"electric potential  φ   (N = {N})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.colorbar(im, ax=ax, fraction=0.046, label="φ")
    plt.tight_layout(); plt.savefig(path, dpi=130)

# ============================ run ============================
if __name__ == "__main__":
    # Choosing N: the dipole separation l must be resolved (l*N cells across the
    # fish; want >= ~20). Plain CG takes O(N) iterations of cheap sparse mat-vecs.
    #   N = 256-512 : fast, well resolved  -> good default for development
    #   N = 1000    : the target resolution -> add a preconditioner for speed
    N = 400
    h = 1.0 / (N - 1)
    l = 1/6
    I = 1.0

    rng = np.random.default_rng(0)
    source_loc, sink_loc = sample_fish(rng, l)

    b = np.zeros((N, N))
    place_current_on_grid(b, *source_loc,  I, h, N)
    place_current_on_grid(b, *sink_loc,   -I, h, N)
    b -= b.mean()                              # enforce net current = 0

    A = build_A(N)
    phi_vec, info, iters = solve_potential(A, b)
    phi = to_grid(phi_vec, N)

    resid = np.abs(A @ phi_vec - b.ravel()).max()
    print(f"N={N} ({N*N} unknowns)  CG info={info} (0=converged)  iters={iters}")
    print(f"max residual |A phi - b| = {resid:.2e}   phi range [{phi.min():.3f}, {phi.max():.3f}]")

    plot_potential(phi, source_loc, sink_loc, N, "/mnt/user-data/outputs/potential.png")
    print("saved /mnt/user-data/outputs/potential.png")
