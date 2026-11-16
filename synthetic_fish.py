"""Synthetic fish masks with a known midline (and optional pectoral fins matched to the real masks).

make_fish(theta_deg, bend, kind) -> (true midline (2000,2) px tail -> head, (P,P) bool mask). Used by the midline notebooks.
"""
import numpy as np

PS, Lf = 912, 180.0
# pectoral fins, in body lengths / degrees, from the real masks (median over fish 0-3, Midline_verification.ipynb §15b):
# 28-32 % of the length behind the snout, protrude ~0.09 L, ~0.035 L wide, 60-70 deg to the body, pointing back.
# length/width below are the drawn fin; they reproduce the real removed area (89 vs 91 px at L = 180) and protrusion
# (0.092 vs 0.089 L) after remove_fins. The two fins differ so their pulls on the midline don't cancel.
FINS = [dict(pos_from_head=0.30, length=0.12, width=0.045, angle=55, side=+1),
        dict(pos_from_head=0.31, length=0.10, width=0.045, angle=72, side=-1)]


def make_fish(theta_deg, bend, kind='C', center=(456.3, 456.7), P=PS, fins=()):
    s = np.linspace(0, Lf, 2000)
    if kind == 'C':
        kappa = np.deg2rad(bend) / Lf * np.ones_like(s)
    else:
        kappa = np.deg2rad(bend) / Lf * 2 * np.cos(2 * np.pi * s / Lf)
    ang = np.concatenate([[0], np.cumsum(kappa[:-1] * np.diff(s))]) + np.deg2rad(theta_deg)
    ds = np.diff(s, prepend=0)
    mid = np.column_stack([np.cumsum(np.cos(ang) * ds), np.cumsum(np.sin(ang) * ds)])
    mid += np.asarray(center) - mid.mean(0)
    u = s / Lf
    half = 1.5 + 9.0 * np.clip(u / 0.75, 0, 1) ** 1.2
    half = np.where(u > 0.85, half * np.sqrt(np.clip((1 - u) / 0.15, 0, 1)) + 1e-9, half)
    disks = list(zip(mid[::4], half[::4]))
    tan = np.column_stack([np.cos(ang), np.sin(ang)]); nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    for fin in fins:                                   # a fin = tapering chain of disks from the body edge outward
        i = int(round((1 - fin['pos_from_head']) * (len(s) - 1)))
        a = np.deg2rad(fin['angle'])
        d = np.cos(a) * -tan[i] + np.sin(a) * fin['side'] * nrm[i]       # backward, at `angle` to the body
        base = mid[i] + fin['side'] * nrm[i] * 0.8 * half[i]
        L_fin = fin['length'] * Lf + 0.2 * half[i]
        for f in np.linspace(0, 1, 60):
            disks.append((base + f * L_fin * d, 0.5 * fin['width'] * Lf * (1 - 0.5 * f)))
    pts = np.array([p for p, _ in disks])
    lo, hi = pts.min(0).astype(int) - 15, pts.max(0).astype(int) + 15
    gx, gy = np.mgrid[lo[0]:hi[0], lo[1]:hi[1]]
    sub = np.zeros(gx.shape, bool)
    for (mx, my), h in disks:
        sub |= (gx - mx) ** 2 + (gy - my) ** 2 <= h ** 2
    mask = np.zeros((P, P), bool); mask[lo[0]:hi[0], lo[1]:hi[1]] = sub
    return mid, mask
