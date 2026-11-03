"""Midline estimation for fish masks.

Functions developed and tested in Midline_verification.ipynb (sections 9, 12, 15-18) and used by the
five-way comparison notebooks. Coordinates are (row, col) pixels of a (P, P) mask; returned midlines are
(n, 2) in [0, 1] (pixel / (P - 1)), ordered tail -> head.

Approaches:
  mask-denis 8  fish_midline_mask_denis(mask, head, 8)
  denis         fish_midline_denis(keypoints tail/middle/head/mouth, shape)   (no mask)
  fin removal   remove_fins(mask, head) before any mask-based approach
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.ndimage import label, map_coordinates, median_filter
from scipy.interpolate import make_lsq_spline, BSpline


# ---------------------------------------------------------------- cross-sections
def _edges(F, P, N, half=40.0, res=0.25):
    '''Signed distances (lo < 0 < hi) from points P (k,2) along unit normals N (k,2) to the mask edge,
       staying in the inside run that contains each point. Sub-pixel: the edge is where the bilinear
       mask crosses 0.5 (= the pixel border). NaN where the point itself is outside the mask.'''
    o = np.arange(-half, half + res / 2, res); c = len(o) // 2; j = np.arange(len(o)); k = np.arange(len(P))
    xy = P[:, None, :] + o[None, :, None] * N[:, None, :]
    V = map_coordinates(F, xy.reshape(-1, 2).T, order=1, mode='constant').reshape(len(P), len(o))
    out = V < 0.5
    jl = np.where(out & (j < c), j, -1).max(1)            # last outside sample left of the point
    jr = np.where(out & (j > c), j, len(o)).min(1)        # first outside sample right of the point

    def cross(jo, ji):                                   # linear interpolation of V across 0.5
        joc = np.clip(jo, 0, len(o) - 1)
        vo, vi = V[k, joc], V[k, ji]
        with np.errstate(divide='ignore', invalid='ignore'):   # vi == vo only where jo is masked below
            e = o[joc] + (0.5 - vo) / (vi - vo) * (o[ji] - o[joc])
        return np.where((jo < 0) | (jo >= len(o)), o[ji], e)

    lo, hi = cross(jl, np.clip(jl + 1, 0, c)), cross(jr, np.clip(jr - 1, c, len(o) - 1))
    lo[out[:, c]] = np.nan; hi[out[:, c]] = np.nan
    return lo, hi

# ---------------------------------------------------------------- Isomap and Denis (EODAssignment port)
def _largest_component(mask):
    lab, n = label(mask, structure=np.ones((3, 3)))
    return mask if n <= 1 else lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def _orient(curve, head):
    '''tail -> head: flip if the first point is closer to the head point'''
    return curve[::-1] if np.linalg.norm(curve[0] - head) < np.linalg.norm(curve[-1] - head) else curve


# def _resample(pts, n):
#     s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
#     u = np.linspace(0, s[-1], n)
#     return np.column_stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])])


def _pixel_graph(mask):
    '''8-connected graph of mask pixels, edge weight = Euclidean step (1 or sqrt 2)'''
    idx = -np.ones(mask.shape, int); rr, cc = np.nonzero(mask); idx[rr, cc] = np.arange(len(rr))
    a, b, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 < mask.shape[0]) & (c2 >= 0) & (c2 < mask.shape[1])
        ok[ok] &= mask[r2[ok], c2[ok]]
        a.append(idx[rr[ok], cc[ok]]); b.append(idx[r2[ok], c2[ok]]); w.append(np.full(ok.sum(), np.hypot(dr, dc)))
    a, b, w = map(np.concatenate, (a, b, w))
    G = sp.csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(rr), len(rr)))
    return G, np.column_stack([rr, cc]).astype(float)

def _fb_interp(x, y, x_new):
    '''Vectorised port of EODAssignment src/field_model.py::_interpolate_points: piecewise cubic
       Hermite with Fritsch-Butland tangents (harmonic mean of neighbouring slopes, 0 at a local
       extremum and at the two ends), evaluated at x_new (constant outside [x0, x_end]).'''
    n = len(x); m = np.diff(y) / np.diff(x)                   # slope of each interval
    tan = np.zeros(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        inner = 2 * m[:-1] * m[1:] / (m[:-1] + m[1:])
    tan[1:-1] = np.where(m[:-1] * m[1:] > 0, inner, 0.0)
    i = np.clip(np.searchsorted(x, x_new, side='right') - 1, 0, n - 2)
    h = x[i + 1] - x[i]; t = (x_new - x[i]) / h
    out = ((2 * t**3 - 3 * t**2 + 1) * y[i] + (t**3 - 2 * t**2 + t) * h * tan[i]
           + (-2 * t**3 + 3 * t**2) * y[i + 1] + (t**3 - t**2) * h * tan[i + 1])
    return np.where(x_new <= x[0], y[0], np.where(x_new >= x[-1], y[-1], out))


def fish_midline_denis(keypoints, shape, n_ctrl=200, uniformity_multiplier=100):
    '''Port of EODAssignment estimate_point_currents_location for one fish: keypoints (4,2) =
       tail, middle, head, mouth; interpolated per coordinate at uniform parameter t = 0, 1/3, 2/3, 1;
       oversampled to n_ctrl*uniformity_multiplier points; n_ctrl points picked evenly by arc length
       (each = the oversampled point just before the target length). No mask is used; `shape` is the
       mask shape, only for the [0,1] scaling. Returns (n_ctrl,2) in [0,1], tail -> mouth.'''
    kp = np.asarray(keypoints, float); t = np.linspace(0, 1, len(kp))
    ts = np.linspace(0, 1, n_ctrl * uniformity_multiplier)
    over = np.column_stack([_fb_interp(t, kp[:, d], ts) for d in range(2)])
    dx, dy = np.diff(over[:, 0]), np.diff(over[:, 1])
    seg = np.sqrt(dx * dx + dy * dy)                          # same arithmetic as the original, so that
    cum = np.r_[0, np.cumsum(seg / seg.sum())]                # floating-point rounding matches too
    idx = np.searchsorted(cum, np.linspace(0, 1, n_ctrl), side='right') - 1
    idx[idx >= len(ts) - 1] = 0                               # original: no interval found -> 0 ...
    for k in range(1, n_ctrl):                                # ... then forced strictly increasing
        if idx[k] <= idx[k - 1]: idx[k] = idx[k - 1] + 1
    # quirk kept from the original: if rounding leaves cum[-1] <= 1, the last point is placed one
    # oversampled step after the previous one (~1/n_ctrl of the length short of the mouth)
    return over[idx] / (np.array(shape) - 1)




# ---------------------------------------------------------------- fin removal (width envelope)
def _frame(curve):
    tan = np.gradient(curve, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    return tan, np.column_stack([-tan[:, 1], tan[:, 0]])


def body_envelope(mask, curve, win_frac=0.2, k=1.25, pad=1.5):
    '''Half-widths of the body on each side of `curve` (px, (n,2)): the measured extent along each normal,
       median-filtered over win_frac of the body length (fins are shorter than half the window, so they
       are filtered out as outliers), times k plus pad px. Returns (env_left, env_right, a, b) with
       a, b the raw measured extents.'''
    tan, nrm = _frame(curve)
    lo, hi = _edges(mask.astype(float), curve, nrm)
    a, b = -lo, hi
    ok = ~np.isnan(a); idx = np.arange(len(a))
    a, b = np.interp(idx, idx[ok], a[ok]), np.interp(idx, idx[ok], b[ok])   # points just outside the mask
    w = max(3, int(win_frac * len(curve)) | 1)
    return k * median_filter(a, w, mode='nearest') + pad, k * median_filter(b, w, mode='nearest') + pad, a, b


def clip_to_envelope(mask, curve, env_l, env_r, beyond=3.0):
    '''Keep mask pixels whose nearest midline point j has them within [-env_l[j], env_r[j]] sideways
       (and, past the two ends, at most `beyond` px further along); largest connected piece.'''
    tan, nrm = _frame(curve)
    pix = np.argwhere(mask).astype(float)
    j = cKDTree(curve).query(pix)[1]
    d = pix - curve[j]
    lat, along = (d * nrm[j]).sum(1), (d * tan[j]).sum(1)
    keep = (lat >= -env_l[j]) & (lat <= env_r[j])
    keep &= ~((j == 0) & (along < -beyond)) & ~((j == len(curve) - 1) & (along > beyond))
    out = np.zeros_like(mask); out[tuple(pix[keep].astype(int).T)] = True
    return _largest_component(out)


def remove_fins(mask, head, n_pass=2, **env_kw):
    '''Width-envelope fin removal: midline (Isomap, few knots so fins pull it less) -> smooth body width
       envelope -> drop pixels outside it. Repeated n_pass times, each time measuring the envelope on the
       ORIGINAL mask around the midline of the previous cleaned mask. Returns the cleaned mask.'''
    clean = mask
    for _ in range(n_pass):
        curve = fish_midline_isomap(clean, head, n_knots=6) * (np.array(mask.shape) - 1)
        env_l, env_r, _, _ = body_envelope(mask, curve, **env_kw)
        clean = clip_to_envelope(mask, curve, env_l, env_r)
    return clean




# ---------------------------------------------------------------- zig-zag limit (P-spline, order-3 penalty)
def turning_deg(cur):
    '''total absolute turning of a curve in degrees'''
    d = np.diff(cur, axis=0); a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.degrees(np.abs(np.diff(a)).sum())


def smooth_midline(curve, lam=1e-2, order=3, n_knots=40, n_out=None):
    '''Refit a midline (n,2) as a penalised spline (P-spline) of arc length with ends pinned.
       The penalty is on `order`-th differences of the spline coefficients: order 2 penalises bending
       itself (straightens real bends too), order 3 penalises *changes* in bending, so a steady C-bend
       costs little while wiggles cost a lot. lam is relative to the number of points.'''
    n = len(curve) if n_out is None else n_out
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]; u = s / s[-1]
    t = np.r_[[0.0] * 4, np.linspace(0, 1, n_knots + 2)[1:-1], [1.0] * 4]
    B = BSpline.design_matrix(u, t, 3).toarray()
    W = np.ones(len(u)); W[[0, -1]] = 1e4                      # pin the ends
    D = np.diff(np.eye(B.shape[1]), order, axis=0)
    c = np.linalg.solve(B.T @ (W[:, None] * B) + lam * len(u) * D.T @ D, B.T @ (W[:, None] * curve))
    dense = BSpline(t, c, 3)(np.linspace(0, 1, 20 * n))
    sd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    ue = np.linspace(0, sd[-1], n)
    return np.column_stack([np.interp(ue, sd, dense[:, 0]), np.interp(ue, sd, dense[:, 1])])




# ---------------------------------------------------------------- mask-based Denis
DENIS_FRAC = (0.0, 0.548, 0.894, 1.0)        # tail, middle, head, mouth along tail -> mouth (median of tracked keypoints)


def mask_keypoints(mask, head, fractions=DENIS_FRAC, band=1.0):
    '''Ordered points along the body taken from the mask instead of tracking. Every pixel gets its distance
       through the body from the head (shortest path inside the mask); the farthest pixel is the tail
       tip, L its distance. The point at fraction f (0 = tail, 1 = head, as in Denis's order) is the
       centroid of the pixels at distance (1 - f) * L +- band. f = 1 is the head pixel itself; f = 0 the
       centroid of the pixels within `band` of the farthest distance. Returns (len(fractions), 2) px.'''
    G, pix = _pixel_graph(_largest_component(mask))
    start = np.argmin(np.linalg.norm(pix - head, axis=1))
    g = dijkstra(G, indices=start)
    L = g.max()
    out = []
    for f in fractions:
        target = (1 - f) * L
        if f >= 1: out.append(pix[start]); continue
        sel = np.abs(g - min(target, L - band)) <= band
        out.append(pix[sel].mean(0))
    return np.array(out)


def fish_midline_mask_denis(mask, head, n_points=4, n_ctrl=200):
    '''Denis's method with the keypoints taken from the mask: n_points = 4 uses his tail/middle/head/mouth
       positions; any other n_points uses evenly spaced positions. Same interpolation, oversampling and
       arc-length picking as fish_midline_denis. Returns (n_ctrl,2) in [0,1], tail -> head.'''
    fr = DENIS_FRAC if n_points == 4 else np.linspace(0, 1, n_points)
    return fish_midline_denis(mask_keypoints(mask, head, fr), mask.shape, n_ctrl=n_ctrl)


# ---------------------------------------------------------------- mask-only: body tips, head by area
def mask_tips(mask):
    '''The two ends of the body: the two pixels furthest apart *through the body* (shortest paths inside
       the mask, two sweeps). Returns pixels (n,2), distance g of every pixel from tip a, indices a, b.'''
    G, pix = _pixel_graph(_largest_component(mask))
    c = np.argmin(np.linalg.norm(pix - pix.mean(0), axis=1))
    a = int(np.argmax(dijkstra(G, indices=c)))
    g = dijkstra(G, indices=a)
    return pix, g, a, int(np.argmax(g))


def mask_slice_points(pix, g, a, b, n_points=8, band=1.0):
    '''n_points evenly spaced along the body from tip a to tip b: the two tip pixels themselves, and in
       between the centroid of the pixels at that distance (+- band px) from tip a (a slice across the body).'''
    L = g[b]; pts = []
    for f in np.linspace(0, 1, n_points):
        if f == 0: pts.append(pix[a])
        elif f == 1: pts.append(pix[b])
        else: pts.append(pix[np.abs(g - f * L) <= band].mean(0))
    return np.array(pts, float)


def end_areas(mask, curve, frac=1/3):
    '''mask pixels whose nearest midline point lies in the first / last `frac` of the midline's length'''
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]; s /= s[-1]
    u = s[cKDTree(curve).query(np.argwhere(mask))[1]]
    return int((u <= frac).sum()), int((u >= 1 - frac).sum())

def slice_centroids(pix, g):
    '''centroids of 1-px slices of the through-body distance g (one point per px of body length), in order'''
    k = np.floor(g).astype(int); cnt = np.bincount(k); nz = cnt > 0
    return np.column_stack([np.bincount(k, pix[:, 0])[nz], np.bincount(k, pix[:, 1])[nz]]) / cnt[nz][:, None]


def _slice_spline(pix, g, n_ctrl=200, n_knots=40, lam=10):
    '''the slice spline from precomputed mask_tips output, in px, running from tip a to tip b'''
    return smooth_midline(slice_centroids(pix, g), lam=lam, order=3, n_knots=n_knots, n_out=n_ctrl)


def fish_midline_slice_spline(mask, head, n_ctrl=200, n_knots=40, lam=10):
    '''Slice spline: the slice centroids of variant B, fitted by ONE penalised spline (40 knots, penalty lam on
       third differences = changes in bending, ends pinned). Same accuracy as the two-stage
       fish_midline_slices with one smoothing setting (tuned in Midline_fitting_12_32_58.ipynb §1).
       Oriented with `head`; returns (n_ctrl,2) in [0,1], tail -> head, evenly spaced by arc length.
       For many frames use fish_midline_with_head, which returns this midline too without finding the tips twice.'''
    pix, g, a, b = mask_tips(mask)
    return _orient(_slice_spline(pix, g, n_ctrl, n_knots, lam), head) / (np.array(mask.shape) - 1)


def fish_midline_mask_denis_auto(mask, n_points=8, n_ctrl=200, frac=1/3):
    '''Mask-only Denis: slice points from tip to tip (snout and tail tips included), Denis's interpolation,
       and the head = the end whose third of the body has more mask area (the head is wider than the tail).
       Returns (midline (n_ctrl,2) in [0,1] tail -> head, points (n_points,2) px tail -> head,
       info dict: tip_a, tip_b (px), area_a, area_b (px in each end's third), head_is_a).'''
    pix, g, a, b = mask_tips(mask)
    pts = mask_slice_points(pix, g, a, b, n_points)                          # tip a -> tip b
    cur = fish_midline_denis(pts, mask.shape, n_ctrl=n_ctrl)
    area_a, area_b = end_areas(mask, cur * (np.array(mask.shape) - 1), frac)
    info = dict(tip_a=pix[a].astype(float), tip_b=pix[b].astype(float), area_a=area_a, area_b=area_b, head_is_a=area_a > area_b)
    if info['head_is_a']: cur, pts = cur[::-1], pts[::-1]
    return cur, pts, info


# ---------------------------------------------------------------- head / tail: area rule, contact, continuity
def fish_midline_with_head(mask, frac=1/3, n_ctrl=200, n_knots=40, lam=10):
    '''The slice-spline midline of a fish mask, oriented by the area rule, together with the two body tips, all
       from one mask_tips call. Area rule: the head end of the body is wider than the tail, so the head is the tip
       whose first `frac` of the body (by distance through the body from that tip) holds more mask pixels.
       Returns dict: tip_a, tip_b (px), area_a, area_b (pixels in each end's `frac`), head_is_a, head, tail (px),
       midline: the slice spline (n_ctrl,2) in [0,1], oriented tail -> head by the area rule (the same curve as
       fish_midline_slice_spline(mask, head, n_ctrl, ...)). If continuity later picks the other end as the head,
       use oriented_midline(body, head_is_a): the curve is the same, only reversed.'''
    pix, g, a, b = mask_tips(mask)
    L = g[b]
    area_a, area_b = int((g <= frac * L).sum()), int((g >= (1 - frac) * L).sum())
    ta, tb = pix[a].astype(float), pix[b].astype(float)
    head_is_a = area_a > area_b
    cur = _slice_spline(pix, g, n_ctrl, n_knots, lam)                          # tip a -> tip b
    if head_is_a: cur = cur[::-1]
    return dict(tip_a=ta, tip_b=tb, area_a=area_a, area_b=area_b, head_is_a=head_is_a,
                head=ta if head_is_a else tb, tail=tb if head_is_a else ta, midline=cur / (np.array(mask.shape) - 1))


def oriented_midline(body, head_is_a):
    '''The midline of a fish_midline_with_head result, tail -> head for the given head side (e.g. from heads_with_continuity)'''
    return body['midline'] if head_is_a == body['head_is_a'] else body['midline'][::-1]


def contact_frames(get_masks, frames, gap=2):
    '''Frames in which each fish's mask touches another fish's mask.
       get_masks(frame) -> list of (P,P) bool masks, one per fish (None or empty where a fish has no mask).
       Two fish are in contact when their masks come within `gap` px of each other (8-neighbour steps): one mask,
       grown by `gap` px, overlaps the other. Only pairs whose bounding boxes are within `gap` px are tested.
       Returns (contact: (len(frames), n_fish) bool, {fish index: array of the frames in contact}).'''
    from scipy.ndimage import binary_dilation
    frames = list(frames); contact = None
    for t, f in enumerate(frames):
        masks = get_masks(f)
        if contact is None: contact = np.zeros((len(frames), len(masks)), bool)
        boxes = []
        for m in masks:
            if m is None or not m.any(): boxes.append(None); continue
            r, c = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
            boxes.append((r[0], r[-1], c[0], c[-1]))
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                bi, bj = boxes[i], boxes[j]
                if bi is None or bj is None: continue
                if bi[0] > bj[1] + gap or bj[0] > bi[1] + gap or bi[2] > bj[3] + gap or bj[2] > bi[3] + gap: continue
                r0, r1 = max(min(bi[0], bj[0]) - gap, 0), max(bi[1], bj[1]) + gap + 1      # window holding both
                c0, c1 = max(min(bi[2], bj[2]) - gap, 0), max(bi[3], bj[3]) + gap + 1
                grown = binary_dilation(masks[i][r0:r1, c0:c1], structure=np.ones((3, 3)), iterations=gap)
                if (grown & masks[j][r0:r1, c0:c1]).any(): contact[t, i] = contact[t, j] = True
    return contact, {i: np.asarray(frames)[contact[:, i]] for i in range(contact.shape[1])}


def heads_with_continuity(tips, contact):
    '''Head side of one fish over consecutive frames: the area rule outside contact, continuity during contact.
       tips: list of fish_midline_with_head(...) dicts, one per consecutive frame (None where the fish has no usable mask);
       contact: bool array of the same length (e.g. a column of contact_frames).
       A contact episode is a run of consecutive contact frames. The frames just before and after it are
       contact-free, so the area rule holds there (anchors). From each anchor the head is chained frame by frame:
       in each frame, the body tip nearest the previous frame's head (a head can't jump to the other end of the
       body in one frame). Both chains agree -> that tip; they disagree -> the chain from the anchor nearer in time;
       only one anchor -> that chain; no anchor -> the area rule.
       Returns (head_is_a: list of bool or None, rule: list of 'area' / 'continuity' / 'area (no anchor)' / None);
       the head of frame t is tips[t]['tip_a'] if head_is_a[t] else tips[t]['tip_b'].'''
    n = len(tips); contact = np.asarray(contact, bool)
    head_is_a = [None if t is None else bool(t['head_is_a']) for t in tips]
    rule = [None if t is None else 'area' for t in tips]
    d = np.diff(np.r_[0, contact.astype(int), 0])
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1):           # contact episodes s..e

        def anchor_head(k):
            return None if not (0 <= k < n) or tips[k] is None else tips[k]['head']

        def chain(order, head):
            out = {}
            for k in order:
                if tips[k] is None or head is None: out[k] = None; continue
                ta, tb = tips[k]['tip_a'], tips[k]['tip_b']
                out[k] = bool(np.linalg.norm(ta - head) <= np.linalg.norm(tb - head))
                head = ta if out[k] else tb
            return out
        fwd = chain(range(s, e + 1), anchor_head(s - 1))
        bwd = chain(range(e, s - 1, -1), anchor_head(e + 1))
        for k in range(s, e + 1):
            if tips[k] is None: continue
            F, B = fwd[k], bwd[k]
            if F is not None and B is not None: head_is_a[k] = F if (F == B or k - (s - 1) <= (e + 1) - k) else B
            elif F is not None or B is not None: head_is_a[k] = F if F is not None else B
            else: rule[k] = 'area (no anchor)'; continue
            rule[k] = 'continuity'
    return head_is_a, rule



# old way of doing spline (bad, since PCA gives line axis)
# def fish_midline_from_mask(mask, n_ctrl=200, smooth=None):
#     """(P,P) bool -> smooth curved midline (n_ctrl,2) in [0,1], oriented tail->head."""
#     xs, ys = np.nonzero(mask)
#     pts = np.column_stack([xs, ys]).astype(float); c = pts.mean(0)
#     v = np.linalg.svd(pts - c, full_matrices=False)[2][0]
#     perp = np.array([-v[1], v[0]])
#     t = (pts - c) @ v
#     w = (pts - c) @ perp

#     # average perpendicular offset per unique t (spline needs strictly increasing x)
#     order = np.argsort(t); t_s, w_s = t[order], w[order]
#     tu, inv, cnt = np.unique(np.round(t_s, 6), return_inverse=True, return_counts=True)
#     wu = np.bincount(inv, weights=w_s) / cnt
#     L = t_s.max() - t_s.min()
#     if smooth is None:
#         smooth = len(tu) * (0.05 * L) ** 2             # gentle default
#     spl = UnivariateSpline(tu, wu, k=3, s=smooth, w=np.sqrt(cnt))
#     ts = np.linspace(t_s.min(), t_s.max(), n_ctrl)
#     curve = c + np.outer(ts, v) + np.outer(spl(ts), perp)

#     # orient tail->head: the head end is bulkier (more pixels) -> that half goes last
#     t_mid = 0.5 * (t.min() + t.max())
#     if np.count_nonzero(t < t_mid) > np.count_nonzero(t >= t_mid):
#         curve = curve[::-1]

#     nx, ny = mask.shape
#     return curve / np.array([nx - 1, ny - 1])
