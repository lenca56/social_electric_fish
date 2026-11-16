"""Motion of a fish around an EOD frame from its ZEUS-mask midline: velocity, head / body / tail angular speed.
Used by EOD_motion_plots_my_pipeline.ipynb and EOD_examples_fast_motion.ipynb."""
import numpy as np
from io_utils import unpack_crop
from midline_methods import fish_midline_with_head

PX_TO_CM = 60 / 912
N_POINTS = 101
MEASURES = ('velocity_cm_s', 'head_angular_deg_s', 'body_angular_deg_s', 'tail_angular_deg_s')


def resample(curve, n=N_POINTS):
    '''n points evenly spaced in arc length along a polyline.'''
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, curve[:, k]) for k in range(2)])


def midline_cm(frame, label, min_pixels=50):
    '''Mask midline (N_POINTS, 2) of ZEUS label in a tracked frame, (x, y) cm, tail -> head; None without a single mask.'''
    hits = [(p, b) for l, p, b in zip(frame['labels'], frame['masks'], frame['boxes']) if l == label]
    if len(hits) != 1: return None
    (packed, (y0, y1, x0, x1)), = hits
    crop = unpack_crop(packed, (y0, y1, x0, x1))
    if crop.sum() < min_pixels: return None
    m = fish_midline_with_head(crop, n_ctrl=N_POINTS)['midline'] * (np.array(crop.shape) - 1) + (y0, x0)
    return resample(m[:, ::-1]) * PX_TO_CM


def direction_deg(a, b):
    '''Direction (°) of the vector a -> b.'''
    d = b - a
    return np.degrees(np.arctan2(d[1], d[0]))


def turn(d0, d1):
    '''Absolute change between two directions (°), wrapped to [0, 180].'''
    return abs((d1 - d0 + 180) % 360 - 180)


def motion(frames, k, label, fps):
    '''(velocity cm/s, head, body and tail angular speed °/s) of one fish around frame k, from its mask midlines in
       frames k - 1 and k + 1 (NaN if either has none). If the head rule picked opposite ends in the two frames, the
       k + 1 midline is reversed first.
       velocity: displacement of the midline centroid; head: direction of the head third; body: direction of the
       whole-body axis (tail tip -> head tip); tail: angle of the tail third relative to the head third, so a rotation
       of the whole fish does not count.'''
    m0, m1 = midline_cm(frames[k - 1], label), midline_cm(frames[k + 1], label)
    if m0 is None or m1 is None: return (np.nan,) * 4
    n = len(m0)
    if np.dot(m0[-1] - m0[n // 2], m1[-1] - m1[n // 2]) < 0: m1 = m1[::-1]      # same head end in both frames
    dt = 2 / fps
    tail_rel = lambda m: direction_deg(m[n // 3], m[0]) - direction_deg(m[-1], m[2 * n // 3])
    return (np.linalg.norm(m1.mean(0) - m0.mean(0)) / dt,
            turn(direction_deg(m0[2 * n // 3], m0[-1]), direction_deg(m1[2 * n // 3], m1[-1])) / dt,
            turn(direction_deg(m0[0], m0[-1]), direction_deg(m1[0], m1[-1])) / dt,
            turn(tail_rel(m0), tail_rel(m1)) / dt)
