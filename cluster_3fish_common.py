"""Shared settings and paths of the 3-fish mask-stage jobs (one place, so the three scripts always agree):

  cluster_3fish_stage_midlines.py   1. mask midlines of every fish, per recording x mask stage
  cluster_3fish_stage_predict.py    2. raw model predictions per recording x mask stage x model setting (mine at several interior
                                       conductivities, Denis' analytic pipeline)
  assign_3fish_stage_rules.py       3. assignment rules (shape, cosine, voltage with fitted factors) and accuracy, all combinations

DATA (relative to the repository root)
  masks      data/3_fish_silent/masks/video<rec>/ : one efish4-track run, read per stage with io_utils.load_stage_masks; the 10
             stages io_utils.STAGES = segment, identity, contact, body_masks, propagate, keypoints, extent, recover, ownership,
             disturbance (letters A, B, C are the same fish at every stage; checked on 14_12_25: IoU 0.92-1.0 with the latest run)
  EOD table  data/3_fish_silent/Data_3fish_<rec>.csv (added 2026-11-19 for 8 recordings; the one of 14_12_25 has the same EODs,
             frames, times and traces as its older _synch_fix_newest table). Only these 8 recordings are in VIDEOS.
OUTPUT       results/eod_assignment_3_fish_silent/mask_stages/{midlines, predictions, assignment}/
"""
import os
import numpy as np
import pandas as pd

LETTERS = 'ABC'
P = 912                                       # video frame size, px
PX_PER_CM_VIDEO = P / 60                      # video calibration (whole frame = 60 cm), for distances between fish
N_MIDLINE = 101                               # midline points (tail -> head) = point currents of both models

# the 8 recordings with an EOD table (and efish4-track mask stages), fixed so a task index always means the same recording;
# not included (no EOD table): 2024-10-14T15_04_15, 2024-10-14T15_24_15, 2024-10-14T16_27_02 (the last has no mask stages either)
VIDEOS = ['2024-10-14T15_47_02', '2024-10-14T16_07_02', '2024-10-14T16_47_02', '2024-10-15T14_12_25', '2024-10-15T14_30_53',
          '2024-10-15T14_50_54', '2024-10-15T15_10_54', '2024-10-15T15_30_54']
# the emitting fish (Kaden's letter) of each silent control, the ground truth for the accuracy. A recording not listed gets the
# fish that the shape rule picks most often over all its stages and settings (assign_3fish_stage_rules.py writes emitter.csv with
# that share, to check; put the letter here once confirmed)
EMITTER = {'2024-10-15T14_12_25': 'C'}       # Kaden's C = ZEUS fish 0, the emitter of this control

# geometry
TANK_OUTLINE = 'data/electrodes/tank_outline_updated_3-fish.csv'          # wall polygon, video px (x = column, y = row)
ELECTRODES = 'data/electrodes/electrode_locations_updated_3-fish.csv'     # my model: electrodes just inside the walls
DENIS_ELECTRODES = 'data/electrodes/electrodes_signed_px_coord_3-fish.csv'  # Denis' pipeline: his original electrode layout

# data and output
MASK_RUN = 'data/3_fish_silent/masks/video{video}'
EOD_TABLE = 'data/3_fish_silent/Data_3fish_{video}.csv'
OUT = 'results/eod_assignment_3_fish_silent/mask_stages'
MIDLINES = OUT + '/midlines/{video}/{stage}.npz'
PREDICTIONS = OUT + '/predictions/{setting}/{video}/{stage}.npz'
ASSIGNMENT = OUT + '/assignment'


def stage_list():
    '''the mask stages to run, from the environment variable MASK_STAGES: 'all' (default: the 10 efish4-track stages), 'final'
       (the run's final masks, outputs/video<rec>.pkl), or a comma list of stage names (e.g. segment,extent,final)'''
    from io_utils import STAGES
    want = os.environ.get('MASK_STAGES', 'all')
    out = list(STAGES) if want == 'all' else [s.strip() for s in want.split(',')]
    bad = [s for s in out if s not in STAGES and s != 'final']
    assert not bad, f'MASK_STAGES: unknown stage(s) {bad}; choose from {list(STAGES)} or final'
    return out


def stage_label(stage):
    '''file name of a stage: 01_segment .. 10_disturbance (pipeline order), or final'''
    from io_utils import STAGES
    return 'final' if stage == 'final' else f'{STAGES.index(stage) + 1:02d}_{stage}'


def setting_tag(model, sigma=None):
    '''folder name of one model setting: mine_int<sigma> or denis'''
    return f'mine_int{sigma:g}' if model == 'mine' else 'denis'


def eod_frames(video):
    '''(n_eods,) video frame of every EOD row of the table, -1 for an EOD before the video starts (Time < 0: the table puts it on
       frame 1, but there is no frame for it, so it gets no midline and no prediction); None without a table.'''
    path = EOD_TABLE.format(video=video)
    if not os.path.exists(path): return None
    d = pd.read_csv(path, usecols=['Frame', 'Time'])
    return np.where(d.Time.to_numpy() >= 0, d.Frame.to_numpy(), -1)


def poles(path):
    '''(n_pairs, 2) + and (n_pairs, 2) - electrode positions (x, y), video px, pairs in ascending order.'''
    EL = pd.read_csv(path).sort_values('pair')
    plus = np.array([(r.x1, r.y1) if str(r.pol1).strip() == '+' else (r.x2, r.y2) for _, r in EL.iterrows()], float)
    minus = np.array([(r.x2, r.y2) if str(r.pol1).strip() == '+' else (r.x1, r.y1) for _, r in EL.iterrows()], float)
    return plus, minus


def nearer_pole(mids, plus, minus):
    '''mids (n, n_fish, N_MIDLINE, 2) (row, col) / (P - 1) -> (n, n_fish, n_pairs): 0 where pair p's + electrode is nearer to the
       fish's midline, 1 where its - electrode is (0 for a fish without a midline).'''
    out = np.zeros(mids.shape[:2] + (len(plus),), int)
    for s in range(0, len(mids), 1000):
        xy = mids[s:s + 1000, :, :, ::-1] * (P - 1)
        dp = np.linalg.norm(xy[:, :, :, None] - plus[None, None, None], axis=-1).min(2)
        dm = np.linalg.norm(xy[:, :, :, None] - minus[None, None, None], axis=-1).min(2)
        out[s:s + 1000] = np.where(np.isfinite(dp) & np.isfinite(dm), dm < dp, 0)
    return out


def save_npz(path, **arrays):
    '''np.savez_compressed to path, written to .tmp and renamed when complete'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'wb') as fh: np.savez_compressed(fh, **arrays)
    os.replace(path + '.tmp', path)
