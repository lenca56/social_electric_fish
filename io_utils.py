"""Reading the project's data files: one reader per file type, so every script loads data the same way.

  load_masks(path, tracking='kaden')           fish masks of a tracked video (Kaden's efish4-track or Shiv's ZEUS), per frame
  load_electrodes(path, P)                     electrode pairs, + and - pole in tank coordinates
  load_tank_outline(path)                      tank outline vertices, video px
  load_eods(path)                              EOD table: per-EOD info and the recorded electrode-pair traces
  load_assignment(path)                        my model's assignment (eod_assignment_shape.csv)
  load_denis_assignment(summary, assigned)     Denis' pipeline's assignment, in the same form
"""
import pickle, re
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

TRACKINGS = ('kaden', 'shiv')


# ---------------------------------------------------------------- mask encodings
def rle_counts(s):
    '''COCO compressed RLE string -> run lengths (port of pycocotools rleFrString)'''
    if isinstance(s, bytes): s = s.decode('ascii')         # the efish4-track .pkl stores counts as bytes
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << 5 * k
            more = c & 0x20
            p += 1; k += 1
            if not more and (c & 0x10): x |= -1 << 5 * k
        if len(cnts) > 2: x += cnts[-2]
        cnts.append(x)
    return cnts


def rle_decode(rle):
    '''{'size': [h, w], 'counts': str} -> (h, w) bool mask, indexed [row = y, col = x]'''
    h, w = rle['size']
    cnts = np.asarray(rle_counts(rle['counts']), np.int64)
    vals = np.zeros(len(cnts), bool); vals[1::2] = True               # runs alternate 0, 1, 0, ... starting with 0
    flat = np.repeat(vals, cnts)
    return np.pad(flat, (0, h * w - len(flat))).reshape((h, w), order='F')   # COCO is column-major


def rle_crop(rle):
    '''COCO RLE mask -> (box (y0, y1, x0, x1), bool crop), the mask without the empty frame around it; (None, None) if empty'''
    h = rle['size'][0]
    cnts = np.asarray(rle_counts(rle['counts']), np.int64)
    starts = np.cumsum(cnts) - cnts
    on_s, on_l = starts[1::2], cnts[1::2]                              # runs alternate 0s, 1s
    keep = on_l > 0; on_s, on_l = on_s[keep], on_l[keep]
    if on_l.size == 0: return None, None
    idx = np.repeat(on_s - (np.cumsum(on_l) - on_l), on_l) + np.arange(on_l.sum())
    xs, ys = np.divmod(idx, h)                                         # column-major order
    y0, x0 = ys.min(), xs.min()
    box = (int(y0), int(ys.max()) + 1, int(x0), int(xs.max()) + 1)
    crop = np.zeros((box[1] - y0, box[3] - x0), bool); crop[ys - y0, xs - x0] = True
    return box, crop


def unpack_crop(packed, box):
    '''ZEUS bit-packed crop -> (y1 - y0, x1 - x0) bool mask of box (y0, y1, x0, x1)'''
    y0, y1, x0, x1 = box
    return np.unpackbits(packed)[:(y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0).astype(bool)


# ---------------------------------------------------------------- fish masks of a tracked video
class Masks:
    '''Fish masks of one tracked video, decoded one frame at a time.
         masks[k]         (n_fish, H, W) bool of video frame k; all False where a fish has no mask or k is outside the video
         masks.crops(k)   {f: (box (y0, y1, x0, x1), crop)} of the fish with a mask in frame k (fast: no full frames)
       Row f is the tracker's fish fish_order[f]; names[f] is its letter (Kaden) or ZEUS label (Shiv).
       Tracker identities: kaden  letter index (0 = A, 1 = B, ...)
                           shiv   ZEUS label (1 largest, 2 second, ..., 0 smallest); remnant labels ("not a fish") are dropped'''

    def __init__(self, path, tracking='kaden', fish_order=None, kind='visible'):
        if tracking not in TRACKINGS: raise ValueError(f'tracking must be one of {TRACKINGS}, not {tracking!r}')
        with open(path, 'rb') as f: d = pickle.load(f)
        if tracking == 'kaden':                            # mask_<kind>[k][i]: COCO RLE of letter i, or None
            self._frames, self.shape, self.n_frames = d[f'mask_{kind}'], tuple(d['image_size']), d['n_frames']
            ids, name = range(len(d['letters'])), lambda i: d['letters'][i]
        else:                                              # frames[k]: labels, boxes (y0, y1, x0, x1), bit-packed crops
            self._frames, self.shape, self.n_frames = d['frames'], tuple(d['shape']), len(d['frames'])
            ids, name = sorted(l for l, s in d['labels_key'].items() if 'not a fish' not in s), lambda l: l
        self.fish_order = tuple(ids) if fish_order is None else tuple(fish_order)
        if not set(self.fish_order) <= set(ids): raise ValueError(f'fish_order {self.fish_order}: {tracking} fish are {tuple(ids)}')
        self.tracking, self.path, self.fps = tracking, path, d['fps']
        self.n_fish, self.names = len(self.fish_order), tuple(name(i) for i in self.fish_order)
        self._row = {fid: f for f, fid in enumerate(self.fish_order)}

    def __len__(self):
        return self.n_frames

    def _items(self, k):
        '''(row, encoded mask) of the wanted fish in frame k'''
        k = int(k)
        if self.tracking == 'kaden':
            if not 0 <= k < self.n_frames: return []
            return [(self._row[i], rle) for i, rle in enumerate(self._frames[k]) if rle is not None and i in self._row]
        fr = self._frames.get(k, {'labels': ()})
        return [(self._row[l], (p, b)) for l, p, b in zip(fr['labels'], fr.get('masks', ()), fr.get('boxes', ())) if l in self._row]

    def crops(self, k):
        out = {}
        for f, enc in self._items(k):
            box, crop = rle_crop(enc) if self.tracking == 'kaden' else (tuple(enc[1]), unpack_crop(*enc))
            if box is None: continue
            if f in out:                                   # a second mask of the same fish: merge the two
                (a, ca), b = out[f], box
                u = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                c = np.zeros((u[1] - u[0], u[3] - u[2]), bool)
                for (y0, y1, x0, x1), m in ((a, ca), (b, crop)): c[y0 - u[0]:y1 - u[0], x0 - u[2]:x1 - u[2]] |= m
                box, crop = u, c
            out[f] = (box, crop)
        return out

    def __getitem__(self, k):
        m = np.zeros((self.n_fish,) + self.shape, bool)
        for f, enc in self._items(k):
            if self.tracking == 'kaden': m[f] = rle_decode(enc); continue
            y0, y1, x0, x1 = enc[1]; m[f, y0:y1, x0:x1] |= unpack_crop(*enc)
        return m


def load_masks(path, tracking='kaden', fish_order=None, kind='visible'):
    '''Fish masks of a tracked video (see Masks): masks = load_masks(path); masks[k] -> (n_fish, H, W) bool.
       tracking   'kaden' (efish4-track .pkl, default) or 'shiv' (ZEUS *tracking_final.pkl)
       fish_order tracker identities in the order wanted (e.g. (1, 0) puts B first); default: all fish, tracker order
       kind       Kaden's 'visible' (default) or 'amodal' (full-body) masks; ZEUS has visible masks only'''
    return Masks(path, tracking, fish_order, kind)


# ---------------------------------------------------------------- electrodes and tank
def load_electrodes(path, P):
    '''Electrode file (columns pair, x1, y1, pol1, x2, y2, pol2; video px, x = column, y = row) ->
       [(pair id, + pole, - pole)], each pole (row, col) / (P - 1) in [0, 1] tank coordinates (P = video frame size, px).'''
    to_tank = lambda x, y: np.array([y / (P - 1), x / (P - 1)])
    pairs = []
    for _, r in pd.read_csv(path).iterrows():
        a, b = to_tank(r['x1'], r['y1']), to_tank(r['x2'], r['y2'])
        pairs.append((int(r['pair']), a, b) if str(r['pol1']).strip() == '+' else (int(r['pair']), b, a))
    return pairs


def load_tank_outline(path):
    '''Tank outline file (columns x, y; tank_outline_electrodes.py) -> (n_vertices, 2) vertices (x, y) in video px'''
    return pd.read_csv(path)[['x', 'y']].to_numpy()


# ---------------------------------------------------------------- EOD tables
_TRACE = re.compile(r'electrode_(\d+)_t(?:ime)?_([mp]?)(\d+)$')   # sample i (electrode_3_time_7) or m12 ... 0 ... p12


def load_eods(path, columns=None, pairs=None, traces=True):
    '''EOD table (one row per EOD: Frame, Time, keypoints, ..., electrode_<pair>_time_<i> samples) -> (info, V, pair_ids)
         info      DataFrame of the other columns (all, or the list `columns`), index = EOD row
         V         (n_eods, n_pairs, n_samples) recorded pair voltages as stored (not demeaned), samples in time order;
                   None with traces=False (reads only `info`, much faster)
         pair_ids  the pairs along V's axis 1: `pairs`, or every pair in the file, ascending'''
    head = pd.read_csv(path, nrows=0).columns
    samples = {}
    for c in head:
        m = _TRACE.match(c)
        if m: samples.setdefault(int(m[1]), []).append(((-1 if m[2] == 'm' else 1) * int(m[3]), c))
    pair_ids = sorted(samples) if pairs is None else list(pairs)
    cols = [c for p in pair_ids for _, c in sorted(samples[p])] if traces else []
    info_cols = [c for c in head if not _TRACE.match(c)] if columns is None else list(columns)
    d = pd.read_csv(path, usecols=info_cols + cols)
    V = d[cols].to_numpy(float).reshape(len(d), len(pair_ids), -1) if traces else None
    return d[info_cols], V, pair_ids


# ---------------------------------------------------------------- assignment results
@dataclass
class Assignment:
    '''One pipeline's assignment of the EODs of one table; row i is EOD row i of the table.
         pick       (n_eods,) fish index picked, -1 = no pick
         recorded   (n_eods, n_pairs) the recorded value the pipeline compares (mine: pair voltages at the peak sample;
                    Denis: signed peak-to-peak), NaN where the pipeline has none
         predicted  (n_eods, n_fish, n_pairs) the model's prediction per candidate fish (mine: unit-current pair voltages,
                    before MODEL_SIGN; Denis: predicted peak-to-peak, his units), NaN where a fish has no model
         score      (n_eods, n_fish) the pipeline's own rule (mine: L2 of max-normalised patterns, smallest wins;
                    Denis: cosine similarity, largest wins)
         table      the file's per-EOD columns (eod_row, Frame, ...; mine also pick_cosine and cos<f>)
         pair_ids   pairs along the last axis;  fish  name of each fish index (mine: ZEUS number; Denis: his letter)'''
    pick: np.ndarray
    recorded: np.ndarray
    predicted: np.ndarray
    score: np.ndarray
    table: pd.DataFrame
    pair_ids: list
    fish: list = field(default_factory=list)


def load_assignment(path):
    '''My model's per-EOD result table (<run>/<tag>/eod_assignment_shape.csv, eod_assignment_control_groups.py) as an Assignment.
       Columns: eod_row, Frame, r_raw_p<q>, r_norm, per fish f u<f>_raw_p<q>, u<f>_norm, d<f> [, cos<f>], pick [, pick_cosine].'''
    d = pd.read_csv(path)
    pair_ids = [int(c[7:]) for c in d.columns if c.startswith('r_raw_p')]
    fish = sorted({int(m[1]) for c in d.columns for m in [re.match(r'u(\d+)_raw_p', c)] if m})
    pred = np.stack([d[[f'u{f}_raw_p{q}' for q in pair_ids]].to_numpy(float) for f in fish], axis=1)
    keep = [c for c in d.columns if not re.match(r'(r|u\d+)_raw_p\d+$', c)]
    return Assignment(pick=d.pick.to_numpy(int), recorded=d[[f'r_raw_p{q}' for q in pair_ids]].to_numpy(float), predicted=pred,
                      score=d[[f'd{f}' for f in fish]].to_numpy(float), table=d[keep], pair_ids=pair_ids, fish=fish)


def load_denis_assignment(summary, assigned, fish=None):
    '''Denis' pipeline (EODAssignment, run by the social_electric_fish_runners) as an Assignment on every EOD row.
         summary   denis_mask_midline_all_eods*.csv: one row per EOD of the table (eod_row, Frame, Time, pick = letter or empty)
         assigned  his Assigned_*.pkl: one row per EOD he assigned (those with a pick), in table order, with recorded_PTP,
                   predicted_PTP_<letter>, similarity_<letter>
         fish      letters in the order wanted, e.g. 'CAB' or {'A': 1, 'B': 2, 'C': 0} (letter -> fish index); default A, B, ...'''
    s = pd.read_csv(summary); dk = pd.read_pickle(assigned); has = s.pick.notna().to_numpy()
    assert has.sum() == len(dk) and (dk.Frame.to_numpy() == s.Frame[has].to_numpy()).all(), f'{assigned} does not match {summary}'
    letters = sorted(c.split('_')[-1] for c in dk.columns if c.startswith('predicted_PTP_'))
    if fish is None: fish = letters
    if isinstance(fish, dict): fish = sorted(fish, key=fish.get)
    fish = list(fish); n, nf = len(s), len(fish)
    rec = np.stack(dk.recorded_PTP.to_numpy()); R = np.full((n, rec.shape[1]), np.nan); R[has] = rec
    U, S = np.full((n, nf, rec.shape[1]), np.nan), np.full((n, nf), np.nan)
    for f, X in enumerate(fish):
        U[has, f] = np.stack(dk[f'predicted_PTP_{X}'].to_numpy()); S[has, f] = dk[f'similarity_{X}'].to_numpy()
    pick = s.pick.map({X: f for f, X in enumerate(fish)}).fillna(-1).to_numpy(int)
    return Assignment(pick=pick, recorded=R, predicted=U, score=S, table=s, pair_ids=list(range(1, rec.shape[1] + 1)), fish=fish)
