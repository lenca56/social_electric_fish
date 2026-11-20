"""Step 3 of 3 (see cluster_kaden_3fish_common.py): assignment rules on the raw predictions of step 2, for every recording x mask stage x
model setting found, with the accuracy against the known emitter. Runs in a few minutes on a laptop (no field solves).

RULES (recorded r: 10 pairs; model p_f = model_sign * model_raw of fish f; each EOD goes to the best fish)
  shape_l2        smallest || r / max|r| - p_f / max|p_f| ||  (shape only, each pattern scaled to max |v| = 1)
  cosine          largest cos(r, p_f)  (shape only; L2 between patterns scaled to unit norm ranks exactly like this, 2 - 2 cos)
  voltage_l2      smallest || r - a_f e_k p_f ||  in volts, with fitted factors (below)
  (A voltage cosine rule, cos(r, a_f e_k p_f), is not used: the fish factor cancels in a cosine, so it is only cosine with the
  electrode gains reweighting the pairs, without the amplitude that makes voltage L2 different.)
  (Denis' own pick, his similarity, is the cosine rule on his predictions: identical on all 226,347 picked EODs of the final masks;
  not a rule of its own here. His similarity is still saved in the prediction files.)

FACTORS (per recording x stage x setting, without the ground truth)
  fish factor a_f (FISH_FACTORS = 'length_law', default): the 4-fish linear law of fish factor vs body length (LENGTH_LAW, from
  results/eod_assign_4fish_kaden/clean_emitter_set/gain_fish_x_signed_electrode/fish_factor_vs_length.png; my model's law for every
  interior conductivity, Denis' law for Denis) at the fish's length, read as a_mine / a_denis from FISH_LENGTHS (fish_lengths_3fish.py:
  two sets of fish, one length per fish of a set: the median over the set's recordings of fish_lengths.py, the median midline length of
  the straightest clean frames at 15.2 px/cm; set 1 = Oct 14 and 14_12_25: A 16.91, B 14.39, C 11.99 cm; set 2 = 14_30_53 to 15_30_54:
  A 20.0, B 16.82, C 14.14 cm). Only the ratios between the fish matter: the
  electrode factors take up the overall scale of each model setting. FISH_FACTORS = 'fit': median ratio |r_p| / |p_f,p| of each
  fish's clean EODs on their N_TOP_PAIRS pairs instead (fewer than MIN_RATIOS: the median of all ratios).
  clean EODs: every fish has a prediction; the shape_l2 fish f is at least ISOLATION_CM from both other fish (midline to midline);
  its N_TOP_PAIRS loudest recorded pairs are its N_TOP_PAIRS largest predicted pairs, with the model's sign; the other fish predict
  at most DOMINANCE x its |p| there; the N_CLEAN best by shape_l2 in all (not per fish: in a silent recording they are the emitter's).
  electrode factors are fitted leave-one-recording-out (GAIN_FIT = 'loro', default): the least-squares sums below pooled over the clean
  EODs of the other 7 recordings with the same model setting and mask stage, applied to the held-out one ('per_recording': its own).
  signed electrode factor e_k (k = the pole of pair p nearer to fish f): plain least squares in volts over all 10 pairs of the clean
  EODs, e_k = sum r_p m_p / sum m_p^2 with m_p = a_f p_f,p (loud EODs carry the most weight; no per-EOD normalisation); an electrode
  with fewer than MIN_RATIOS_ELECTRODE terms gets the median of the fitted e_k.

ACCURACY: % of EODs given to the emitter. The emitter is cluster_kaden_3fish_common.EMITTER[recording] where given, else inferred: the
fish the shape_l2 rule picks most often on the 'predicted' EODs, pooled over every stage and setting of the recording (in a silent
control nearly every EOD is the emitter's); emitter.csv lists the emitter, its source and that share. Three EOD sets:
  Every set leaves out the double-detected EODs (DOUBLES, eod_double_detections_3fish.py: the extra member of two EODs < 7 ms apart),
  which are also kept out of the factor fit.
  all        every EOD with a video frame (no pick counts as wrong)
  predicted  EODs where every fish has a prediction (all midlines at that stage)
  good       EODs on frames whose tracking is good: every fish passes the 4-fish mask QC on the final masks (GOOD_FRAMES,
             kaden_3fish_mask_qc.py: has a mask; area within 20% and principal-axis length 0.8-1.2 x its median; >= 99% in one piece;
             centroid step <= 15 px and area change <= 15% to the previous and next frame; not within 5 px of another fish); the
             same EODs at every stage

OUTPUT  results/eod_assignment_3_fish_silent/mask_stages/assignment/
  accuracy.csv                        setting, video, stage, rule, subset, n, n_picked, emitter, accuracy (%), shares A/B/C (%)
  emitter.csv                         video, emitter, source (given / inferred), share of shape_l2 picks (%)
  factors.csv                         setting, video, stage, factor (fish A.., electrode 1+..), value, n_ratios (fish: clean EODs;
                                      electrode: least-squares terms)
  picks/<setting>/<rec>/<stage>.csv   per EOD: eod_row, Frame, pick and score per rule
  accuracy_vs_stage_<rec>.png         accuracy per rule (columns) and EOD set (rows), one line per setting
  accuracy_summary_final.csv / .png   final masks: accuracy pooled over the recordings per setting x rule, all vs good EODs
                                      (_bars.png: bars of the median over recordings, whiskers 25th-75th percentile)
  accuracy_numerical_sigma_rule_final.png      my numerical model only: bars per rule x interior conductivity
  accuracy_best_numerical_vs_analytic_final.png  the best numerical conductivity x rule vs the analytic model's rules
  accuracy_analytic_rules_final.png            the analytic model alone, per rule
  accuracy_mine_int4_vs_analytic_final.png     numerical (interior 4 x water) vs analytic, per rule
                                      (and _per_video.csv)

    python kaden_3fish_assign_rules.py [--workers 4]
"""
import os, sys, glob, json, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_kaden_3fish_common as C

# ================================================================================================================ hyperparameters
RULES = ['shape_l2', 'cosine', 'voltage_l2']
ISOLATION_CM = 10.0           # clean EODs: the emitter at least this far from both other fish
N_TOP_PAIRS = 2               # clean EODs: the loudest recorded pairs = the emitter's largest predicted pairs; ratios on these
DOMINANCE = 0.5               # clean EODs: other fish predict at most this x the emitter's |p| on those pairs
N_CLEAN = 1000                # clean EODs in all (best shape_l2)
FISH_FACTORS = 'length_law'   # 'length_law': a_f from the 4-fish law at the fish's length; 'fit': median ratio of the fish's clean EODs
LENGTH_LAW = {'mine': (0.010661805, 0.313860493),     # a = slope x L (cm) + intercept, 4-fish Kaden fish x signed-electrode fit
              'denis': (0.000204483, 0.002989576)}    # (fish_factors_and_length_fits.csv: 'slope s (per cm)', 'intercept c')
FISH_LENGTHS = 'results/eod_assignment_3_fish_silent/fish_lengths/fish_lengths_3fish.csv'   # video, fish, L_cm, a_mine, a_denis (fish_lengths_3fish.py)
MIN_RATIOS = 50               # FISH_FACTORS = 'fit': fewer ratios for a fish: it gets the median over all ratios
MIN_RATIOS_ELECTRODE = 10     # fewer least-squares terms for a signed electrode: the median of the fitted electrode factors
GOOD_FRAMES = 'results/eod_assignment_3_fish_silent/mask_stages/mask_qc/mask_qc_good_frames.csv'   # video, frame (kaden_3fish_mask_qc.py)
DOUBLES = 'results/eod_assignment_3_fish_silent/eod_intervals/double_detections_3fish.csv'   # video, eod_row: dropped everywhere
OUT_ROOT = os.environ.get('OUT_ROOT', C.OUT)
# recorded vector r: 'native' = as saved in each prediction file (my model: main phase, Denis: his signed peak-to-peak); 'main_phase' =
# all 10 pairs at the one sample where |V| peaks (mean-subtracted traces); 'ptp' = Denis' signed peak-to-peak per pair (max - min,
# signed by which comes first). Both from the EOD table's traces; the sign is aligned to the native one (median cosine > 0).
RECORDED = os.environ.get('RECORDED', 'native')
# electrode factors: 'loro' = pooled over the clean EODs of the OTHER recordings with the same model setting and mask stage (each with its
# own fish set's factors), applied to the held-out recording (train on 7, test on 1; no recording is scored with factors fitted on itself);
# 'per_recording' = fitted on the recording's own clean EODs
GAIN_FIT = os.environ.get('GAIN_FIT', 'loro')
ONLY_SETTINGS = [x for x in os.environ.get('ONLY_SETTINGS', '').split(',') if x]   # e.g. mine_int4,denis (default: all)
ONLY_STAGES = [x for x in os.environ.get('ONLY_STAGES', '').split(',') if x]       # e.g. final (default: all)
ASSIGN_DIR = 'assignment' if RECORDED == 'native' else f'assignment_recorded_{RECORDED}'
ORDER = [f'{p}{s}' for p in range(1, 11) for s in '+-']
L = np.array(list(C.LETTERS))


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def best(score, largest):
    '''(n,) index of the best fish per EOD (-1 where no fish has a finite score)'''
    s = np.where(np.isfinite(score), score, -np.inf if largest else np.inf)
    i = s.argmax(1) if largest else s.argmin(1)
    return np.where(np.isfinite(score).any(1), i, -1)


def law_factors(video, model):
    '''(3,) length-law fish factors of fish A, B, C in this recording for 'mine' or 'denis' (FISH_LENGTHS, column a_<model>)'''
    T = pd.read_csv(FISH_LENGTHS); T = T[T.video == video].set_index('fish')[f'a_{model}']
    assert all(x in T.index for x in C.LETTERS), f'{video}: no factor for every fish in {FISH_LENGTHS} (python fish_lengths_3fish.py)'
    return T.reindex(list(C.LETTERS)).to_numpy(float)


def clean_table(R, Pm, pole, shape, mids):
    '''the clean EODs (see FACTORS): one row per clean EOD x top pair (eod, fish, shape, pair, pole, ratio); EODs with a NaN in R
       are never clean (mask R to restrict the fit to a subset of EODs)'''
    ok = np.isfinite(Pm).all((1, 2)) & np.isfinite(R).all(1)
    f = best(shape, largest=False)
    xy = mids[:, :, ::5, ::-1] * (C.P - 1) / C.PX_PER_CM_VIDEO                                   # (n, 3, 21, 2) cm
    rows = []
    for i in np.flatnonzero(ok & (f >= 0)):
        e = f[i]
        if min(np.linalg.norm(xy[i, e][:, None] - xy[i, g][None], axis=-1).min() for g in range(3) if g != e) < ISOLATION_CM: continue
        top = np.argsort(-np.abs(R[i]))[:N_TOP_PAIRS]
        if set(top) != set(np.argsort(-np.abs(Pm[i, e]))[:N_TOP_PAIRS]): continue
        if (np.sign(R[i, top]) != np.sign(Pm[i, e, top])).any(): continue
        if (np.abs(np.delete(Pm[i], e, 0)[:, top]) > DOMINANCE * np.abs(Pm[i, e, top])).any(): continue
        rows += [dict(eod=i, fish=e, shape=shape[i, e], pair=p, pole=pole[i, e, p], ratio=abs(R[i, p]) / abs(Pm[i, e, p])) for p in top]
    T = pd.DataFrame(rows, columns=['eod', 'fish', 'shape', 'pair', 'pole', 'ratio'])
    keep = T.drop_duplicates('eod').sort_values('shape').head(N_CLEAN).eod                    # the N_CLEAN best in all
    return T[T.eod.isin(set(keep))]


def electrode_sums(R, Pm, pole, a, ev, fi):
    '''least-squares sums (num, den, n), each (20,) in ORDER, of the signed electrode factors over all 10 pairs of the clean EODs ev
       given to fish fi: e_k = num_k / den_k. Sums of several recordings add up to their pooled fit.'''
    r = R[ev].ravel(); m = (a[fi, None] * Pm[ev, fi]).ravel(); k = (np.arange(10)[None] * 2 + pole[ev, fi]).ravel()   # all 10 pairs
    return np.bincount(k, r * m, 20), np.bincount(k, m * m, 20), np.bincount(k, minlength=20)


def electrode_factors(num, den, n):
    '''(10, 2) signed electrode factors from the least-squares sums; fewer than MIN_RATIOS_ELECTRODE terms: the median of the others'''
    ok = (n >= MIN_RATIOS_ELECTRODE) & (den > 0)
    return np.where(ok, num / np.where(den > 0, den, 1), np.median(num[ok] / den[ok]) if ok.any() else 1.0).reshape(10, 2)


def fit_factors(R, Pm, pole, shape, mids, a_law=None):
    '''fish factors a (3,), signed electrode factors E (10, 2) and the factor table, from clean EODs (see FACTORS); a_law: the
       length-law fish factors (FISH_FACTORS = 'length_law') or None (fit them).'''
    T = clean_table(R, Pm, pole, shape, mids)
    if a_law is not None: a = np.asarray(a_law, float)
    else:
        a_all = T.ratio.median() if len(T) else 1.0
        a = np.array([T.ratio[T.fish == g].median() if (T.fish == g).sum() >= MIN_RATIOS else a_all for g in range(3)])
    ev, fi = T.drop_duplicates('eod')[['eod', 'fish']].to_numpy(int).T                        # clean EODs and their fish
    num, den, n = electrode_sums(R, Pm, pole, a, ev, fi)
    E = electrode_factors(num, den, n)
    tab = pd.DataFrame({'factor': [f'fish {x}' for x in C.LETTERS] + [f'electrode {e}' for e in ORDER],
                        'value': list(a) + list(E.ravel()),
                        'n_ratios': [int((fi == g).sum()) for g in range(3)] + list(n.astype(int))})
    return a, E, tab


_TRACES = {}


def recorded_measure(video, rows, measure):
    '''(len(rows), 10) recorded main phase or signed peak-to-peak of EOD table rows (see RECORDED)'''
    if video not in _TRACES:
        from eod_assign import recorded_pair_traces_all
        t = pd.read_csv(C.EOD_TABLE.format(video=video), usecols=[f'electrode_{p}_time_{i}' for p in range(1, 11) for i in range(1, 26)])
        _TRACES.clear(); _TRACES[video] = recorded_pair_traces_all(t, list(range(1, 11)), 25)          # (n, 10, 25) mean-subtracted
    V = _TRACES[video][rows]
    if measure == 'main_phase':
        return V[np.arange(len(V)), :, np.abs(V).max(1).argmax(1)]
    if measure == 'ptp':                                                    # as EODAssignment src/data_loader.py
        return -np.ptp(V, axis=2) * np.sign(V.argmax(2) - V.argmin(2))
    raise ValueError(measure)


def load_prediction(path):
    '''one prediction file with its midlines and EOD sets: dict of setting, video, stage, R (n, 10), Pm (n, 3, 10) in the
       recording's sign, pole, frames, rows, mids (n, 3, N_MIDLINE, 2), single (not a double detection), good (good-tracking frame),
       shape (shape_l2 score (n, 3))'''
    setting, video, stage = path.split(os.sep)[-3], path.split(os.sep)[-2], os.path.basename(path)[:-4]
    z = np.load(path); R, U, sign, pole = z['recorded'], z['model_raw'], float(z['model_sign']), z['nearer_pole']
    frames, rows = z['frames'], z['eod_rows']
    if RECORDED != 'native':                                                # same EODs, the other recorded measure, sign aligned
        Rn = recorded_measure(video, rows, RECORDED); ok = np.isfinite(R).all(1)
        c = (R[ok] * Rn[ok]).sum(1) / (np.linalg.norm(R[ok], axis=1) * np.linalg.norm(Rn[ok], axis=1)); R = np.sign(np.median(c)) * Rn
    D = pd.read_csv(DOUBLES) if os.path.exists(DOUBLES) else pd.DataFrame(columns=['video', 'eod_row'])
    single = ~np.isin(rows, D.eod_row[D.video == video].to_numpy())                       # not a double detection
    zm = np.load(C.MIDLINES.replace(C.OUT, OUT_ROOT).format(video=video, stage=stage)); mf = zm['frames']
    j = np.clip(np.searchsorted(mf, frames), 0, len(mf) - 1); found = (mf[j] == frames) & (frames >= 0)
    mids = np.full((len(frames), 3, C.N_MIDLINE, 2), np.nan); mids[found] = zm['mids'][j[found]]
    Pm = sign * U                                                                               # (n, 3, 10) in the recording's sign
    G = pd.read_csv(GOOD_FRAMES) if os.path.exists(GOOD_FRAMES) else pd.DataFrame(columns=['video', 'frame'])
    good = (frames >= 0) & np.isin(frames, G.frame[G.video == video].to_numpy())
    unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
    return dict(setting=setting, video=video, stage=stage, R=R, Pm=Pm, pole=pole, frames=frames, rows=rows, mids=mids,
                single=single, good=good, shape=np.linalg.norm(unit(R)[:, None] - unit(Pm), axis=2))


def own_sums(path):
    '''this recording's fish factors and least-squares electrode sums from its own clean EODs (for the pooled leave-one-out fit)'''
    d = load_prediction(path); setting, video = d['setting'], d['video']
    a = law_factors(video, 'denis' if setting == 'denis' else 'mine') if FISH_FACTORS == 'length_law' else None
    R = np.where(d['single'][:, None], d['R'], np.nan)                                          # no double detections
    T = clean_table(R, d['Pm'], d['pole'], d['shape'], d['mids'])
    if a is None:
        a_all = T.ratio.median() if len(T) else 1.0
        a = np.array([T.ratio[T.fish == g].median() if (T.fish == g).sum() >= MIN_RATIOS else a_all for g in range(3)])
    ev, fi = T.drop_duplicates('eod')[['eod', 'fish']].to_numpy(int).T
    return (setting, d['stage'], video), electrode_sums(R, d['Pm'], d['pole'], a, ev, fi)


def run_one(job):
    '''rules, accuracy and factors of one prediction file; job = path or (path, (num, den, n) electrode sums to use instead of its own)'''
    path, sums = job if isinstance(job, tuple) else (job, None)
    d = load_prediction(path)
    setting, video, stage, R, Pm, pole = d['setting'], d['video'], d['stage'], d['R'], d['Pm'], d['pole']
    frames, rows, mids, single, good = d['frames'], d['rows'], d['mids'], d['single'], d['good']
    S = {'shape_l2': d['shape'], 'cosine': cos(R[:, None], Pm)}
    a_law = None
    if FISH_FACTORS == 'length_law':
        a_law = law_factors(video, 'denis' if setting == 'denis' else 'mine')
    a, E, tab = fit_factors(np.where(single[:, None], R, np.nan), Pm, pole, S['shape_l2'], mids, a_law)   # no double detections
    if sums is not None:                                                    # leave-one-recording-out electrode factors
        E = electrode_factors(*sums)
        tab = pd.DataFrame({'factor': [f'fish {x}' for x in C.LETTERS] + [f'electrode {e}' for e in ORDER], 'value': list(a) + list(E.ravel()),
                            'n_ratios': list(tab.n_ratios[:3]) + list(sums[2].astype(int)), 'gain_fit': GAIN_FIT})
    V = a[None, :, None] * E[np.arange(10)[None, None, :], pole] * Pm                           # predicted volts
    S['voltage_l2'] = np.linalg.norm(R[:, None] - V, axis=2)
    picks = {r: best(S[r], largest=r == 'cosine') for r in S}
    picks = {r: picks[r] for r in RULES if r in picks}
    # per-EOD table
    out = pd.DataFrame({'eod_row': rows, 'Frame': frames, **{f'pick_{r}': np.where(p >= 0, L[np.clip(p, 0, 2)], '') for r, p in picks.items()},
                        **{f'{r}_{x}': S[r][:, g] for r in S for g, x in enumerate(C.LETTERS)}})
    pp = f'{OUT_ROOT}/{ASSIGN_DIR}/picks/{setting}/{video}/{stage}.csv'; os.makedirs(os.path.dirname(pp), exist_ok=True); out.to_csv(pp, index=False)
    # accuracy
    subsets = {'all': (frames >= 0) & single, 'predicted': np.isfinite(Pm).all((1, 2)) & single, 'good': good & single}
    acc = []
    for r, p in picks.items():
        for sub, k in subsets.items():
            if not k.any(): continue
            acc.append(dict(setting=setting, video=video, stage=stage, rule=r, subset=sub, n=int(k.sum()), n_picked=int((p[k] >= 0).sum()),
                            **{f'share_{x}': (p[k] == g).mean() * 100 for g, x in enumerate(C.LETTERS)}))
    return pd.DataFrame(acc), tab.assign(setting=setting, video=video, stage=stage)


def plot(A, video, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    A = A[A.video == video]; rules = [r for r in RULES if r in set(A.rule)]; subs = [s for s in ('all', 'predicted', 'good') if s in set(A.subset)]
    stages = sorted(set(A.stage)); fig, axs = plt.subplots(len(subs), len(rules), figsize=(4.2 * len(rules), 3.3 * len(subs)), squeeze=False, sharey='row')
    for i, sub in enumerate(subs):
        for j, r in enumerate(rules):
            ax = axs[i, j]
            for s, g in A[(A.subset == sub) & (A.rule == r)].groupby('setting'):
                g = g.set_index('stage').reindex(stages); ax.plot(range(len(stages)), g.accuracy, '-o', ms=3.5, lw=1.5, label=s)
            ax.set_xticks(range(len(stages))); ax.set_xticklabels([s.split('_', 1)[-1] for s in stages], rotation=60, fontsize=7.5)
            n = A[(A.subset == sub) & (A.rule == r)].n
            ax.set_title(f'{r}, {sub} EODs (n = {n.max() if len(n) else 0:,})', fontsize=9); ax.grid(alpha=0.3)
            if j == 0: ax.set_ylabel('% given to the emitter')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=len(l), frameon=False, fontsize=9, title='model setting')
    fig.suptitle(f'3-fish {video} (emitter {A.emitter.iloc[0]}): accuracy per mask stage, rule and model setting', fontsize=11, x=0.3, ha='center')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=110); plt.close(fig)


def summary(A, out, stage='final'):
    '''accuracy over all recordings (pooled: EODs given to their emitter / EODs) per setting x rule, all vs good EODs, at one mask
       stage: accuracy_summary_<stage>.csv (+ per recording) and accuracy_summary_<stage>.png'''
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    A = A[(A.stage == stage) & A.subset.isin(['all', 'good'])].copy()
    if not len(A): return
    A['correct'] = A.accuracy / 100 * A.n
    P = A.groupby(['subset', 'setting', 'rule'])[['correct', 'n']].sum(); P['accuracy'] = P.correct / P.n * 100
    P = P.reset_index(); P.to_csv(f'{out}/accuracy_summary_{stage}.csv', index=False)
    A.pivot_table(index=['video', 'setting'], columns=['subset', 'rule'], values='accuracy').round(2).to_csv(f'{out}/accuracy_summary_{stage}_per_video.csv')
    settings = [x for x in ['mine_int1', 'mine_int4', 'mine_int16', 'mine_int100', 'denis'] if x in set(P.setting)]
    rules = [r for r in RULES if r in set(P.rule)]; subs = [x for x in ('all', 'good') if x in set(P.subset)]
    fig, axs = plt.subplots(1, len(subs), figsize=(6.8 * len(subs), 3.9), squeeze=False, gridspec_kw=dict(wspace=0.35))
    for ax, sub in zip(axs[0], subs):
        M = P[P.subset == sub].pivot(index='setting', columns='rule', values='accuracy').reindex(index=settings, columns=rules)
        nn = int(P[P.subset == sub].groupby('setting').n.max().max())
        im = ax.imshow(M.to_numpy(float), cmap='viridis', vmin=np.nanmin(P.accuracy), vmax=100, aspect='auto')
        for i in range(len(settings)):
            for j in range(len(rules)):
                v = M.iloc[i, j]
                if np.isfinite(v): ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, color='w' if v < np.nanmean(P.accuracy) else 'k')
        ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(range(len(settings))); ax.set_yticklabels(settings, fontsize=9)
        ax.set_title(f'{sub} EODs ({nn:,} over {A.video.nunique()} recordings)', fontsize=10)
    fig.colorbar(im, ax=axs[0].tolist(), label='% given to the emitter', shrink=0.85)
    fig.suptitle(f'3-fish, {stage} masks: accuracy per model setting and rule, pooled over recordings', fontsize=11)
    fig.savefig(f'{out}/accuracy_summary_{stage}.png', dpi=110, bbox_inches='tight'); plt.close(fig)
    # bars: median over the recordings of the % given to the emitter, whiskers = 25th-75th percentile over the recordings
    NAME = {s: f'numerical, σ conductivity = {s[8:]}' for s in settings if s.startswith('mine_int')}; NAME['denis'] = 'analytic'
    COL = {'mine_int1': '#9ec5f4', 'mine_int4': '#5598e7', 'mine_int16': '#256abf', 'mine_int100': '#104281', 'denis': '#eb6834'}
    RULE = {'shape_l2': 'shape L2', 'cosine': 'cosine', 'voltage_l2': 'voltage L2', 'voltage_cosine': 'voltage cosine',
}
    Q = A.groupby(['subset', 'setting', 'rule']).accuracy.describe(percentiles=[0.25, 0.5, 0.75])
    fig, axs = plt.subplots(len(subs), 1, figsize=(13, 4.3 * len(subs)), squeeze=False)
    w = 0.8 / len(settings)
    for ax, sub in zip(axs[:, 0], subs):
        q = Q.loc[sub]; lo_all = np.nanmin(q['25%'])
        for i, st in enumerate(settings):
            x = np.arange(len(rules)) + (i - (len(settings) - 1) / 2) * w
            med = np.array([q.loc[(st, r), '50%'] if (st, r) in q.index else np.nan for r in rules])
            q1 = np.array([q.loc[(st, r), '25%'] if (st, r) in q.index else np.nan for r in rules])
            q3 = np.array([q.loc[(st, r), '75%'] if (st, r) in q.index else np.nan for r in rules]); ok = np.isfinite(med)
            base = np.floor(lo_all - 1)
            ax.bar(x[ok], med[ok] - base, w * 0.9, bottom=base, color=COL.get(st, '0.5'), label=NAME.get(st, st), edgecolor='white', linewidth=0.6)
            ax.errorbar(x[ok], med[ok], yerr=[med[ok] - q1[ok], q3[ok] - med[ok]], fmt='none', ecolor='#33332f', elinewidth=1, capsize=2.5)
            for xi, v, top in zip(x[ok], med[ok], q3[ok]):
                ax.text(xi, top + 0.08 * (100 - base) / 10, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5, color='#33332f')
        ax.set_ylim(base, 100.4); ax.set_xticks(range(len(rules))); ax.set_xticklabels([RULE.get(r, r) for r in rules], fontsize=10)
        ax.set_ylabel('% given to the emitter (right fish)')
        n_tot = int(A[(A.subset == sub) & (A.setting == settings[0])].groupby('video').n.max().sum())
        ax.set_title(f'{sub} EODs ({n_tot:,} over {A.video.nunique()} recordings): bar and label = median over recordings, '
                     f'whiskers = 25th-75th percentile (y axis starts at {base:.0f}%)', fontsize=10, loc='left')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(settings), fontsize=9, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'3-fish, {stage} masks: % of EODs given to the right fish per rule and model '
                 '(σ = interior conductivity of the fish, × water)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{out}/accuracy_summary_{stage}_bars.png', dpi=110); plt.close(fig)
    print(P.pivot_table(index='setting', columns=['subset', 'rule'], values='accuracy').reindex(settings).round(2).to_string())


def best_settings(A, out, stage='final'):
    '''which interior conductivity and rule is best for my numerical model (plot 1), and the best numerical setting x rule vs the
       analytic model (plot 2), for all and good EODs. Per setting x rule: median over the recordings and their 25th-75th
       percentile; 'best' = highest median. accuracy_numerical_sigma_rule_<stage>.png, accuracy_best_numerical_vs_analytic_<stage>.png'''
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    A = A[(A.stage == stage) & A.subset.isin(['all', 'good'])]
    if not len(A): return
    Q = A.groupby(['subset', 'setting', 'rule']).accuracy.quantile([0.25, 0.5, 0.75]).unstack()
    subs = [x for x in ('all', 'good') if x in set(A.subset)]; rules = [r for r in RULES if r in set(A.rule)]
    RULE = {'shape_l2': 'shape L2', 'cosine': 'cosine', 'voltage_l2': 'voltage L2', 'voltage_cosine': 'voltage cosine'}
    RCOL = dict(zip(RULES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
    num = sorted({s for s in A.setting if s.startswith('mine_int')}, key=lambda s: float(s[8:])); sig = np.array([float(s[8:]) for s in num])
    # plot 1: numerical model only, bars: one group per rule, one bar per interior conductivity (blues, light = low sigma)
    SCOL = dict(zip(num, ['#9ec5f4', '#5598e7', '#256abf', '#104281']))
    fig, axs = plt.subplots(len(subs), 1, figsize=(11.5, 4.1 * len(subs)), squeeze=False)
    best = {}; w = 0.8 / len(num)
    for ax, sub in zip(axs[:, 0], subs):
        M = Q.loc[sub][0.5]; M = M[[i for i in M.index if i[0] in num]]; best[sub] = M.idxmax()
        lo = np.floor(min(Q.loc[(sub, st, r), 0.25] for st in num for r in rules if (sub, st, r) in Q.index) - 0.5)
        for i, st in enumerate(num):
            x = np.arange(len(rules)) + (i - (len(num) - 1) / 2) * w
            q = np.array([Q.loc[(sub, st, r)].to_numpy() if (sub, st, r) in Q.index else [np.nan] * 3 for r in rules])
            ax.bar(x, q[:, 1] - lo, w * 0.9, bottom=lo, color=SCOL[st], edgecolor='white', linewidth=0.6, label=f'σ conductivity = {st[8:]}')
            ax.errorbar(x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt='none', ecolor='#33332f', elinewidth=1, capsize=2.5)
            for xi, (a, m, b), r in zip(x, q, rules):
                star = (st, r) == best[sub]
                ax.text(xi, b + 0.025 * (100.3 - lo), f'{m:.1f}' + ('\n★' if star else ''), ha='center', va='bottom', fontsize=8,
                        color='#33332f', weight='bold' if star else 'normal')
        ax.set_xticks(range(len(rules))); ax.set_xticklabels([RULE.get(r, r) for r in rules], fontsize=10); ax.set_ylim(lo, 100.3)
        ax.set_ylabel('% given to the emitter (right fish)')
        n_tot = int(A[(A.subset == sub) & (A.setting == num[0])].groupby('video').n.max().sum())
        st, r = best[sub]
        ax.set_title(f'{sub} EODs ({n_tot:,}, {A.video.nunique()} recordings); ★ best median: {RULE[r]}, σ = {st[8:]} '
                     f'(y axis starts at {lo:.0f}%)', fontsize=10, loc='left')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(num), fontsize=9, frameon=False, loc='lower center')
    fig.suptitle(f'3-fish, {stage} masks, numerical model: which conductivity and rule?\n(bar and label = median over recordings, '
                 'whiskers = 25th-75th percentile; σ = fish interior conductivity × water)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{out}/accuracy_numerical_sigma_rule_{stage}.png', dpi=110); plt.close(fig)
    # plot 2: the best numerical setting x rule vs the analytic model's rules
    fig, axs = plt.subplots(1, len(subs), figsize=(6.4 * len(subs), 4.4), squeeze=False)
    for ax, sub in zip(axs[0], subs):
        st, r = best[sub]
        items = [(f'numerical\nσ = {st[8:]}, {RULE[r]}', Q.loc[(sub, st, r)].to_numpy(), '#256abf')]
        items += [(f'analytic\n{RULE[x]}', Q.loc[(sub, 'denis', x)].to_numpy(), '#eb6834') for x in rules if (sub, 'denis', x) in Q.index]
        lo = np.floor(min(q[0] for _, q, _ in items) - 0.5)
        for i, (lab, q, c) in enumerate(items):
            ax.bar(i, q[1] - lo, 0.62, bottom=lo, color=c, edgecolor='white')
            ax.errorbar(i, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], color='#33332f', elinewidth=1, capsize=3)
            ax.text(i, q[2] + 0.03 * (100.3 - lo), f'{q[1]:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(items))); ax.set_xticklabels([x[0] for x in items], fontsize=8.5); ax.set_ylim(lo, 100.3)
        ax.set_ylabel('% given to the emitter (right fish)')
        ax.set_title(f'{sub} EODs (y axis starts at {lo:.0f}%)', fontsize=10, loc='left')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
    fig.suptitle(f'3-fish, {stage} masks: best numerical setting vs the analytic model (bar = median over recordings, '
                 'whiskers = 25th-75th percentile)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(f'{out}/accuracy_best_numerical_vs_analytic_{stage}.png', dpi=110); plt.close(fig)
    print('best numerical (highest median over recordings):', {k: f'{v[0]} {v[1]}' for k, v in best.items()})


def analytic_only(A, out, stage='final'):
    '''the analytic model (Denis) alone: % given to the emitter per rule, all vs good EODs (median over the recordings, whiskers
       25th-75th percentile): accuracy_analytic_rules_<stage>.png'''
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    A = A[(A.stage == stage) & (A.setting == 'denis') & A.subset.isin(['all', 'good'])]
    if not len(A): return
    rules = [r for r in RULES if r in set(A.rule)]; subs = [x for x in ('all', 'good') if x in set(A.subset)]
    RULE = {'shape_l2': 'shape L2', 'cosine': 'cosine\n(= analytic\'s own rule)', 'voltage_l2': 'voltage L2', 'voltage_cosine': 'voltage cosine'}
    Q = A.groupby(['subset', 'rule']).accuracy.quantile([0.25, 0.5, 0.75]).unstack()
    fig, axs = plt.subplots(len(subs), 1, figsize=(8, 3.9 * len(subs)), squeeze=False)
    for ax, sub in zip(axs[:, 0], subs):
        q = Q.loc[sub].reindex(rules); best = q[0.5].idxmax(); lo = np.floor(q[0.25].min() - 0.5)
        for i, r in enumerate(rules):
            a_, m, b_ = q.loc[r]
            ax.bar(i, m - lo, 0.6, bottom=lo, color='#eb6834', edgecolor='white')
            ax.errorbar(i, m, yerr=[[m - a_], [b_ - m]], color='#33332f', elinewidth=1, capsize=3)
            ax.text(i, b_ + 0.025 * (100.3 - lo), f'{m:.1f}' + ('\n★' if r == best else ''), ha='center', va='bottom', fontsize=9,
                    weight='bold' if r == best else 'normal', color='#33332f')
        ax.set_xticks(range(len(rules))); ax.set_xticklabels([RULE.get(r, r) for r in rules], fontsize=9.5); ax.set_ylim(lo, 100.3)
        ax.set_ylabel('% given to the emitter (right fish)')
        n_tot = int(A[A.subset == sub].groupby('video').n.max().sum())
        ax.set_title(f'{sub} EODs ({n_tot:,}, {A.video.nunique()} recordings); ★ best median (y axis starts at {lo:.0f}%)', fontsize=10, loc='left')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
    fig.suptitle(f'3-fish, {stage} masks, analytic model: % of EODs given to the right fish per rule\n'
                 '(bar and label = median over recordings, whiskers = 25th-75th percentile)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(f'{out}/accuracy_analytic_rules_{stage}.png', dpi=110); plt.close(fig)


def numerical_vs_analytic(A, out, stage='final', numerical='mine_int4', note=''):
    '''my numerical model at one interior conductivity vs the analytic model: % given to the emitter per rule, all vs good EODs
       (median over the recordings, whiskers 25th-75th percentile): accuracy_<numerical>_vs_analytic_<stage>.png; note: appended to
       the title (e.g. how the electrode factors were fitted)'''
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    pair = [numerical, 'denis']; A = A[(A.stage == stage) & A.setting.isin(pair) & A.subset.isin(['all', 'good'])]
    if A.setting.nunique() < 2: return
    rules = [r for r in RULES if r in set(A.rule)]; subs = [x for x in ('all', 'good') if x in set(A.subset)]
    RULE = {'shape_l2': 'shape L2', 'cosine': 'cosine', 'voltage_l2': 'voltage L2'}
    NAME = {numerical: f'numerical, σ conductivity = {numerical[8:]}', 'denis': 'analytic'}; COL = {numerical: '#5598e7', 'denis': '#eb6834'}
    Q = A.groupby(['subset', 'setting', 'rule']).accuracy.quantile([0.25, 0.5, 0.75]).unstack(); w = 0.36
    fig, axs = plt.subplots(len(subs), 1, figsize=(9, 3.9 * len(subs)), squeeze=False)
    for ax, sub in zip(axs[:, 0], subs):
        lo = np.floor(Q.loc[sub][0.25].min() - 0.5)
        for k, st in enumerate(pair):
            for i, r in enumerate(rules):
                a_, m, b_ = Q.loc[(sub, st, r)]; x = i + (k - 0.5) * w
                ax.bar(x, m - lo, w * 0.92, bottom=lo, color=COL[st], edgecolor='white', label=NAME[st] if i == 0 else None)
                ax.errorbar(x, m, yerr=[[m - a_], [b_ - m]], color='#33332f', elinewidth=1, capsize=3)
                ax.text(x, b_ + 0.025 * (100.3 - lo), f'{m:.1f}', ha='center', va='bottom', fontsize=9, color='#33332f')
        ax.set_xticks(range(len(rules))); ax.set_xticklabels([RULE.get(r, r) for r in rules], fontsize=10); ax.set_ylim(lo, 100.3)
        ax.set_ylabel('% given to the emitter (right fish)')
        n_tot = int(A[(A.subset == sub) & (A.setting == numerical)].groupby('video').n.max().sum())
        ax.set_title(f'{sub} EODs ({n_tot:,}, {A.video.nunique()} recordings; y axis starts at {lo:.0f}%)', fontsize=10, loc='left')
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e6e6e3', lw=0.8); ax.set_axisbelow(True)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, fontsize=9.5, frameon=False, loc='lower center')
    fig.suptitle(f'3-fish, {stage} masks: numerical (σ = {numerical[8:]}) vs analytic model, % of EODs given to the right fish per rule\n'
                 '(bar and label = median over recordings, whiskers = 25th-75th percentile)' + (f'\n{note}' if note else ''),
                 fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{out}/accuracy_{numerical}_vs_analytic_{stage}.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    files = sorted(glob.glob(f'{OUT_ROOT}/predictions/*/*/*.npz'))
    files = [f for f in files if (not ONLY_SETTINGS or f.split(os.sep)[-3] in ONLY_SETTINGS)
             and (not ONLY_STAGES or os.path.basename(f)[:-4] in ONLY_STAGES)]
    print(f'recorded vector: {RECORDED}; output {OUT_ROOT}/{ASSIGN_DIR}', flush=True)
    print(f'{len(files)} prediction files in {OUT_ROOT}/predictions', flush=True)
    print(f'electrode factors: {GAIN_FIT}', flush=True)
    with mp.get_context('fork').Pool(a.workers) as pool:
        if GAIN_FIT == 'loro':                                              # pass 1: each recording's sums; pass 2: score with the others'
            own = dict(pool.map(own_sums, files)); tot = {}
            for (st, sg, v), x in own.items(): tot[(st, sg)] = [t + y for t, y in zip(tot.get((st, sg), [0, 0, 0]), x)]
            jobs = []
            for f in files:
                k = (f.split(os.sep)[-3], os.path.basename(f)[:-4], f.split(os.sep)[-2])
                jobs.append((f, tuple(t - y for t, y in zip(tot[k[:2]], own[k]))))
            res = pool.map(run_one, jobs)
        else:
            res = pool.map(run_one, files)
    A = pd.concat([r[0] for r in res], ignore_index=True); F = pd.concat([r[1] for r in res], ignore_index=True)
    out = f'{OUT_ROOT}/{ASSIGN_DIR}'; os.makedirs(out, exist_ok=True)
    em = []                                                    # the emitter: given, or the shape rule's majority fish
    for v, g in A[(A.rule == 'shape_l2') & (A.subset == 'predicted')].groupby('video'):
        sh = {x: np.average(g[f'share_{x}'], weights=g.n) for x in C.LETTERS}
        e = C.EMITTER.get(v, max(sh, key=sh.get))
        em.append(dict(video=v, emitter=e, source='given' if v in C.EMITTER else 'inferred', shape_l2_share=round(sh[e], 2)))
    EM = pd.DataFrame(em).set_index('video'); EM.to_csv(f'{out}/emitter.csv'); print(EM.to_string(), flush=True)
    A['emitter'] = A.video.map(EM.emitter)
    A['accuracy'] = [r[f'share_{r.emitter}'] if isinstance(r.emitter, str) else np.nan for _, r in A.iterrows()]
    A.to_csv(f'{out}/accuracy.csv', index=False); F.to_csv(f'{out}/factors.csv', index=False)
    for v in sorted(set(A.video)):
        if A[A.video == v].accuracy.notna().any(): plot(A, v, f'{out}/accuracy_vs_stage_{v}.png')
    print(A[A.subset == 'all'].pivot_table(index=['video', 'setting', 'stage'], columns='rule', values='accuracy').round(2).to_string())
    summary(A, out); best_settings(A, out); analytic_only(A, out); numerical_vs_analytic(A, out)
    print(f'written to {out}/')
