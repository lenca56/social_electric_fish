"""Step 3 of 3 (see cluster_3fish_common.py): assignment rules on the raw predictions of step 2, for every recording x mask stage x
model setting found, with the accuracy against the known emitter. Runs in a few minutes on a laptop (no field solves).

RULES (recorded r: 10 pairs; model p_f = model_sign * model_raw of fish f; each EOD goes to the best fish)
  shape_l2        smallest || r / max|r| - p_f / max|p_f| ||  (shape only, each pattern scaled to max |v| = 1)
  cosine          largest cos(r, p_f)  (shape only; L2 between patterns scaled to unit norm ranks exactly like this, 2 - 2 cos)
  voltage_l2      smallest || r - a_f e_k p_f ||  in volts, with fitted factors (below)
  voltage_cosine  largest cos(r, a_f e_k p_f)  (the electrode factors reweight the pairs)
  denis_own       Denis' own pick (his similarity), Denis setting only

FACTORS (per recording x stage x setting, fitted without the ground truth, as for the 4-fish recordings)
  clean EODs: every fish has a prediction; the shape_l2 fish f is at least ISOLATION_CM from both other fish (midline to midline);
  its N_TOP_PAIRS loudest recorded pairs are its N_TOP_PAIRS largest predicted pairs, with the model's sign; the other fish predict
  at most DOMINANCE x its |p| there; per fish the N_CLEAN best by shape_l2.
  ratios |r_p| / |p_f,p| on those pairs; fish factor a_f = median of fish f's ratios (a fish with fewer than MIN_RATIOS, e.g. a silent
  fish of a control, gets the median of all ratios); signed electrode factor e_k = median of ratio / a_f over the ratios of
  electrode k = the pole of pair p nearer to fish f (fewer than MIN_RATIOS_ELECTRODE: e_k = 1).

ACCURACY (recordings in cluster_3fish_common.EMITTER): % of EODs given to the emitter, on three EOD sets
  all        every EOD with a video frame (no pick counts as wrong)
  predicted  EODs where every fish has a prediction (all midlines at that stage)
  good       EODs on frames that pass the mask QC of the final masks (GOOD_FRAMES), the same EODs at every stage

OUTPUT  results/eod_assignment_3_fish_silent/mask_stages/assignment/
  accuracy.csv                        setting, video, stage, rule, subset, n, n_picked, accuracy (%), shares A/B/C (%)
  factors.csv                         setting, video, stage, factor (fish A.., electrode 1+..), value, n_ratios
  picks/<setting>/<rec>/<stage>.csv   per EOD: eod_row, Frame, pick and score per rule
  accuracy_vs_stage_<rec>.png         accuracy per rule (columns) and EOD set (rows), one line per setting

    python assign_3fish_stage_rules.py [--workers 4]
"""
import os, sys, glob, json, argparse
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
import cluster_3fish_common as C

# ================================================================================================================ hyperparameters
RULES = ['shape_l2', 'cosine', 'voltage_l2', 'voltage_cosine', 'denis_own']
ISOLATION_CM = 10.0           # clean EODs: the emitter at least this far from both other fish
N_TOP_PAIRS = 2               # clean EODs: the loudest recorded pairs = the emitter's largest predicted pairs; ratios on these
DOMINANCE = 0.5               # clean EODs: other fish predict at most this x the emitter's |p| on those pairs
N_CLEAN = 1000                # clean EODs per fish (best shape_l2)
MIN_RATIOS = 50               # fewer ratios for a fish: it gets the median over all ratios
MIN_RATIOS_ELECTRODE = 10     # fewer ratios for a signed electrode: factor 1
GOOD_FRAMES = 'results/eod_assignment_3_fish_silent/with_latest_run_tracking/mask_qc_good_frames.csv'   # 14_12_25 only
GOOD_FRAMES_FOR = '2024-10-15T14_12_25'
OUT_ROOT = os.environ.get('OUT_ROOT', C.OUT)
ORDER = [f'{p}{s}' for p in range(1, 11) for s in '+-']
L = np.array(list(C.LETTERS))


def cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def best(score, largest):
    '''(n,) index of the best fish per EOD (-1 where no fish has a finite score)'''
    s = np.where(np.isfinite(score), score, -np.inf if largest else np.inf)
    i = s.argmax(1) if largest else s.argmin(1)
    return np.where(np.isfinite(score).any(1), i, -1)


def fit_factors(R, Pm, pole, shape, mids):
    '''fish factors a (3,), signed electrode factors E (10, 2) and the factor table, from clean EODs (see FACTORS).'''
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
    keep = T.drop_duplicates('eod').sort_values('shape').groupby('fish').head(N_CLEAN).eod
    T = T[T.eod.isin(set(keep))]
    a_all = T.ratio.median() if len(T) else 1.0
    a = np.array([T.ratio[T.fish == g].median() if (T.fish == g).sum() >= MIN_RATIOS else a_all for g in range(3)])
    T = T.assign(g=T.ratio / a[T.fish.to_numpy().astype(int)]) if len(T) else T.assign(g=[])
    E = np.ones((10, 2))
    for p in range(10):
        for s in range(2):
            k = (T.pair == p) & (T.pole == s)
            if k.sum() >= MIN_RATIOS_ELECTRODE: E[p, s] = T.g[k].median()
    tab = pd.DataFrame({'factor': [f'fish {x}' for x in C.LETTERS] + [f'electrode {e}' for e in ORDER],
                        'value': list(a) + list(E.ravel()),
                        'n_ratios': [int((T.fish == g).sum()) for g in range(3)] +
                                    [int(((T.pair == p) & (T.pole == s)).sum()) for p in range(10) for s in range(2)]})
    return a, E, tab


def run_one(path):
    '''rules, accuracy and factors of one prediction file'''
    setting, video, stage = path.split(os.sep)[-3], path.split(os.sep)[-2], os.path.basename(path)[:-4]
    z = np.load(path); R, U, sign, pole = z['recorded'], z['model_raw'], float(z['model_sign']), z['nearer_pole']
    frames, rows = z['frames'], z['eod_rows']
    zm = np.load(C.MIDLINES.replace(C.OUT, OUT_ROOT).format(video=video, stage=stage)); mf = zm['frames']
    j = np.clip(np.searchsorted(mf, frames), 0, len(mf) - 1); found = (mf[j] == frames) & (frames >= 0)
    mids = np.full((len(frames), 3, C.N_MIDLINE, 2), np.nan); mids[found] = zm['mids'][j[found]]
    Pm = sign * U                                                                               # (n, 3, 10) in the recording's sign
    unit = lambda x: x / np.abs(x).max(axis=-1, keepdims=True)
    S = {'shape_l2': np.linalg.norm(unit(R)[:, None] - unit(Pm), axis=2), 'cosine': cos(R[:, None], Pm)}
    a, E, tab = fit_factors(R, Pm, pole, S['shape_l2'], mids)
    V = a[None, :, None] * E[np.arange(10)[None, None, :], pole] * Pm                           # predicted volts
    S['voltage_l2'] = np.linalg.norm(R[:, None] - V, axis=2); S['voltage_cosine'] = cos(R[:, None], V)
    picks = {r: best(S[r], largest=r in ('cosine', 'voltage_cosine')) for r in S}
    if 'pick_similarity' in z.files:
        picks['denis_own'] = np.array([C.LETTERS.index(x) if x else -1 for x in z['pick_similarity']])
    picks = {r: picks[r] for r in RULES if r in picks}
    # per-EOD table
    out = pd.DataFrame({'eod_row': rows, 'Frame': frames, **{f'pick_{r}': np.where(p >= 0, L[np.clip(p, 0, 2)], '') for r, p in picks.items()},
                        **{f'{r}_{x}': S[r][:, g] for r in S for g, x in enumerate(C.LETTERS)}})
    pp = f'{OUT_ROOT}/assignment/picks/{setting}/{video}/{stage}.csv'; os.makedirs(os.path.dirname(pp), exist_ok=True); out.to_csv(pp, index=False)
    # accuracy
    good = (np.isin(frames, pd.read_csv(GOOD_FRAMES).frame.to_numpy()) if video == GOOD_FRAMES_FOR and os.path.exists(GOOD_FRAMES)
            else np.zeros(len(frames), bool))
    subsets = {'all': frames >= 0, 'predicted': np.isfinite(Pm).all((1, 2)), 'good': good}
    em = C.LETTERS.index(C.EMITTER[video]) if video in C.EMITTER else None
    acc = []
    for r, p in picks.items():
        for sub, k in subsets.items():
            if not k.any(): continue
            acc.append(dict(setting=setting, video=video, stage=stage, rule=r, subset=sub, n=int(k.sum()), n_picked=int((p[k] >= 0).sum()),
                            accuracy=(p[k] == em).mean() * 100 if em is not None else np.nan,
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
            ax.set_xticks(range(len(stages)), [s.split('_', 1)[1] for s in stages], rotation=60, fontsize=7.5)
            n = A[(A.subset == sub) & (A.rule == r)].n
            ax.set_title(f'{r}, {sub} EODs (n = {n.max() if len(n) else 0:,})', fontsize=9); ax.grid(alpha=0.3)
            if j == 0: ax.set_ylabel('% given to the emitter')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper right', ncol=len(l), frameon=False, fontsize=9, title='model setting')
    fig.suptitle(f'3-fish {video}: accuracy per mask stage, rule and model setting', fontsize=11, x=0.3, ha='center')
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    files = sorted(glob.glob(f'{OUT_ROOT}/predictions/*/*/*.npz'))
    print(f'{len(files)} prediction files in {OUT_ROOT}/predictions', flush=True)
    with mp.get_context('fork').Pool(a.workers) as pool: res = pool.map(run_one, files)
    A = pd.concat([r[0] for r in res], ignore_index=True); F = pd.concat([r[1] for r in res], ignore_index=True)
    out = f'{OUT_ROOT}/assignment'; os.makedirs(out, exist_ok=True)
    A.to_csv(f'{out}/accuracy.csv', index=False); F.to_csv(f'{out}/factors.csv', index=False)
    for v in sorted(set(A.video)):
        if A[A.video == v].accuracy.notna().any(): plot(A, v, f'{out}/accuracy_vs_stage_{v}.png')
    print(A[A.subset == 'all'].pivot_table(index=['setting', 'stage'], columns='rule', values='accuracy').round(2).to_string())
    print(f'written to {out}/')
