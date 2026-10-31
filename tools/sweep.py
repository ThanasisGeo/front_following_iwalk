# =====================================================================================================
# tools/sweep.py: train + evaluate one model config over several seeds x cross-validation folds x variants,
# then pool the results. Designed for one Kaggle session (and resumable across sessions).
# =====================================================================================================
#   python tools/sweep.py --cfg real_PathUNetPoseSmallCV --name cv5 --seeds 0 1 2 --folds all --mirror both
#
# For every combination it runs train_path.py (--test_seq <fold>, or for DARKO --set dataset_specs.test_fold=<fold>)
# and eval_path.py (best + last checkpoint):
#   run folder  results/<cfg>_<name>_<variant>_s<seed>_f<fold>/   (train log in train_stdout.txt)
#   variants    base (no augmentation, augment.mirror=0) and/or mirror (augment.mirror=0.5)
#   folds       Real-IM: each sequence becomes the test set once, the other four are used for training/validation.
#               DARKO: fold k = trial fold k of dataset_specs.n_folds (--set dataset_specs.test_fold=k); every
#               trial is a test trial exactly once over the n_folds folds.
# Finished runs (both evals present) are skipped, unfinished ones resume from trainstate_last.pt, so the same
# command can be re-run in a new session. After a run completes, its trainstate is deleted (only
# pathnet_best.pt and pathnet_<epochs>.pt are kept, ~40 MB per run).
#
# Pooling: for each variant / checkpoint / seed, the per-window results of all folds are concatenated. Every
# window of the dataset is then a test window exactly once (Real-IM: all ~1800 windows, ~86+ turns instead of 26;
# DARKO: all ~11.8K windows of the 508 trials).
# The summary reports mean +- std over seeds, plus constant velocity on the same windows.
#   results/sweep_<name>/summary.md, summary.csv
# SIGTERM (e.g. Kaggle's timeout) stops after the current run's training has saved; the summary covers the
# runs finished so far.
# =====================================================================================================
import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.path_utils import load_cfg, test_set_name  # noqa: E402

SUBSETS = ['all', 'straight', 'turn', 'turn_heading', 'turn_new', 'turn_ongoing']
METRICS = [('ade', 'ADE'), ('fde', 'FDE'), ('err_1s', '@1s'), ('err_2s', '@2s'), ('lat_err', 'sideways@end')]
VARIANTS = {'base': ['augment.mirror=0'], 'mirror': ['augment.mirror=0.5']}
STOP = {'requested': False}


def _stop(signum, frame):
    print(f'\n[sweep] signal {signum}: finishing the current step, then stopping', flush=True)
    STOP['requested'] = True


def run_dir(cfg, exp):
    return os.path.join('results', f'{cfg}_{exp}')


def fold_args(cfg_name, fold):
    # train/eval arguments that select a fold, and the test-set name eval_path.py uses in its output folder.
    cfg = load_cfg(cfg_name)
    if cfg['dataset'] == 'DARKO':
        over = [f'dataset_specs.test_fold={fold}']
        cfg['dataset_specs']['test_fold'] = int(fold)
        return [], over, test_set_name(cfg)
    return ['--test_seq', fold], [], fold


def per_window_path(cfg, exp, ckpt, fold):
    return os.path.join(run_dir(cfg, exp), 'eval', f'pathnet_{ckpt}_{fold_args(cfg, fold)[2]}', 'per_window.csv')


def read_per_window(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ('key', 'seq')}


def one_run(args, variant, seed, fold, py):
    exp = f'{args.name}_{variant}_s{seed}_f{fold}'
    rdir = run_dir(args.cfg, exp)
    done = all(os.path.exists(per_window_path(args.cfg, exp, c, fold)) for c in ('best', 'last'))
    if done:
        print(f'[sweep] skip (done): {exp}', flush=True)
        return 0
    os.makedirs(rdir, exist_ok=True)
    trainstate = os.path.join(rdir, 'models', 'trainstate_last.pt')
    final = os.path.join(rdir, 'models', f'pathnet_{args.epochs:03d}.pt')
    if not os.path.exists(final) or os.path.exists(trainstate):
        fold_cli, fold_set, _ = fold_args(args.cfg, fold)
        cmd = [py, '-u', 'train_path.py', '--cfg', args.cfg, '--exp', exp, '--seed', str(seed)] + fold_cli + [
               '--epochs', str(args.epochs), '--batch_size', str(args.batch_size), '--select', args.select,
               '--num_workers', str(args.num_workers), '--save_freq', str(args.epochs)]
        overrides = VARIANTS[variant] + fold_set + args.set
        if overrides:
            cmd += ['--set'] + overrides
        if os.path.exists(trainstate):
            cmd += ['--resume', trainstate]
        print(f'[sweep] train {exp}', flush=True)
        with open(os.path.join(rdir, 'train_stdout.txt'), 'a') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'TQDM_DISABLE': '1'}).returncode
        if rc != 0:
            print(f'[sweep] training {exp} exited with {rc} (124 = stopped by signal); see {rdir}/train_stdout.txt', flush=True)
            return rc
    for ckpt in ('best', 'last'):
        if os.path.exists(per_window_path(args.cfg, exp, ckpt, fold)):
            continue
        cmd = [py, '-u', 'eval_path.py', '--cfg', args.cfg, '--exp', exp, '--ckpt', ckpt, '--num_workers', str(args.num_workers)]
        with open(os.path.join(rdir, f'eval_{ckpt}_stdout.txt'), 'w') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'TQDM_DISABLE': '1'}).returncode
        if rc != 0:
            print(f'[sweep] eval {exp} ({ckpt}) failed with {rc}; see {rdir}/eval_{ckpt}_stdout.txt', flush=True)
            return rc
    if os.path.exists(trainstate):
        os.remove(trainstate)  # finished: the resume file is no longer needed
    w = read_per_window(per_window_path(args.cfg, exp, 'last', fold))
    t = w['turn'] > 0
    print(f"[sweep] done {exp}: last ckpt FDE {w['m_fde'].mean()*1000:.0f} mm (CV {w['cv_fde'].mean()*1000:.0f}), "
          f"turns ({int(t.sum())}) {w['m_fde'][t].mean()*1000:.0f} mm", flush=True)
    return 0


def summarize(args, folds):
    out_dir = os.path.join('results', f'sweep_{args.name}')
    os.makedirs(out_dir, exist_ok=True)
    rows, lines = [], []
    lines.append(f'# Sweep `{args.name}`: {args.cfg}\n')
    lines.append(f'Seeds {args.seeds}, folds {folds}, {args.epochs} epochs, extra overrides {args.set or "none"}. '
                 f'Pooled over folds; mean ± std over seeds; mm.\n')
    cv_done = False
    for ckpt in ('last', 'best'):
        lines.append(f'\n## Checkpoint: {ckpt}\n')
        header = '| variant | seeds | subset | n | ' + ' | '.join(l for _, l in METRICS) + ' |'
        lines += [header, '|' + '---|' * (4 + len(METRICS))]
        cv_rows = {}
        for variant in args.variants:
            per_seed = []
            for seed in args.seeds:
                parts = [per_window_path(args.cfg, f'{args.name}_{variant}_s{seed}_f{f}', ckpt, f) for f in folds]
                if not all(os.path.exists(p) for p in parts):
                    continue
                ws = [read_per_window(p) for p in parts]
                per_seed.append({k: np.concatenate([w[k] for w in ws]) for k in ws[0]})
            if not per_seed:
                lines.append(f'| {variant} | 0 | (no complete seed yet) |' + ' |' * (1 + len(METRICS)))
                continue
            for s in SUBSETS:
                vals = {}
                for m, _ in METRICS:
                    v = [w[f'm_{m}'][w[s] > 0].mean() * 1000 for w in per_seed if (w[s] > 0).any()]
                    vals[m] = (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
                n = int((per_seed[0][s] > 0).sum())
                cells = ' | '.join(f'{vals[m][0]:.0f} ± {vals[m][1]:.0f}' for m, _ in METRICS)
                lines.append(f'| {variant} | {len(per_seed)} | {s} | {n} | {cells} |')
                rows.append({'ckpt': ckpt, 'variant': variant, 'seeds': len(per_seed), 'subset': s, 'n': n,
                             **{f'{m}_mean': round(vals[m][0], 1) for m, _ in METRICS},
                             **{f'{m}_std': round(vals[m][1], 1) for m, _ in METRICS}})
                w0 = per_seed[0]
                cv_rows[s] = (n, {m: w0[f'cv_{m}'][w0[s] > 0].mean() * 1000 for m, _ in METRICS})
        if cv_rows and not cv_done:
            for s, (n, v) in cv_rows.items():
                lines.append(f'| constant velocity | – | {s} | {n} | ' + ' | '.join(f'{v[m]:.0f}' for m, _ in METRICS) + ' |')
            cv_done = True
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if rows:
        with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print('\n'.join(lines))
    print(f'\n[sweep] summary written to {out_dir}/summary.md and summary.csv')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='real_PathUNetPoseSmallCV')
    p.add_argument('--name', default='cv5')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--folds', nargs='+', default=['all'], help="'all' = every sequence once, 'default' = cfg test list, or names")
    p.add_argument('--mirror', choices=['off', 'on', 'both'], default='both')
    p.add_argument('--set', nargs='*', default=[], help='cfg overrides applied to every run')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--select', default='val_ade')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--summary_only', action='store_true')
    args = p.parse_args()
    args.variants = {'off': ['base'], 'on': ['mirror'], 'both': ['base', 'mirror']}[args.mirror]

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    cfg = load_cfg(args.cfg)
    ds = cfg['dataset_specs']
    if cfg['dataset'] == 'DARKO':
        seqs, default = [str(k) for k in range(ds.get('n_folds', 7))], [str(ds.get('test_fold', 0))]
    else:
        seqs = list(ds['train']) + [s for s in ds['test'] if s not in ds['train']]
        default = list(ds['test'])
    folds = seqs if args.folds == ['all'] else (default if args.folds == ['default'] else args.folds)

    if not args.summary_only:
        jobs = [(v, s, f) for s in args.seeds for v in args.variants for f in folds]  # seed-major: complete seeds first
        print(f'[sweep] {len(jobs)} runs: variants {args.variants} x seeds {args.seeds} x folds {folds}', flush=True)
        for k, (variant, seed, fold) in enumerate(jobs, 1):
            if STOP['requested']:
                break
            print(f'[sweep] ({k}/{len(jobs)})', end=' ', flush=True)
            rc = one_run(args, variant, seed, fold, sys.executable)
            if rc == 124 or STOP['requested']:
                break
    summarize(args, folds)


if __name__ == '__main__':
    main()
