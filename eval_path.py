# =====================================================================================================
# eval_path.py: evaluate a path-only model (models/path_models.py) on the test sequence(s)
# =====================================================================================================
# Reads the predicted path directly from the model (no PoseNet), always evaluates the constant-velocity
# baseline (B0) next to it, and reports every metric for every turn subset (utils/turns.py).
#
#   python eval_path.py --cfg real_PathUNetPose --exp m1_s0 --ckpt best             # a trained run
#   python eval_path.py --cfg real_PathUNet --exp released \
#          --ckpt results/real_Path_GRU_exp/models/pathnet_3600.pt                   # B1: released weights
#
# If results/<cfg>_<exp>/config.yml exists (written by train_path.py), the model is rebuilt from it, so
# overrides and the fold used in training are applied automatically.
# Outputs in results/<cfg>_<exp>/eval/<ckpt>_<test sequences>/: metrics.json and per_window.csv.
# =====================================================================================================
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset_gta import DatasetGTA
from models.path_models import get_path_model
from utils.path_utils import (apply_overrides, apply_test_seq, constant_velocity, format_table, load_cfg, predict, score,
                              set_seed, summarize, write_per_window_csv)
from utils.sequence_order import temporal_rank


def resolve_ckpt(models_dir, ckpt):
    if os.path.isfile(ckpt):
        return ckpt
    if ckpt == 'best':
        path = os.path.join(models_dir, 'pathnet_best.pt')
    elif ckpt == 'last':
        epochs = [int(m.group(1)) for p in glob.glob(os.path.join(models_dir, 'pathnet_*.pt'))
                  if (m := re.search(r'pathnet_(\d+)\.pt$', p))]
        if not epochs:
            raise FileNotFoundError(f'no pathnet_<N>.pt in {models_dir}')
        path = os.path.join(models_dir, f'pathnet_{max(epochs):03d}.pt')
    else:
        path = os.path.join(models_dir, f'pathnet_{int(ckpt):03d}.pt')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def main(args):
    run_dir = f'results/{args.cfg}_{args.exp}'
    run_cfg = os.path.join(run_dir, 'config.yml')
    if os.path.isfile(run_cfg):
        with open(run_cfg) as f:
            cfg = yaml.safe_load(f)
        apply_overrides(cfg, args.set)
        print(f'model config: {run_cfg}')
    else:
        cfg = load_cfg(args.cfg, args.set)
        print(f'model config: cfg/{args.cfg}.yml')
    if args.test_seq:
        apply_test_seq(cfg, args.test_seq)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = resolve_ckpt(os.path.join(run_dir, 'models'), args.ckpt)
    model = get_path_model(cfg).to(device)
    model.load_ckpt(ckpt, device)
    print(f'checkpoint: {ckpt}')

    dataset = DatasetGTA('test', cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    res = predict(model, loader, cfg['dataset_specs']['root_idx'])
    sc = score(res, cfg)
    subsets = sc['subsets']
    summary = summarize(sc['errors'], sc['labels'], subsets)
    summary_cv = summarize(sc['errors_cv'], sc['labels'], subsets)

    test_seqs = cfg['dataset_specs']['test']
    print(f"\n{cfg['model_name']} ({os.path.basename(ckpt)}) on {', '.join(test_seqs)}: {len(res['keys'])} windows")
    print(format_table(summary, summary_cv, subsets, model_name=cfg['model_name']))
    print(f"ground truth itself in occupied cells: {sc['gt_collision'].mean() * 100:.1f}% of points "
          f"(reference for coll%)")

    # Error per future frame (0.2 s ... 3.0 s): shows WHERE in the horizon a model fails (e.g. the near future).
    cv_pred = constant_velocity(res['hist'], cfg['dataset_specs']['n_pred'], sc['thresholds']['vel_frames'])
    err_m = np.linalg.norm(res['pred'] - res['gt'], axis=-1) * 1000
    err_cv = np.linalg.norm(cv_pred - res['gt'], axis=-1) * 1000
    per_frame = {}
    print(f"\nerror per future frame (mm), frames at {', '.join(f'{(t + 1) * 0.2:.1f}' for t in range(err_m.shape[1]))} s")
    for s_name in ('all', 'straight', 'turn'):
        mask = sc['labels'][s_name]
        if mask.any():
            per_frame[s_name] = {'model': err_m[mask].mean(0).round(1).tolist(), 'cv': err_cv[mask].mean(0).round(1).tolist()}
            print(f"  {s_name:9s} model: " + ' '.join(f'{v:5.0f}' for v in err_m[mask].mean(0)))
            print(f"  {'':9s} CV:    " + ' '.join(f'{v:5.0f}' for v in err_cv[mask].mean(0)))

    # Per-window time stamps (the .npz files store windows shuffled), for paired / block-bootstrap comparisons.
    root_idx = cfg['dataset_specs']['root_idx']
    ranks = {dataset.idx2scene[sub]: temporal_rank(dataset.pose[sub], root_idx)[0] for sub in dataset.pose}
    seq_fidx = [k.rsplit('.', 1) for k in res['keys']]
    t_rank = np.array([ranks[s][int(i)] for s, i in seq_fidx])

    tag = os.path.splitext(os.path.basename(ckpt))[0] if os.path.isfile(args.ckpt) else f'pathnet_{args.ckpt}'
    out_dir = args.out_dir or os.path.join(run_dir, 'eval', f"{tag}_{'+'.join(test_seqs)}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({'model_name': cfg['model_name'], 'pose_input': cfg['model_specs'].get('pose_input'),
                   'ckpt': ckpt, 'test_seqs': test_seqs, 'n_windows': len(res['keys']),
                   'thresholds': sc['thresholds'], 'gt_collision': float(sc['gt_collision'].mean()),
                   'model': summary, 'constant_velocity': summary_cv, 'per_frame_mm': per_frame}, f, indent=1)
    columns = {'seq': [s for s, _ in seq_fidx], 'fidx': [int(i) for _, i in seq_fidx], 't_rank': t_rank}
    columns.update(sc['features'])
    columns.update(sc['labels'])
    columns.update({f'm_{k}': v for k, v in sc['errors'].items()})
    columns.update({f'cv_{k}': v for k, v in sc['errors_cv'].items()})
    write_per_window_csv(os.path.join(out_dir, 'per_window.csv'), res['keys'], columns)
    print(f'saved {out_dir}/metrics.json and per_window.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='real_PathUNetPose', help='cfg name (cfg/<name>.yml)')
    parser.add_argument('--exp', default='exp', help='run name -> results/<cfg>_<exp>/')
    parser.add_argument('--ckpt', default='best', help="'best', 'last', an epoch number, or a checkpoint path")
    parser.add_argument('--test_seq', default=None, help='evaluate on this sequence instead of the cfg test list')
    parser.add_argument('--set', nargs='*', default=[], help='cfg overrides, e.g. model_specs.n_feats_pose_path=64')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default=None)
    main(parser.parse_args())
