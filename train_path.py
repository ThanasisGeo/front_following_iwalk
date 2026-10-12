# =====================================================================================================
# train_path.py: train a path-only model (models/path_models.py). No PoseNet, no pose losses.
# =====================================================================================================
# Differences from train.py (see CLAUDE/pose_conditioned_path.md §5.3):
#   * path losses only: path MSE, final position, and, for heat-map models, trajectory-map BCE and collision
#   * --seed, so runs are reproducible
#   * leak-free validation: the LAST --val_frac of each training sequence in time, with a gap so that no
#     training window shares a frame with a validation window. (The .npz files store windows shuffled, so the
#     time order is rebuilt first, see utils/sequence_order.py.) The test sequence is never used here.
#   * --epochs is the TOTAL number of epochs. --resume continues from trainstate_last.pt (model, optimizer,
#     LR scheduler, RNG states), so a run can be spread over several Kaggle sessions without LR workarounds
#   * SIGTERM/Ctrl-C (e.g. `timeout 11h ...`) finishes the current epoch, saves, and exits cleanly
#
# Files in results/<cfg>_<exp>/:
#   config.yml          effective cfg (after --set / --test_seq). eval_path.py rebuilds the model from it
#   train_metrics.csv   one row per epoch: losses, LR, validation ADE/FDE (all windows and turns)
#   models/pathnet_<N>.pt  every save_freq epochs and at the end;  models/pathnet_best.pt  best validation
#   models/trainstate_last.pt  everything needed to resume (overwritten each save)
#
#   python train_path.py --cfg real_PathUNetPose --exp m1_s0 --seed 0 --epochs 600 \
#          --init_released results/real_Path_GRU_exp/models/pathnet_3600.pt --lr 1e-5
# =====================================================================================================
import argparse
import csv
import json
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.dataset_gta import DatasetGTA
from models.path_models import get_path_model
from utils.path_utils import load_cfg, predict, score, seed_worker, set_seed, summarize
from utils.sequence_order import temporal_rank

STOP = {'requested': False}


def _request_stop(signum, frame):
    print(f'\nsignal {signum} received: stopping after this epoch and saving')
    STOP['requested'] = True


def path_loss(out, label, cfg):
    # Path-only version of train.py's loss_func. Terms are switched on by the cfg `loss:` block, and the
    # heat-map terms only apply when the model returns 'traj_map'.
    loss_cfg, root_idx = cfg['loss'], cfg['dataset_specs']['root_idx']
    traj = out['traj']
    dev = traj.device
    gt = label['pose']['pose'][:, :, root_idx, :2].to(dev).float()
    terms = {}
    if loss_cfg.get('use_traj', True):
        terms['traj'] = F.mse_loss(traj, gt) * loss_cfg.get('weight_traj', 1.0)
    if loss_cfg.get('use_final_pos', True):
        terms['final_pos'] = F.mse_loss(traj[:, -1], gt[:, -1]) * loss_cfg.get('weight_final_pos', 1.0)
    if 'traj_map' in out:
        if loss_cfg.get('use_traj_map', False):
            bce = nn.BCELoss(weight=torch.tensor(loss_cfg['weight_bce'], device=dev))
            terms['traj_map'] = bce(out['traj_map'], label['pose']['traj_map'].to(dev).float()) \
                * loss_cfg.get('weight_traj_map', 1.0)
        if loss_cfg.get('use_collision', False):
            # Penalise heat-map mass on non-free cells (|occupancy| > 0), as in train.py.
            occ = label['env']['map'].to(dev).float().abs()
            terms['collision'] = (occ * out['traj_map']).mean() * loss_cfg.get('weight_collision', 1.0)
    return sum(terms.values()), {k: v.item() for k, v in terms.items()}


def split_train_val(dataset, val_frac, root_idx, n_total):
    # Per training sequence: the latest val_frac of the windows (in time) -> validation. Training windows must
    # end before the first validation frame. Window at time t covers frames t .. t + n_total - 1.
    by_sub = {}
    for k, key in dataset.data.items():
        sub, fidx = (int(x) for x in key.split('.'))
        by_sub.setdefault(sub, []).append((k, fidx))
    train_idx, val_idx = [], []
    for sub, items in sorted(by_sub.items()):
        rank, n_chains = temporal_rank(dataset.pose[sub], root_idx)
        name = dataset.idx2scene[sub]
        if n_chains != 1:
            print(f'WARNING: {name} splits into {n_chains} continuous pieces; validation split is approximate')
        ks = np.array([k for k, _ in items])
        t = np.array([rank[f] for _, f in items])
        if val_frac <= 0:
            train_idx += ks.tolist()
            continue
        t_cut = np.quantile(t, 1 - val_frac)
        val, train = t >= t_cut, t <= t_cut - n_total
        train_idx += ks[train].tolist()
        val_idx += ks[val].tolist()
        print(f'  {name}: {train.sum()} train / {val.sum()} val windows ({(~train & ~val).sum()} dropped as gap)')
    return train_idx, val_idx


def save_trainstate(path, model, optimizer, scheduler, epoch, best, cfg, meta):
    state = {'model': model.pathnet.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best, 'cfg': cfg, 'meta': meta,
             'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)  # never leave a half-written resume file behind


def main(args):
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    run_dir = f'results/{args.cfg}_{args.exp}'
    models_dir = os.path.join(run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    state = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        cfg = state['cfg']
        print(f"resuming from {args.resume} (epoch {state['epoch']}); using the cfg stored in it")
    else:
        cfg = load_cfg(args.cfg, args.set, args.test_seq)
        if args.lr:
            cfg['lr'] = args.lr
    with open(os.path.join(run_dir, 'config.yml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = {'cfg_name': args.cfg, 'exp': args.exp, 'seed': args.seed, 'test_seq': args.test_seq,
            'init_released': args.init_released}
    with open(os.path.join(run_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = cfg['dataset_specs']
    n_total = ds['n_hist'] + ds['n_pred']

    print('train sequences:', ds['train'], '| test (never used here):', ds['test'])
    dataset = DatasetGTA('train', cfg)
    train_idx, val_idx = split_train_val(dataset, args.val_frac, ds['root_idx'], n_total)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader_train = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=seed_worker, generator=gen, drop_last=True)
    loader_val = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker) if val_idx else None

    model = get_path_model(cfg).to(device)
    if state is None and args.init_released:
        model.init_from_released(args.init_released)
        print(f'initialised from released weights: {args.init_released}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['model_name']} (pose_input={model.pose_input}): {n_params / 1e6:.2f} M parameters, device {device}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_step_size'], gamma=cfg['lr_gamma'])
    start_epoch, best = 0, {'metric': args.select, 'value': float('inf'), 'epoch': None}
    if state is not None:
        model.pathnet.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best = state['epoch'], state['best']
        torch.set_rng_state(state['rng']['torch'])
        np.random.set_state(state['rng']['numpy'])
        random.setstate(state['rng']['python'])
        if state['rng']['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['rng']['cuda'])

    save_freq = args.save_freq or cfg.get('save_freq', 100)
    log_path = os.path.join(run_dir, 'train_metrics.csv')
    log_fields = ['epoch', 'lr', 'train_loss', 'loss_terms', 'val_ade', 'val_fde', 'val_turn_n', 'val_turn_ade',
                  'val_turn_fde', 'val_cv_fde', 'seconds']
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(log_fields)

    epoch = start_epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses, terms_sum = [], {}
        for b, (poses_inp, poses_label) in enumerate(tqdm(loader_train, desc=f'epoch {epoch}', leave=False)):
            if args.max_train_batches and b >= args.max_train_batches:
                break
            optimizer.zero_grad()
            loss, terms = path_loss(model(poses_inp), poses_label, cfg)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            for k, v in terms.items():
                terms_sum[k] = terms_sum.get(k, 0.0) + v
        scheduler.step()
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'train_loss': np.mean(losses),
               'loss_terms': ' '.join(f'{k}={v / len(losses):.4g}' for k, v in terms_sum.items())}

        if loader_val is not None and (epoch % args.val_freq == 0 or epoch == args.epochs or STOP['requested']):
            sc = score(predict(model, loader_val, ds['root_idx']), cfg)
            s = summarize(sc['errors'], sc['labels'], ['all', 'turn'])
            s_cv = summarize(sc['errors_cv'], sc['labels'], ['all'])
            row.update({'val_ade': s['all']['ade'], 'val_fde': s['all']['fde'], 'val_turn_n': s['turn']['n'],
                        'val_turn_ade': s['turn'].get('ade'), 'val_turn_fde': s['turn'].get('fde'),
                        'val_cv_fde': s_cv['all']['fde']})
            value = row.get(args.select)
            if value is not None and value < best['value']:
                best = {'metric': args.select, 'value': float(value), 'epoch': epoch}
                model.save_ckpt(os.path.join(models_dir, 'pathnet_best.pt'))
                with open(os.path.join(models_dir, 'best.json'), 'w') as f:
                    json.dump(best, f)
        row['seconds'] = time.time() - t0

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([row.get(k, '') if not isinstance(row.get(k), float) else f'{row[k]:.6g}'
                                    for k in log_fields])
        msg = f"epoch {epoch}/{args.epochs}  loss {row['train_loss']:.4f}  ({row['loss_terms']})  {row['seconds']:.1f}s"
        if 'val_ade' in row:
            turn = f"turn({row['val_turn_n']}) FDE {row['val_turn_fde'] * 1000:.0f}" if row['val_turn_n'] else 'no turns'
            msg += (f"  | val ADE {row['val_ade'] * 1000:.0f} FDE {row['val_fde'] * 1000:.0f} mm, {turn}, "
                    f"CV FDE {row['val_cv_fde'] * 1000:.0f}  [best {best['metric']} @ {best['epoch']}]")
        print(msg, flush=True)

        if epoch % save_freq == 0 or epoch == args.epochs or STOP['requested']:
            model.save_ckpt(os.path.join(models_dir, f'pathnet_{epoch:03d}.pt'))
            save_trainstate(os.path.join(models_dir, 'trainstate_last.pt'), model, optimizer, scheduler,
                            epoch, best, cfg, meta)
        if STOP['requested']:
            print(f'stopped at epoch {epoch}; resume with --resume {models_dir}/trainstate_last.pt')
            return 124

    if loader_val is None:
        model.save_ckpt(os.path.join(models_dir, 'pathnet_best.pt'))  # no validation: "best" = last
    print(f"done: epoch {epoch}, best {best['metric']} = {best['value']:.4f} at epoch {best['epoch']}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='real_PathUNetPose', help='cfg name (cfg/<name>.yml)')
    parser.add_argument('--exp', default='exp_00', help='run name -> results/<cfg>_<exp>/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=600, help='TOTAL epochs (a resumed run continues up to this)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0, help='initial LR (0 = cfg lr)')
    parser.add_argument('--test_seq', default=None, help='fold: use this sequence as the test set, train on the rest')
    parser.add_argument('--set', nargs='*', default=[], help='cfg overrides, e.g. model_specs.n_feats_pose_path=64')
    parser.add_argument('--init_released', default=None, help='warm start from a released pathnet_N.pt')
    parser.add_argument('--resume', default=None, help='path to trainstate_last.pt')
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--select', default='val_ade', choices=['val_ade', 'val_fde', 'val_turn_fde'],
                        help='validation metric that picks pathnet_best.pt')
    parser.add_argument('--save_freq', type=int, default=0, help='0 = cfg save_freq')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_train_batches', type=int, default=0, help='debug: limit batches per epoch')
    raise SystemExit(main(parser.parse_args()))
