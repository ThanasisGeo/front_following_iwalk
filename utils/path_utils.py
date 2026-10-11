# =====================================================================================================
# utils/path_utils.py: shared helpers for train_path.py / eval_path.py (path-only models)
# =====================================================================================================
#   config:   load_cfg (cfg file + "a.b=value" overrides + optional test-sequence fold)
#   running:  set_seed, predict (collect model paths + everything needed for metrics)
#   metrics:  constant_velocity (baseline B0), window_errors (per-window errors), summarize, format_table
# All arrays are numpy, positions in metres in the actor frame. Index 14 of a 15-frame future = 3 s.
# =====================================================================================================
import csv
import os
import random
import signal

import numpy as np
import torch
import yaml

from utils.turns import BASE_SUBSETS, get_thresholds, lat_bin_names, turn_features, turn_labels

# Same timings as the paper's eval.py: future frame i is at (i + 1) * 0.2 s.
HORIZONS = {'err_0.5s': [1, 2], 'err_1s': [4], 'err_1.5s': [7, 8], 'err_2s': [9], 'err_3s': [14]}


# ----------------------------------------------------------------------------- config
def _parse_override(item):
    key, sep, value = item.partition('=')
    if not sep:
        raise ValueError(f'override must look like key.sub=value, got {item!r}')
    return key.strip().split('.'), yaml.safe_load(value)


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        keys, value = _parse_override(item)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def apply_test_seq(cfg, test_seq):
    # Leave-one-sequence-out fold: `test_seq` becomes the only test sequence, all others are used for training.
    ds = cfg['dataset_specs']
    seqs = list(ds['train']) + [s for s in ds['test'] if s not in ds['train']]
    if test_seq not in seqs:
        raise ValueError(f'test_seq {test_seq!r} is not one of {seqs}')
    ds['test'] = [test_seq]
    ds['train'] = [s for s in seqs if s != test_seq]
    return cfg


def load_cfg(name_or_path, overrides=None, test_seq=None):
    path = name_or_path if name_or_path.endswith(('.yml', '.yaml')) else f'cfg/{name_or_path}.yml'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, overrides)
    if test_seq:
        apply_test_seq(cfg, test_seq)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    # Gives every DataLoader worker its own reproducible numpy/random state. Workers also ignore SIGTERM/SIGINT:
    # `timeout` and Ctrl-C signal the whole process group, and a killed worker would crash the main process
    # before it can save. Only the main process reacts (train_path.py stops after the epoch and saves).
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, loader, root_idx):
    # Runs the model over a loader. Returns numpy arrays: pred/gt (N, n_pred, 2), hist (N, n_hist, 2),
    # occ (N, H, W) occupancy map, and keys "<sequence>.<window index>".
    model.eval()
    out = {'pred': [], 'gt': [], 'hist': [], 'occ': [], 'keys': []}
    for poses_inp, poses_label in loader:
        out['pred'].append(model(poses_inp)['traj'].cpu().numpy())
        out['gt'].append(poses_label['pose']['pose'][:, :, root_idx, :2].numpy())
        out['hist'].append(poses_inp['pose']['pose'][:, :, root_idx, :2].numpy())
        out['occ'].append(poses_inp['env']['map'][:, 0, :, :, 0].numpy())
        out['keys'] += list(poses_inp['env']['item_key'])
    return {k: (v if k == 'keys' else np.concatenate(v)) for k, v in out.items()}


# ----------------------------------------------------------------------------- metrics
def constant_velocity(hist, n_pred, vel_frames=4):
    # Baseline B0: keep the average velocity of the last vel_frames frames (0.8 s).
    v = (hist[:, -1] - hist[:, -1 - vel_frames]) / vel_frames
    return hist[:, -1:] + v[:, None, :] * np.arange(1, n_pred + 1)[None, :, None]


def map_cells(points, max_dist, map_size):
    # Same convention as utils/gta_utils.traj2map: column = x, row = y, cell = floor((p + d) / resolution).
    resol = 2 * max_dist / map_size
    idx = np.floor((points + max_dist) / resol).astype(int)
    inside = ((idx >= 0) & (idx < map_size)).all(axis=-1)
    return idx[..., 0], idx[..., 1], inside


def collision_stats(points, occ, max_dist, map_size, occupied_thresh=0.5):
    # Per window: share of the points that fall in an occupied cell (occupancy >= 0.5; the maps hold
    # 1 = occupied, 0 = free, -0.01 = presumably unknown), and the share outside the map (> max_dist away).
    col, row, inside = map_cells(points, max_dist, map_size)
    n = np.arange(len(points))[:, None]
    occupied = occ[n, np.clip(row, 0, map_size - 1), np.clip(col, 0, map_size - 1)] >= occupied_thresh
    return (occupied & inside).mean(axis=1), (~inside).mean(axis=1)


def window_errors(pred, gt, hist, occ=None, max_dist=None, map_size=None, miss_thresh=1.0):
    e = np.linalg.norm(pred - gt, axis=-1)  # (N, n_pred)
    out = {'ade': e.mean(axis=1), 'fde': e[:, -1]}
    for name, idx in HORIZONS.items():
        out[name] = e[:, idx].mean(axis=1)

    true_d, pred_d = gt[:, -1] - hist[:, -1], pred[:, -1] - hist[:, -1]
    out['head_err'] = np.abs(np.degrees(np.arctan2(true_d[:, 0] * pred_d[:, 1] - true_d[:, 1] * pred_d[:, 0],
                                                   (true_d * pred_d).sum(-1))))
    # Endpoint error split along / across the TRUE direction of travel. lat_err shows whether a turn was taken
    # at all; along_err is signed (+ = predicted too far, - = too short).
    u = true_d / np.maximum(np.linalg.norm(true_d, axis=-1, keepdims=True), 1e-6)
    err = pred[:, -1] - gt[:, -1]
    out['along_err'] = (err * u).sum(-1)
    out['lat_err'] = np.abs(u[:, 0] * err[:, 1] - u[:, 1] * err[:, 0])
    out['miss'] = (out['fde'] > miss_thresh).astype(float)
    if occ is not None:
        out['collision'], out['off_map'] = collision_stats(pred, occ, max_dist, map_size)
    return out


def score(res, cfg):
    # Everything evaluation needs from one predict() result: per-window errors of the model and of the
    # constant-velocity baseline, turn features + labels, and the ground truth's own collision rate (a sanity
    # reference for how coarse the occupancy maps are).
    ds, th = cfg['dataset_specs'], get_thresholds(cfg)
    d, size = ds['max_dist_from_human'], ds['map_size']
    feats = turn_features(res['hist'], res['gt'], th)
    cv = constant_velocity(res['hist'], ds['n_pred'], th['vel_frames'])
    return {
        'features': feats,
        'labels': turn_labels(feats, th),
        'errors': window_errors(res['pred'], res['gt'], res['hist'], res['occ'], d, size),
        'errors_cv': window_errors(cv, res['gt'], res['hist'], res['occ'], d, size),
        'gt_collision': collision_stats(res['gt'], res['occ'], d, size)[0],
        'subsets': subset_names(th),
        'thresholds': th,
    }


def subset_names(th):
    return BASE_SUBSETS + lat_bin_names(th)


def summarize(errors, labels, subsets):
    # Mean of every per-window metric for every subset. Subsets with no windows get n = 0 and no metrics.
    summary = {}
    for s in subsets:
        mask = labels[s]
        summary[s] = {'n': int(mask.sum())}
        if mask.any():
            summary[s].update({m: float(v[mask].mean()) for m, v in errors.items()})
    return summary


def format_table(summary_model, summary_cv, subsets, model_name='model'):
    mm = lambda x: f'{x * 1000:6.0f}'
    head = (f"{'subset':16s} {'n':>4s} | {model_name[:6]:>6s}: {'ADE':>6s} {'FDE':>6s} {'@1s':>6s} {'@2s':>6s} "
            f"{'lat':>6s} {'head°':>6s} {'miss%':>6s} {'coll%':>6s} | {'CV':>2s}: {'ADE':>6s} {'FDE':>6s}")
    lines = [head, '-' * len(head)]
    for s in subsets:
        m, c = summary_model[s], summary_cv[s]
        if m['n'] == 0:
            lines.append(f'{s:16s} {0:4d} |')
            continue
        coll = f"{m['collision'] * 100:6.1f}" if 'collision' in m else f"{'-':>6s}"
        lines.append(f"{s:16s} {m['n']:4d} | {'':6s}  {mm(m['ade'])} {mm(m['fde'])} {mm(m['err_1s'])} "
                     f"{mm(m['err_2s'])} {mm(m['lat_err'])} {m['head_err']:6.1f} {m['miss'] * 100:6.1f} {coll} "
                     f"|     {mm(c['ade'])} {mm(c['fde'])}")
    lines.append('(distances in mm; lat = sideways part of the 3-s endpoint error; CV = constant velocity B0)')
    return '\n'.join(lines)


def write_per_window_csv(path, keys, columns):
    # columns: {name: array of length N}. One row per window.
    names = list(columns)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key'] + names)
        for i, key in enumerate(keys):
            row = []
            for n in names:
                v = columns[n][i]
                row.append(int(v) if isinstance(v, (bool, np.bool_)) else (f'{v:.6g}' if isinstance(v, (float, np.floating)) else v))
            w.writerow([key] + row)
