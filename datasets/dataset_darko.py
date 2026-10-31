# =====================================================================================================
# datasets/dataset_darko.py: the DARKO dataset (UPTor paper, arXiv 2505.14866) for the path-only models
# =====================================================================================================
# Reads the raw trials (UPtor/darko/data/<trial>.npy, one row per frame: [time ms, flag, 30 joints x (x,y,z)],
# metres, z up, 15 Hz) and returns the SAME batch dict as datasets/dataset_gta.py, so train_path.py,
# eval_path.py and models/path_models.py run unchanged. Plan: UPtor/training_plan.md
#
# Per window (n_hist + n_pred consecutive frames of one trial, stride 1 frame):
#   * joints -> our 18 (COCO-17 order + root). Root (joint 17) = DARKO pelvis by default (the joint the paper
#     scores), or the hip midpoint as in Real-IM (dataset_specs.root_joint: pelvis | hip_mid)
#   * actor frame as for Real-IM: origin = root at "now" (last past frame, also in height), +x = root motion
#     over the last heading_frames frames (3 = 0.2 s), z up, yaw only
#   * past-path maps: Gaussian blobs, utils/gta_utils.traj2map on the +-map_extent grid (as stored for Real-IM)
#   * occupancy map: DARKO has none -> every cell "unknown" (-0.01)
#
# Split (by TRIAL, never by window, so no trial is in two sets): the trials are shuffled with split_seed and
# cut into n_folds folds. Fold test_fold is the test set (1/7 of the trials ~ the paper's 5K of 35K frames);
# val_frac of the remaining trials (whole trials) is the validation set; the rest is training.
#   mode: 'train' | 'val' | 'test' | 'trainval' (train + val)
# Window keys are "<trial>.<start frame>", like Real-IM's "<sequence>.<window>".
# =====================================================================================================
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.gta_utils import traj2map

# Our joint order (COCO-17) as DARKO joint indices (UPtor/darko/darko.yaml); joint 17 (root) is added below.
DARKO_TO_COCO = [27, 26, 29, 25, 28, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
DARKO_PELVIS, DARKO_LHIP, DARKO_RHIP = 0, 1, 2
UNKNOWN = -0.01


def darko_split(trials, ds):
    # Deterministic split of the trial names: {'train': [...], 'val': [...], 'test': [...]}.
    rng = np.random.default_rng(ds.get('split_seed', 0))
    order = [trials[i] for i in rng.permutation(len(trials))]
    folds = np.array_split(np.arange(len(order)), ds.get('n_folds', 7))
    test_idx = set(folds[ds.get('test_fold', 0)].tolist())
    test = [t for i, t in enumerate(order) if i in test_idx]
    rest = [t for i, t in enumerate(order) if i not in test_idx]
    n_val = int(round(ds.get('val_frac', 0.1) * len(rest)))
    return {'train': sorted(rest[n_val:]), 'val': sorted(rest[:n_val]), 'test': sorted(test)}


def split_name(ds):
    return f"darko_fold{ds.get('test_fold', 0)}of{ds.get('n_folds', 7)}_s{ds.get('split_seed', 0)}"


class DatasetDARKO(Dataset):

    def __init__(self, mode, cfg):
        ds = cfg['dataset_specs']
        self.n_hist, self.n_pred = ds['n_hist'], ds['n_pred']
        self.n_total = self.n_hist + self.n_pred
        self.root_idx = ds['root_idx']
        self.heading_frames = ds.get('heading_frames', 3)
        self.map_extent, self.map_size = ds['map_extent'], ds['map_size']
        self.use_traj_map = ds.get('use_traj_map', True)  # false: skip the past-path maps (models without maps)
        self.root_joint = ds.get('root_joint', 'pelvis')
        if self.root_joint not in ('pelvis', 'hip_mid'):
            raise ValueError(f"dataset_specs.root_joint must be 'pelvis' or 'hip_mid', got {self.root_joint!r}")

        data_dir = os.path.join(os.path.dirname(__file__), ds['data_file'])
        files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        if not files:
            raise FileNotFoundError(f'no DARKO trials (*.npy) in {data_dir}')
        trials = [os.path.splitext(os.path.basename(f))[0] for f in files]
        split = darko_split(trials, ds)
        keep = set(split['train'] + split['val']) if mode == 'trainval' else set(split[mode])

        self.pose, self.data = {}, []
        for f, name in zip(files, trials):
            if name not in keep:
                continue
            a = np.load(f).astype(np.float32)
            j = a[:, 2:].reshape(len(a), 30, 3)
            root = j[:, DARKO_PELVIS] if self.root_joint == 'pelvis' else (j[:, DARKO_LHIP] + j[:, DARKO_RHIP]) / 2
            self.pose[name] = np.concatenate((j[:, DARKO_TO_COCO], root[:, None]), axis=1)  # (T, 18, 3) world
            self.data += [(name, s) for s in range(len(a) - self.n_total + 1)]
        self.idx2scene = {name: name for name in self.pose}  # same attribute name as DatasetGTA
        print(f'DARKO {mode}: {len(self.pose)} trials, {len(self.data)} windows of {self.n_total} frames')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, s = self.data[idx]
        w = self.pose[name][s:s + self.n_total]                         # (n_total, 18, 3) world frame
        now = self.n_hist - 1
        origin = w[now, self.root_idx].copy()
        d = w[now, self.root_idx, :2] - w[now - self.heading_frames, self.root_idx, :2]
        yaw = np.arctan2(d[1], d[0])
        c, s_ = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]], dtype=np.float32)  # world -> actor
        pose_a = torch.from_numpy((w - origin) @ R.T)                   # (n_total, 18, 3) actor frame
        pose_hist, pose_label = pose_a[:self.n_hist], pose_a[self.n_hist:]

        pose_a_mat = torch.eye(4)                                       # actor -> world, as in Real-IM
        pose_a_mat[:3, :3], pose_a_mat[:3, 3] = torch.from_numpy(R.T.copy()), torch.from_numpy(origin)
        n = self.map_size
        hist = {'pose': pose_hist}
        if self.use_traj_map:
            hist['traj_map'] = traj2map(pose_hist[None, :, self.root_idx, :2], self.map_extent, 2 * self.map_extent / n,
                                        (1, self.n_hist, n, n, 1), device=torch.device('cpu'))[0].float()
        env = {'pose_a_mat': pose_a_mat, 'item_key': f'{name}.{s}', 'map': torch.full((1, n, n, 1), UNKNOWN)}
        return {'env': env, 'pose': hist}, {'env': env, 'pose': {'pose': pose_label}}
