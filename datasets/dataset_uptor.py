# =====================================================================================================
# datasets/dataset_uptor.py: Human3.6M and CMU-Mocap as in the UPTor paper (arXiv 2505.14866), path-only models
# =====================================================================================================
# Same batch dict, actor frame and maps as datasets/dataset_darko.py (this class only replaces how the trials are
# read, resampled and split). Plan: UPTor/docs/training_plan.md, section 10. The protocol follows the authors'
# loaders (UPTor/human_36m.py, UPTor/cmu_mocap.py):
#
#   H36M  data.npz -> positions_3d[subject][action] (T, 32, 3), metres, z up, 50 Hz. Resampled to hz (10) by taking
#         every 5th frame. Split by subject: train S1 S5 S6 S7 S8, test S9 S11 (exactly the paper's).
#         Test windows = STPOTR's protocol: offset 0, consecutive NON-overlapping windows (test_stride: n_hist+n_pred).
#   CMU   <subject>_<trial>.npy rows [time ms, 0, 31 joints x (x, y, z)], CMU units, y up, 30 Hz. Scaled to metres
#         (x 0.0564), axes (x, y, z) -> (x, -z, y), every 3rd frame. The paper's test trials are not published:
#         random by-trial folds as for DARKO (n_folds 14 ~ the paper's 7% of frames).
#         Test windows = the authors' loader: every start frame at 30 Hz (all 3 offsets, stride 1).
#
# Training uses every offset of the resampling as its own trial (5x / 3x the windows); validation = whole held-out
# training trials (val_frac), offset 0. Keys: "<trial>@<offset>.<start frame>".
#   mode: 'train' | 'val' | 'test' | 'trainval' (train + val)
# =====================================================================================================
import glob
import os

import numpy as np

from datasets.dataset_darko import DatasetDARKO, darko_split

# Our joint order (COCO-17: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles; left before right) as
# indices of each dataset's joints. Neither has eyes or ears: the nose / head point is repeated (left = right
# copies, so mirroring stays consistent). Joint 17 (root) = pelvis / CMU root.
H36M_TO_COCO = [14, 14, 14, 15, 15, 17, 25, 18, 26, 19, 27, 6, 1, 7, 2, 8, 3]
CMU_TO_COCO = [16, 16, 16, 16, 16, 17, 24, 18, 25, 19, 26, 1, 6, 2, 7, 3, 8]  # CMU positions are bone END points
CMU_TO_M = 2.54 / 100 / 0.45
NATIVE_HZ = {'H36M': 50, 'CMU': 30}
H36M_TRAIN, H36M_TEST = ['S1', 'S5', 'S6', 'S7', 'S8'], ['S9', 'S11']


def read_h36m(path):
    raw = np.load(os.path.join(path, 'data.npz'), allow_pickle=True)['positions_3d'].item()
    out = {}
    for subject, actions in raw.items():
        for action, a in actions.items():
            a = a.astype(np.float32)
            out[f'{subject}/{action}'] = np.concatenate((a[:, H36M_TO_COCO], a[:, :1]), axis=1)
    return out


def read_cmu(path):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, '*.npy'))):
        a = np.load(f).astype(np.float32)
        j = a[:, 2:].reshape(len(a), 31, 3) * CMU_TO_M
        j = np.stack((j[..., 0], -j[..., 2], j[..., 1]), axis=-1)   # y up -> z up, still right-handed
        out[os.path.splitext(os.path.basename(f))[0]] = np.concatenate((j[:, CMU_TO_COCO], j[:, :1]), axis=1)
    return out


def uptor_split(dataset, trials, ds):
    # {'train': [...], 'val': [...], 'test': [...]} of trial names.
    if dataset == 'CMU':
        return darko_split(trials, ds)
    test = sorted(t for t in trials if t.split('/')[0] in H36M_TEST)
    rest = [t for t in sorted(trials) if t.split('/')[0] in H36M_TRAIN]
    rest = [rest[i] for i in np.random.default_rng(ds.get('split_seed', 0)).permutation(len(rest))]
    n_val = int(round(ds.get('val_frac', 0.1) * len(rest)))
    return {'train': sorted(rest[n_val:]), 'val': sorted(rest[:n_val]), 'test': test}


def uptor_split_name(dataset, ds):
    if dataset == 'H36M':
        return 'h36m_S9+S11'
    return f"cmu_fold{ds.get('test_fold', 0)}of{ds.get('n_folds', 14)}_s{ds.get('split_seed', 0)}"


class DatasetUPTor(DatasetDARKO):

    def __init__(self, mode, cfg):
        # Everything except reading/splitting is DatasetDARKO's (actor frame, maps, __getitem__).
        ds, name = cfg['dataset_specs'], cfg['dataset']
        self._setup(ds)
        data_dir = os.path.join(os.path.dirname(__file__), ds['data_file'])
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'no {name} data in {data_dir}')
        full = read_h36m(data_dir) if name == 'H36M' else read_cmu(data_dir)
        split = uptor_split(name, list(full), ds)
        keep = split['train'] + split['val'] if mode == 'trainval' else split[mode]

        step = NATIVE_HZ[name] // ds['hz']
        if mode in ('train', 'trainval'):
            offsets, stride = range(step), 1
        elif mode == 'val':
            offsets, stride = [0], 1
        else:
            offsets = [0] if ds.get('test_offsets', 'all') == 'first' else range(step)
            stride = ds.get('test_stride', 1)
        self.pose, self.data = {}, []
        for trial in keep:
            for o in offsets:
                p = full[trial][o::step]
                key = f'{trial}@{o}'
                self.pose[key] = p
                self.data += [(key, s) for s in range(0, len(p) - self.n_total + 1, stride)]
        self.idx2scene = {k: k for k in self.pose}
        print(f'{name} {mode}: {len(keep)} trials, {len(self.data)} windows of {self.n_total} frames at {ds["hz"]} Hz')
