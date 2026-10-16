# =====================================================================================================
# tools/export_predictions.py: run the authors' PathNet and our model on every window of a sequence and save
# the predicted paths (plus ground truth, constant velocity, turn labels and errors) to one .npz file.
# =====================================================================================================
# The .npz is read by u_minnesote_dataset_analysis/compare_models_visual.py, which only needs numpy, open3d
# and matplotlib (no torch). Run from the repo root in the poseforecast env:
#
#   python tools/export_predictions.py --seq me_f2_seq1_raw \
#          --out ../../u_minnesote_dataset_analysis/predictions
#
# Everything is in the actor frame of each window (person at the origin at the current frame), metres.
# Window index `fidx` = index into the .npz arrays joints_a / pose_a_mat_list of that sequence.
# Note: me_f2_seq1_raw is the test sequence. The other four were used for TRAINING by both models.
# =====================================================================================================
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dataset_gta import DatasetGTA  # noqa: E402
from models.path_models import get_path_model  # noqa: E402
from utils.path_utils import apply_test_seq, constant_velocity, load_cfg, predict, score  # noqa: E402

MODELS = {
    # name: (cfg, checkpoint)
    'authors': ('real_PathUNet', 'results/real_Path_GRU_exp/models/pathnet_3600.pt'),
    'ours': ('real_PathUNetPoseSmallCV', 'results/real_PathUNetPoseSmallCV_m1smallcv_s0/models/pathnet_400.pt'),
}


def main(args):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out, keys = {}, None
    for name, (cfg_name, ckpt) in MODELS.items():
        cfg = apply_test_seq(load_cfg(cfg_name), args.seq)  # this sequence becomes the "test" set to load it
        model = get_path_model(cfg).to(device)
        model.load_ckpt(ckpt, device)
        loader = DataLoader(DatasetGTA('test', cfg), batch_size=64, shuffle=False, num_workers=args.num_workers)
        res = predict(model, loader, cfg['dataset_specs']['root_idx'])
        if keys is None:
            keys = res['keys']
            sc = score(res, cfg)
            out.update(gt=res['gt'], hist=res['hist'], occ=res['occ'])
            out.update({f'feat_{k}': v for k, v in sc['features'].items()})
            out.update({f'label_{k}': v for k, v in sc['labels'].items()})
            out['pred_cv'] = constant_velocity(res['hist'], cfg['dataset_specs']['n_pred'], sc['thresholds']['vel_frames'])
        assert res['keys'] == keys, 'window order differs between models'
        out[f'pred_{name}'] = res['pred']
        print(f'{name}: {cfg_name} {ckpt}')
    for name in ('authors', 'ours', 'cv'):
        e = np.linalg.norm(out[f'pred_{name}'] - out['gt'], axis=-1)
        out[f'ade_{name}'], out[f'fde_{name}'] = e.mean(1), e[:, -1]
    out['fidx'] = np.array([int(k.rsplit('.', 1)[1]) for k in keys])
    out['seq'] = np.array(args.seq)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'{args.seq}.npz')
    np.savez_compressed(path, **out)
    n, t = len(keys), out['label_turn'].sum()
    print(f"saved {path}: {n} windows ({t} turns) | mean FDE authors {out['fde_authors'].mean()*1000:.0f} "
          f"ours {out['fde_ours'].mean()*1000:.0f} CV {out['fde_cv'].mean()*1000:.0f} mm")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='me_f2_seq1_raw')
    p.add_argument('--out', default='../../u_minnesote_dataset_analysis/predictions')
    p.add_argument('--num_workers', type=int, default=2)
    main(p.parse_args())
