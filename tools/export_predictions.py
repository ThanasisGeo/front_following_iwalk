# =====================================================================================================
# tools/export_predictions.py: run any set of models on any sequence(s) and save their predicted paths
# (plus ground truth, constant velocity, turn labels and errors) to predictions/<seq>.npz for the viewer.
# =====================================================================================================
# The .npz is read by u_minnesote_dataset_analysis/compare_models_visual.py (numpy + open3d + matplotlib only).
# Models are listed by short name in tools/model_registry.yml. Run from the repo root in the poseforecast env:
#
#   python tools/export_predictions.py                                 # default models, all 5 sequences
#   python tools/export_predictions.py --seq me_f2_seq1_raw kl_f4_seq2_raw
#   python tools/export_predictions.py --models authors mirror_s1     # pick models from the registry
#   python tools/export_predictions.py --add mine=real_PathUNetPoseSmallCV:path/to/pathnet_400.pt   # ad hoc model
#
# Everything is in the actor frame of each window (person at the origin at "now"), metres.
# Window index `fidx` = index into joints_a / pose_a_mat_list of that sequence's dataset .npz.
# For every model and sequence the file stores seen_<model>: True if the model was TRAINED on that sequence,
# so its predictions there are not test results.
# =====================================================================================================
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datasets.dataset_gta import DatasetGTA  # noqa: E402
from models.path_models import get_path_model  # noqa: E402
from utils.path_utils import apply_test_seq, constant_velocity, load_cfg, predict, score  # noqa: E402

MAP_EXTENT = 2.5  # the stored occupancy / path maps cover +-2.5 m (0.125 m cells), whatever the cfg says


def load_registry():
    with open(os.path.join(ROOT, 'tools', 'model_registry.yml')) as f:
        return yaml.safe_load(f)


def model_cfg(spec):
    # The model's own cfg: config.yml of its run folder if present, else the named cfg file.
    run_cfg = os.path.join(spec['run'], 'config.yml') if spec.get('run') else None
    if run_cfg and os.path.exists(run_cfg):
        with open(run_cfg) as f:
            return yaml.safe_load(f)
    if not spec.get('cfg'):
        raise SystemExit(f"model {spec}: needs a run folder with config.yml or a `cfg:` entry")
    return load_cfg(spec['cfg'])


def all_sequences():
    ds = load_cfg('real_PathUNet')['dataset_specs']
    return list(ds['train']) + [s for s in ds['test'] if s not in ds['train']]


def main(args):
    os.chdir(ROOT)
    reg = load_registry()
    for item in args.add:
        name, _, rest = item.partition('=')
        cfg_name, _, ckpt = rest.partition(':')
        reg[name] = {'cfg': cfg_name, 'ckpt': ckpt, 'label': name}
        args.models = (args.models or []) + [name]
    names = args.models or [k for k, v in reg.items() if v.get('default')]
    missing = [n for n in names if n not in reg]
    if missing:
        raise SystemExit(f'unknown model(s) {missing}; registry has {list(reg)}')
    seqs = all_sequences() if args.seq == ['all'] else args.seq
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seq in seqs:
        out, keys = {}, None
        for name in names:
            spec = reg[name]
            cfg = model_cfg(spec)
            seen = seq in cfg['dataset_specs']['train']
            model = get_path_model(cfg).to(device)
            model.load_ckpt(spec['ckpt'], device)
            cfg_seq = apply_test_seq(load_cfg('real_PathUNet'), seq)   # only to load this sequence as "test"
            cfg_seq['model_specs'], cfg_seq['model_name'] = cfg['model_specs'], cfg['model_name']
            loader = DataLoader(DatasetGTA('test', cfg_seq), batch_size=64, shuffle=False, num_workers=args.num_workers)
            res = predict(model, loader, cfg['dataset_specs']['root_idx'])
            if keys is None:
                keys = res['keys']
                sc = score(res, cfg_seq)
                out.update(gt=res['gt'], hist=res['hist'], occ=res['occ'])
                out.update({f'feat_{k}': v for k, v in sc['features'].items()})
                out.update({f'label_{k}': v for k, v in sc['labels'].items()})
                out['pred_cv'] = constant_velocity(res['hist'], cfg['dataset_specs']['n_pred'], sc['thresholds']['vel_frames'])
            assert res['keys'] == keys, 'window order differs between models'
            out[f'pred_{name}'], out[f'seen_{name}'] = res['pred'], np.array(seen)
            out[f'label_name_{name}'] = np.array(spec.get('label', name))
            print(f"{seq}: {name:14s} {spec['ckpt']}{'   (TRAINED on this sequence)' if seen else ''}")
        for name in names + ['cv']:
            e = np.linalg.norm(out[f'pred_{name}'] - out['gt'], axis=-1)
            out[f'ade_{name}'], out[f'fde_{name}'] = e.mean(1), e[:, -1]
        out['fidx'] = np.array([int(k.rsplit('.', 1)[1]) for k in keys])
        out['seq'], out['models'], out['map_extent'] = np.array(seq), np.array(names), np.array(MAP_EXTENT)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f'{seq}.npz')
        np.savez_compressed(path, **out)
        t = out['label_turn'].astype(bool)
        summary = '  '.join(f"{n} {out[f'fde_{n}'].mean()*1000:.0f}" for n in names + ['cv'])
        print(f"saved {path}: {len(keys)} windows ({t.sum()} turns) | mean FDE (mm): {summary}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seq', nargs='+', default=['all'], help="sequence name(s), or 'all'")
    p.add_argument('--models', nargs='+', default=None, help='model names from tools/model_registry.yml (default: default ones)')
    p.add_argument('--add', nargs='*', default=[], help='ad hoc model: name=cfg:checkpoint_path')
    p.add_argument('--out', default=os.path.join(ROOT, '..', 'u_minnesote_dataset_analysis', 'predictions'))
    p.add_argument('--num_workers', type=int, default=2)
    main(p.parse_args())
