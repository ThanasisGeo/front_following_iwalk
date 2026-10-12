# =====================================================================================================
# models/path_models.py: path-only models (no PoseNet). Plan: CLAUDE/pose_conditioned_path.md
# =====================================================================================================
# Every model here predicts only WHERE the person walks during the next n_pred frames. They share one
# interface, which train_path.py / eval_path.py rely on:
#   forward(poses_inp) -> {'traj':     (B, n_pred, 2), metres, actor frame,
#                          'traj_map': (B, n_pred, H, W, 1), only for models with a heat-map output}
#   save_ckpt(path) / load_ckpt(path, device)
#   init_from_released(path)   UNet models: warm start from the released PathNet weights
#
#   PathUNet      B1/B2: the released PathNet (UNet, models/pathnet.py) on its own. Its checkpoints are plain
#                 UNet state_dicts, so results/real_Path_GRU_exp/models/pathnet_3600.pt loads directly.
#   PathUNetPose  M1: the same UNet, but its bottleneck MLP also receives the past skeletons, encoded by
#                 PoseHistEncoder (small per-frame MLP + GRU over root-relative skeletons), model_specs.pose_input: learned
#
#   model_specs.traj_map_input (default true): false = the UNet sees ONLY the occupancy map (1 input channel)
#                 instead of [occupancy map + 15 past-path heat-maps]. The past hip positions still reach the
#                 bottleneck as numbers (and inside the skeletons). Can't warm start from the released weights.
# =====================================================================================================
import torch
from torch import nn

from models.pathnet import UNet
from utils.gta_utils import map2traj, get_relative_pose

class PoseHistEncoder(nn.Module):
    # Past skeletons (B, n_hist, J, 3) -> one feature vector (B, d).
    # The skeletons are made root-relative, so this describes the BODY (orientation, lean, gait). Where the
    # person is comes from the root path, which the UNet already receives.
    def __init__(self, n_joints, root_idx, d):
        super().__init__()
        self.n_joints, self.root_idx, self.out_dim = n_joints, root_idx, d
        self.frame_mlp = nn.Sequential(nn.Linear(n_joints * 3, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, pose):
        bs, nf = pose.shape[:2]
        rel = get_relative_pose(pose, self.root_idx, self.n_joints).reshape(bs, nf, -1)
        return self.gru(self.frame_mlp(rel))[1][0]  # final hidden state


class UNetPose(UNet):
    # The released UNet with a wider first bottleneck layer: fc1 sees [bottleneck, past root xy, pose feature].
    # UNet.forward concatenates its second argument to the bottleneck, so forward() is inherited unchanged; the
    # caller passes [past root xy, pose feature] as that argument. The pose encoder is a submodule here, so it
    # is saved in pathnet_N.pt together with the UNet.
    def __init__(self, in_channels, out_channels, cfg, pose_encoder):
        super().__init__(in_channels, out_channels, cfg)
        f = cfg['model_specs']['n_feats_unet']
        self.n_in_released = f * 8 * 5 * 5 + self.n_hist * 2  # fc1 input width of the released UNet
        self.fc1 = nn.Linear(self.n_in_released + pose_encoder.out_dim, f * 8 * 16)
        self.pose_encoder = pose_encoder


class PathUNet(nn.Module):
    default_pose_input = 'none'

    def __init__(self, cfg):
        super().__init__()
        ds, ms = cfg['dataset_specs'], cfg['model_specs']
        self.n_hist, self.n_pred, self.n_joints = ds['n_hist'], ds['n_pred'], ds['n_joints']
        self.root_idx = ds['root_idx']
        self.max_dist_from_human, self.map_size = ds['max_dist_from_human'], ds['map_size']
        self.resol = 2 * self.max_dist_from_human / self.map_size  # metres per map cell
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pose_input = ms.get('pose_input', self.default_pose_input)
        self.traj_map_input = ms.get('traj_map_input', True)
        in_channels = 1 + self.n_hist if self.traj_map_input else 1
        if self.pose_input == 'none':
            self.pathnet = UNet(in_channels, self.n_pred, cfg)
        else:
            self.pathnet = UNetPose(in_channels, self.n_pred, cfg, self._make_pose_encoder(cfg))

    def _make_pose_encoder(self, cfg):
        ms = cfg['model_specs']
        if self.pose_input == 'learned':
            return PoseHistEncoder(self.n_joints, self.root_idx, ms.get('n_feats_pose_path', 128))
        raise ValueError(f"unknown model_specs.pose_input: {self.pose_input!r} (none | learned)")

    def forward(self, poses_inp):
        pose = poses_inp['pose']['pose'].to(self.device).float()           # (B, n_hist, J, 3) past skeletons
        maps = poses_inp['env']['map'].to(self.device).float()              # (B, 1, H, W, 1) occupancy
        if self.traj_map_input:
            traj_map_hist = poses_inp['pose']['traj_map'].to(self.device).float()  # (B, n_hist, H, W, 1)
            maps = torch.cat((maps, traj_map_hist), dim=1)
        bs = pose.shape[0]

        traj_hist = pose[:, :, self.root_idx, :2]
        extra = [traj_hist.reshape(bs, -1)]
        if self.pose_input != 'none':
            extra.append(self.pathnet.pose_encoder(pose))
        traj_map_pred, _ = self.pathnet(maps, torch.cat(extra, dim=1))
        traj_pred = map2traj(traj_map_pred, self.max_dist_from_human, self.resol, self.device)
        return {'traj': traj_pred.float(), 'traj_map': traj_map_pred}

    def save_ckpt(self, path):
        torch.save(self.pathnet.state_dict(), path)

    def load_ckpt(self, path, device=None):
        self.pathnet.load_state_dict(torch.load(path, map_location=device or self.device))

    def init_from_released(self, path):
        # Warm start from a released PathNet checkpoint (plain UNet state_dict).
        # Without pose input this is an ordinary strict load. With pose input, every released weight is copied,
        # and the extra fc1 input columns (the pose feature) start at ZERO. The model then computes exactly what
        # the released model computes and can only learn to use the pose from there.
        if not self.traj_map_input:
            raise ValueError('traj_map_input: false changes the first conv layer; train it from scratch (INIT="none")')
        sd = torch.load(path, map_location='cpu')
        if self.pose_input == 'none':
            self.pathnet.load_state_dict(sd)
            return
        w_old = sd.pop('fc1.weight')
        if w_old.shape[1] != self.pathnet.n_in_released:
            raise ValueError(f'released fc1 has {w_old.shape[1]} inputs, expected {self.pathnet.n_in_released}')
        missing, unexpected = self.pathnet.load_state_dict(sd, strict=False)
        new_keys = {'fc1.weight'} | {k for k in self.pathnet.state_dict() if k.startswith('pose_encoder.')}
        if unexpected or set(missing) - new_keys:
            raise RuntimeError(f'unexpected keys {unexpected}, missing keys {sorted(set(missing) - new_keys)}')
        with torch.no_grad():
            self.pathnet.fc1.weight.zero_()
            self.pathnet.fc1.weight[:, :w_old.shape[1]] = w_old


class PathUNetPose(PathUNet):
    # M1: PathUNet with the past skeletons as an extra input (pose_input defaults to 'learned').
    default_pose_input = 'learned'

    def __init__(self, cfg):
        super().__init__(cfg)
        if self.pose_input == 'none':
            raise ValueError("PathUNetPose needs model_specs.pose_input 'learned'")


named_path_models = {
    'PathUNet': PathUNet,
    'PathUNetPose': PathUNetPose,
}


def get_path_model(cfg):
    return named_path_models[cfg['model_name']](cfg)
