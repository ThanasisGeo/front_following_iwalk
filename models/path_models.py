# =====================================================================================================
# models/path_models.py: path-only models (no PoseNet). Plan: CLAUDE/pose_conditioned_path.md
# =====================================================================================================
# Every model here predicts only WHERE the person walks during the next n_pred frames. They share one
# interface, which train_path.py / eval_path.py rely on:
#   forward(poses_inp) -> {'traj':     (B, n_pred, 2), metres, actor frame,
#                          'height':   (B, n_pred), root height in the actor frame, only with predict_height,
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
#   model_specs.mlp_width (default: none = released 4096): width of the UNet's bottleneck MLP fc1 -> fc2 -> fc3,
#                 which holds ~69 M of the ~72 M parameters. model_specs.mlp_dropout adds dropout before fc2 and fc3.
#                 A smaller width (e.g. 512) is the main lever against overfitting; it rules out warm starts.
#   model_specs.output (default 'heatmap'): 'cv_residual' = the path is constant velocity (from the last
#                 cv_vel_frames past frames of the input) PLUS a learned correction from a small head on the UNet
#                 bottleneck features. The head's last layer starts at zero, so an untrained model is exactly
#                 constant velocity. The heat-maps are still produced and trained (auxiliary losses) but not used
#                 for the path. Rules out warm starts.
#   model_specs.aux_decoder (default true): false (cv_residual only) = no heat-maps at all. The bottleneck MLP and
#                 the UNet decoder are removed; what remains is the UNet encoder, the pose encoder and the
#                 correction head. forward() then returns only 'traj'.
#   model_specs.map_input (default 'unet'): 'none' (cv_residual, aux_decoder: false only) = no UNet and no maps at
#                 all, for datasets without occupancy maps (DARKO). The correction head reads only [past root xy,
#                 pose feature]; the past-path maps would only repeat the root xy.
#   model_specs.pose_repr (default 'positions'): what the pose encoder reads. 'bone_dirs' = only the direction of
#                 each bone of a reduced common skeleton (BONES), no bone lengths and no eyes / ears, so datasets
#                 that place their joints differently can be mixed. Changes only the encoder's input layer.
#   model_specs.predict_height (default false; cv_residual only): true = the head also outputs a height correction
#                 per future frame (zero-initialised) on top of "the root stays at its current height", so the
#                 model predicts the root in 3D ('height') as UPTor does. The path (x, y) is unchanged.
# =====================================================================================================
import torch
from torch import nn

from models.pathnet import UNet
from utils.gta_utils import map2traj, get_relative_pose

# Bones of a reduced skeleton shared by all datasets (COCO-17 joint indices; 'H' = hip midpoint, 'S' = shoulder
# midpoint). Eyes and ears are left out: Human3.6M and CMU have none (they are copies of the head joint there).
BONES = [('H', 11), ('H', 12), (11, 13), (13, 15), (12, 14), (14, 16),   # hips, legs
         ('H', 'S'), ('S', 0),                                            # spine, neck -> nose / head
         ('S', 5), ('S', 6), (5, 7), (7, 9), (6, 8), (8, 10)]            # shoulders, arms


def bone_directions(pose):
    # (..., J, 3) joints -> (..., len(BONES), 3) unit vectors from parent to child joint. Bone LENGTHS are removed,
    # so skeletons whose joints are placed differently (e.g. H3.6M's hips are twice as far apart as DARKO's) look
    # the same when they move the same way.
    points = {'H': (pose[..., 11, :] + pose[..., 12, :]) / 2, 'S': (pose[..., 5, :] + pose[..., 6, :]) / 2}
    joint = lambda k: points[k] if isinstance(k, str) else pose[..., k, :]
    vec = torch.stack([joint(child) - joint(parent) for parent, child in BONES], dim=-2)
    return vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class PoseHistEncoder(nn.Module):
    # Past skeletons (B, n_hist, J, 3) -> one feature vector (B, d). It describes the BODY (orientation, lean, gait);
    # where the person is comes from the root path, which the model receives separately.
    #   repr 'positions' (default): all joints relative to the root joint
    #   repr 'bone_dirs': the direction of each bone in BONES (no lengths), for mixing datasets
    def __init__(self, n_joints, root_idx, d, repr='positions'):
        super().__init__()
        if repr not in ('positions', 'bone_dirs'):
            raise ValueError(f"unknown model_specs.pose_repr: {repr!r} (positions | bone_dirs)")
        self.n_joints, self.root_idx, self.out_dim, self.repr = n_joints, root_idx, d, repr
        n_in = len(BONES) * 3 if repr == 'bone_dirs' else n_joints * 3
        self.frame_mlp = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, pose):
        bs, nf = pose.shape[:2]
        if self.repr == 'bone_dirs':
            x = bone_directions(pose)
        else:
            x = get_relative_pose(pose, self.root_idx, self.n_joints)
        return self.gru(self.frame_mlp(x.reshape(bs, nf, -1)))[1][0]  # final hidden state


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

        # Smaller bottleneck MLP. UNet.forward only calls fc1/fc2/fc3 (ReLU after fc1 and fc2), so swapping the
        # layers here is enough; models/pathnet.py stays unchanged.
        self.mlp_width = ms.get('mlp_width')
        if self.mlp_width:
            p = ms.get('mlp_dropout', 0.0)
            n_in, n_out = self.pathnet.fc1.in_features, self.pathnet.fc3.out_features
            self.pathnet.fc1 = nn.Linear(n_in, self.mlp_width)
            self.pathnet.fc2 = nn.Sequential(nn.Dropout(p), nn.Linear(self.mlp_width, self.mlp_width))
            self.pathnet.fc3 = nn.Sequential(nn.Dropout(p), nn.Linear(self.mlp_width, n_out))

        # Constant velocity + learned correction. The head is a submodule of the UNet so it is saved in pathnet_N.pt.
        # Its input is the UNet bottleneck (conv features before the MLP) plus [past root xy, pose feature].
        self.output = ms.get('output', 'heatmap')
        if self.output not in ('heatmap', 'cv_residual'):
            raise ValueError(f"unknown model_specs.output: {self.output!r} (heatmap | cv_residual)")
        # model_specs.prior (cv_residual only): 'cv' = straight line at the current velocity (default);
        # 'unicycle' = the head also outputs a turn rate and an acceleration, and the prior becomes the
        # non-holonomic rollout heading(t) = heading_cv + w t, speed(t) = speed_cv + a t. Both start at 0
        # (zero-initialised layer), so training starts exactly at constant velocity.
        self.prior = ms.get('prior', 'cv')
        if self.prior not in ('cv', 'unicycle'):
            raise ValueError(f"unknown model_specs.prior: {self.prior!r} (cv | unicycle)")
        self.n_prior_params = 2 if self.prior == 'unicycle' else 0
        self.predict_height = ms.get('predict_height', False)
        # head output: [x, y correction per frame | height correction per frame (predict_height) | prior params]
        self.n_head_out = self.n_pred * (3 if self.predict_height else 2) + self.n_prior_params
        if self.output == 'cv_residual':
            self.cv_vel_frames = ms.get('cv_vel_frames', 4)
            w = ms.get('cv_head_width', 256)
            head = nn.Sequential(nn.Linear(self.pathnet.fc1.in_features, w), nn.ReLU(),
                                 nn.Dropout(ms.get('mlp_dropout', 0.0)),
                                 nn.Linear(w, self.n_head_out))
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.pathnet.cv_head = head

        # No auxiliary branch: delete the bottleneck MLP and the decoder, so they are neither trained nor saved.
        self.aux_decoder = ms.get('aux_decoder', True)
        if not self.aux_decoder:
            if self.output != 'cv_residual':
                raise ValueError('aux_decoder: false needs output: cv_residual (otherwise the path comes from the heat-maps)')
            for name in ('fc1', 'fc2', 'fc3', 'upconv3', 'decoder3', 'upconv2', 'decoder2', 'upconv1', 'decoder1', 'conv'):
                delattr(self.pathnet, name)

        # No maps: replace the UNet by a plain container holding only the pose encoder and a correction head sized
        # for [past root xy, pose feature]. It is still called `pathnet`, so save_ckpt / load_ckpt work unchanged.
        self.map_input = ms.get('map_input', 'unet')
        if self.map_input not in ('unet', 'none'):
            raise ValueError(f"unknown model_specs.map_input: {self.map_input!r} (unet | none)")
        if self.map_input == 'none':
            if self.aux_decoder or self.output != 'cv_residual':
                raise ValueError('map_input: none needs output: cv_residual and aux_decoder: false')
            container = nn.Module()
            n_in = self.n_hist * 2
            if self.pose_input != 'none':
                container.pose_encoder = self.pathnet.pose_encoder
                n_in += container.pose_encoder.out_dim
            w = ms.get('cv_head_width', 256)
            head = nn.Sequential(nn.Linear(n_in, w), nn.ReLU(), nn.Dropout(ms.get('mlp_dropout', 0.0)),
                                 nn.Linear(w, self.n_head_out))
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            container.cv_head = head
            self.pathnet = container

    def _encode(self, maps):
        # The encoder half of UNet.forward (same input folding): (B, C, H, W, 1) maps -> conv bottleneck (B, 8f, 5, 5).
        u = self.pathnet
        bs, n_inp, h, w, chn = maps.shape
        x = maps.permute((0, 2, 3, 1, 4)).reshape((bs, h, w, -1)).permute((0, 3, 1, 2))
        enc3 = u.encoder3(u.pool2(u.encoder2(u.pool1(u.encoder1(x)))))
        return u.bottleneck(u.pool3(enc3))

    def _make_pose_encoder(self, cfg):
        ms = cfg['model_specs']
        if self.pose_input == 'learned':
            return PoseHistEncoder(self.n_joints, self.root_idx, ms.get('n_feats_pose_path', 128),
                                   ms.get('pose_repr', 'positions'))
        raise ValueError(f"unknown model_specs.pose_input: {self.pose_input!r} (none | learned)")

    def forward(self, poses_inp):
        pose = poses_inp['pose']['pose'].to(self.device).float()           # (B, n_hist, J, 3) past skeletons
        if self.map_input != 'none':
            maps = poses_inp['env']['map'].to(self.device).float()          # (B, 1, H, W, 1) occupancy
            if self.traj_map_input:
                traj_map_hist = poses_inp['pose']['traj_map'].to(self.device).float()  # (B, n_hist, H, W, 1)
                maps = torch.cat((maps, traj_map_hist), dim=1)
        bs = pose.shape[0]

        traj_hist = pose[:, :, self.root_idx, :2]
        extra = [traj_hist.reshape(bs, -1)]
        if self.pose_input != 'none':
            extra.append(self.pathnet.pose_encoder(pose))
        extra = torch.cat(extra, dim=1)
        if self.map_input == 'none':
            bottleneck = extra.new_zeros(bs, 0)                             # nothing from maps
        elif self.aux_decoder:
            traj_map_pred, bottleneck = self.pathnet(maps, extra)
        else:
            bottleneck = self._encode(maps)
        if self.output == 'cv_residual':
            # Velocity of THIS window, measured on its own past: average over the last cv_vel_frames frames (0.8 s).
            k = self.cv_vel_frames
            vel = (traj_hist[:, -1] - traj_hist[:, -1 - k]) / k
            steps = torch.arange(1, self.n_pred + 1, device=pose.device, dtype=pose.dtype)
            head_out = self.pathnet.cv_head(torch.cat((bottleneck.reshape(bs, -1), extra), dim=1))
            correction = head_out[:, :self.n_pred * 2]
            if self.prior == 'unicycle':
                # learned turn rate (rad/frame, |w| < 0.2) and acceleration (m/frame^2, |a| < 0.005), bounded
                turn = 0.2 * torch.tanh(head_out[:, -2:-1])
                acc = 0.005 * torch.tanh(head_out[:, -1:])
                speed = (vel.norm(dim=1, keepdim=True) + acc * steps[None]).clamp(min=0)   # (B, n_pred)
                heading = torch.atan2(vel[:, 1:], vel[:, :1]) + turn * steps[None]
                step_xy = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1) * speed[..., None]
                traj_cv = traj_hist[:, -1:] + step_xy.cumsum(dim=1)
            else:
                traj_cv = traj_hist[:, -1:] + vel[:, None, :] * steps[None, :, None]
            out = {'traj': traj_cv + correction.reshape(bs, self.n_pred, 2)}
            if self.predict_height:  # current root height + learned correction
                out['height'] = pose[:, -1:, self.root_idx, 2] + head_out[:, self.n_pred * 2:self.n_pred * 3]
            if self.aux_decoder:
                out['traj_map'] = traj_map_pred
            return out
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
        if self.mlp_width:
            raise ValueError('mlp_width changes the bottleneck MLP; train it from scratch (INIT="none")')
        if self.output != 'heatmap':
            raise ValueError('output: cv_residual adds a new head; train it from scratch (INIT="none")')
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
