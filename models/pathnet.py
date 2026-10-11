# =====================================================================================================
# models/pathnet.py: overview
# =====================================================================================================
# Four classes live here. Only ONE of them is part of the model in the paper / the released weights:
#
#   class        | used by the released configs?           | what it is
#   -------------|------------------------------------------|---------------------------------------------
#   UNet         | YES: it's the "PathNet" inside Path_GRU  | map-to-map network: occupancy map + past-trajectory
#                |   (models/path_gru.py imports it)        | heat-maps in, future-trajectory heat-maps out
#   PoseNet      | no, only Path_MLP below uses it          | plain 4-layer MLP pose predictor. NOT the PoseNet in
#                |                                          | path_gru.py: same name, different class
#   Path_MLP     | no, registered in models/motion_pred.py  | older version of Path_GRU: same UNet path stage, but
#                |   but no cfg selects it                  | an MLP instead of GRUs for the pose stage
#   PathNet_v2   | no, registered but no cfg selects it     | early baseline: ResNet18 on the occupancy map that
#                |                                          | regresses (x, y) directly, without heat-maps or a pose net
#
# Path_MLP and PathNet_v2 look like experiments the authors kept from development (ablations / earlier
# versions). They are still selectable via `model_name:` in a cfg, but only partly work with the
# current train.py / eval.py (see the notes above each class).
#
# Shared conventions (same as path_gru.py):
#   * Everything is in the ACTOR frame: the human's current pose is at the origin, metres, z up.
#   * Maps are map_size x map_size (40 x 40) grids covering [-max_dist_from_human, +max_dist_from_human]
#     in x and y (+-4 m for Real-IM, so 0.2 m per cell). Row index = y, column index = x
#     (see utils/gta_utils.traj2map / map2traj).
#   * Map tensors are 5-D: (batch, n_maps, H, W, 1). The trailing 1 is a channel axis that is always 1.
# =====================================================================================================

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from utils.gta_utils import map2traj
import torchvision
from utils.vis_utils import vis_skeleton_pcd
from utils.gta_utils import traj2map


# -----------------------------------------------------------------------------------------------------
# UNet: the "PathNet" (stage 1 of the paper's model). Predicts WHERE the person will walk.
# -----------------------------------------------------------------------------------------------------
# Input  x:     (B, 1 + n_hist, 40, 40, 1). Channel 0 is the occupancy map, channels 1..15 are one
#               heat-map per past frame showing where the person's root joint was.
#        poses: (B, n_hist, 2), the past root (x, y) as plain numbers.
# Output:       (B, n_pred, 40, 40, 1), one heat-map per FUTURE frame, each pixel in [0, 1] (sigmoid).
#               Path_GRU turns each map into one (x, y) with a soft-argmax (utils/gta_utils.map2traj).
#
# Architecture (n_feats_unet = f = 32; spatial size in brackets):
#
#   in(16)[40] -enc1-> f[40] -pool-> enc2 2f[20] -pool-> enc3 4f[10] -pool-> bottleneck 8f[5]
#                                                                                 | flatten (8f*5*5 = 6400)
#                                                     concat past (x, y) (30) --> +
#                                                     MLP fc1 -> fc2 -> fc3 (width 8f*16 = 4096)
#                                                                                 | reshape back to 8f[5]
#   out(15)[40] <-1x1 conv- dec1 f[40] <-up- dec2 2f[20] <-up- dec3 4f[10] <-up-+
#          ^ skip from enc1        ^ skip from enc2   ^ skip from enc3
#
# * It's a standard 3-level U-Net (encoder, decoder, skip connections) with one twist: the bottleneck goes
#   through a fully-connected MLP that also sees the raw past trajectory numbers. The MLP mixes global
#   information (where the person is heading) into the whole 5x5 feature grid before decoding.
# * CONSTRAINT: the hard-coded "5 * 5" in fc1/fc3 means the input must be 40x40 (40 / 2 / 2 / 2 = 5).
#   Changing map_size needs these lines changed (and the data regenerated).
# * cfg['model_specs']['n_feats_unet_mlp'] is stored below but never used. The MLP width is 8f*16.
# * The commented-out encoder4/decoder4 are a removed 4th level (it wouldn't divide 40 evenly anyway).
# -----------------------------------------------------------------------------------------------------
class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super(UNet, self).__init__()
        features = cfg['model_specs']['n_feats_unet']
        self.init_features, self.in_channels, self.out_channels = features, in_channels, out_channels
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = PoseForecastFPSDF._block(features * 4, features * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck")

        # self.upconv4 = nn.ConvTranspose2d(
        #     features * 16, features * 8, kernel_size=2, stride=2
        # )
        # self.decoder4 = PoseForecastFPSDF._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2, padding=0
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_unet_mlp']
        self.fc1 = nn.Linear(features * 8 * 5 * 5 + self.n_hist * 2, features * 8 * 16)
        self.fc2 = nn.Linear(features * 8 * 16, features * 8 * 16)
        self.fc3 = nn.Linear(features * 8 * 16, features * 8 * 5 * 5)

    def forward(self, x, poses):
        # Input x shape: batch_size, n_frames, 18, 17, 4
        # (The line above is outdated. Here x is (B, 1 + n_hist, 40, 40, 1). First the 5-D map stack is
        #  folded into an ordinary image batch (B, 16, 40, 40): each map becomes one image channel.)
        bs, n_inp, h, w, chn = x.shape
        x = x.permute((0, 2, 3, 1, 4)).reshape((bs, h, w, -1))  # bs, h, w, chn * n_inp
        x = x.permute((0, 3, 1, 2))  # bs, chn * n_inp, h, w

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))
        feat_neck = self.pool3(enc3)

        bottleneck = self.bottleneck(feat_neck)
        bottleneck_flatten = bottleneck.reshape((bs, -1))
        poses_flatten = poses.reshape((bs, -1))
        feat = torch.cat((bottleneck_flatten, poses_flatten), dim=1)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        feat = self.fc3(feat).reshape((bs, -1, 5, 5))

        # dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(feat)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out = self.conv(dec1)
        # Sigmoid = an independent "is the person here at frame t?" probability per pixel (not a softmax over
        # the map). That matches the BCE trajectory-map loss in train.py.
        #
        # !! IMPORTANT for anyone changing this class: the last reshape(...).permute(...) below does NOT
        # turn (B, 15, H, W) back into (B, 15, H, W, 1) element-for-element. reshape reinterprets memory,
        # so output frame k at pixel (r, c) actually comes from a different conv channel/pixel. (Checked: a
        # value at frame 3, row 10, col 20 ends up at frame 0, row 8, col 28.) The correct inverse of the
        # input folding would be out.permute(0, 2, 3, 1).unsqueeze(3).permute(0, 4, 1, 2, 3).
        # Training still works because the scrambling is a FIXED permutation and the conv layers learn to
        # compensate, but it removes the "pixel matches pixel" structure a U-Net is good at.
        # The released weights were trained WITH this behaviour, so "fixing" it makes them useless. Only
        # change it for models you train from scratch, and compare both versions.
        sig = nn.Sigmoid()
        x = sig(out.reshape((bs, self.out_channels, -1)))
        x = x.reshape((bs, self.out_channels, h, w))
        x = x.reshape((bs, h, w, chn, -1)).permute((0, 4, 1, 2, 3))
        # `bottleneck` is returned but all callers ignore it (it's the conv output from BEFORE the MLP).
        return x, bottleneck

    # One U-Net "level": two (3x3 conv -> BatchNorm -> ReLU) layers that keep the spatial size.
    # The `name` prefixes become the parameter names in the checkpoints (e.g. "encoder1.enc1conv1.weight"),
    # so renaming them breaks loading of the released weights.
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


# -----------------------------------------------------------------------------------------------------
# PoseNet (MLP version). NOT USED by the released model. Only Path_MLP below uses it.
# -----------------------------------------------------------------------------------------------------
# A plain 4-layer MLP: flatten every input pose number into one vector, then output all future poses at
# once. It has no notion of time. Path_GRU replaced it with the GRU-based PoseNet in path_gru.py
# (same class name, different module).
# -----------------------------------------------------------------------------------------------------
class PoseNet(torch.nn.Module):
    def __init__(self, in_feats, n_feats, out_feats):
        super(PoseNet, self).__init__()
        self.fc1 = nn.Linear(in_feats, n_feats)
        self.fc2 = nn.Linear(n_feats, n_feats * 2)
        self.fc3 = nn.Linear(n_feats * 2, n_feats)
        self.fc4 = nn.Linear(n_feats, out_feats)

    def forward(self, pose_raw):
        bs = pose_raw.shape[0]
        pose_f = pose_raw.reshape(bs, -1)
        x = F.relu(self.fc1(pose_f))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# -----------------------------------------------------------------------------------------------------
# Path_MLP: older two-stage model. NOT USED by the released configs (selectable as model_name: Path_MLP).
# -----------------------------------------------------------------------------------------------------
# Same two-stage idea as Path_GRU (read path_gru.py for the full explanation of the stages and of
# train vs test mode). The difference is stage 2:
#   stage 1 (path): the same UNet as Path_GRU -> future (x, y) per frame.
#   stage 2 (pose): the MLP PoseNet above. Its input is the past root-relative poses followed by the LAST
#                   pose copied n_pred times (a "the person keeps this pose" guess), and it outputs all
#                   future relative poses at once. The future root position comes from stage 1.
#
# Compatibility with the current scripts:
#   * train.py: should work. The forward() output dict has the same keys as Path_GRU.
#   * eval.py:  breaks. eval.py calls load_model(pathnet, posenet, device) with 3 arguments, but the
#               load_model below only takes 2.
#   * The released checkpoints don't fit it (the posenet layers have different shapes).
# -----------------------------------------------------------------------------------------------------
class Path_MLP(torch.nn.Module):

    def __init__(self, cfg):
        super(Path_MLP, self).__init__()
        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_mlp']
        self.test_mode = not (cfg['mode'] == 'train' and cfg['dataset_specs']['use_pose_gt'])
        self.train_pathnet = cfg['model_specs']['train_pathnet']
        self.train_posenet = cfg['model_specs']['train_posenet']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_feats_inp = (self.n_hist + self.n_pred) * self.n_joints * 3
        self.pathnet = UNet(1 + self.n_hist, self.n_pred, cfg)
        self.posenet = PoseNet(n_feats_inp, self.n_feats_mlp, self.n_pred * self.n_joints * 3)
        self.pathnet.train() if self.train_pathnet else self.pathnet.eval()
        self.posenet.train() if self.train_posenet else self.posenet.eval()

        self.max_dist_from_human = cfg['dataset_specs']['max_dist_from_human']
        self.map_size = cfg['dataset_specs']['map_size']
        self.resol = 2 * self.max_dist_from_human / self.map_size
        self.root_idx = cfg['dataset_specs']['root_idx']

    def pred_pathnet(self, pose, occ_map, traj_map_hist):
        # Predict trajectory map first and then predict pose.
        traj_hist = pose[:, :, self.root_idx, :2]
        map_hist = torch.cat((occ_map, traj_map_hist), dim=1)
        traj_map_pred, bottleneck = self.pathnet(map_hist, traj_hist)
        traj_pred = map2traj(traj_map_pred, self.max_dist_from_human, self.resol, self.device)
        return traj_pred, traj_map_pred

    def pred_posenet(self, pose, traj):
        bs = pose.shape[0]

        traj_pred_3d_nj, pose_last_nf = self.prepare_pose_raw(pose, traj)
        pose_pred_raw = traj_pred_3d_nj + pose_last_nf

        pose_rel = self.get_rel_pose(pose)  # bs * nf * nj * 3

        pose_raw = torch.cat((pose_rel, pose_last_nf), dim=1)  # bs * (nf + np) * nj * 3
        # pose_raw = torch.cat((pose, pose_pred_raw), dim=1)  # bs * (nf + np) * nj * 3

        # Go through MLP for pose prediction.
        traj_pred_outp = self.posenet(pose_raw)  # bs * np * nj * 3
        traj_pred_outp = traj_pred_outp.reshape(bs, self.n_pred, self.n_joints, 3)
        traj_pred_rel = self.get_rel_pose(traj_pred_outp)  # Normalize the predicted pose.
        # traj_pred_rel = traj_pred_outp

        # Add trajectory prediction to the relative pose.
        pose_pred = traj_pred_rel + traj_pred_3d_nj  # bs * np * nj * 3
        return pose_pred

    def prepare_pose_raw(self, pose, traj):
        # Predicted 3d trajectories in shape bs * np * 3.
        pos_root_last = pose[:, -1, self.root_idx, :]  # bs * 3
        pos_root_last_np = pos_root_last.unsqueeze(1).repeat((1, self.n_pred, 1))  # bs * np * 3
        traj_pred_3d = pos_root_last_np
        traj_pred_3d[:, :, :2] = traj
        traj_pred_3d_nj = traj_pred_3d.unsqueeze(2).repeat((1, 1, self.n_joints, 1))  # bs * np * nj * 3

        # Extract relative pose by subtracting the root joint.
        pose_rel = self.get_rel_pose(pose)  # bs * nf * nj * 3

        # Extend the last pose to the future and use as the raw pose.
        pose_last = pose_rel[:, -1, :, :]  # bs * nj * 3
        pose_last_nf = pose_last.unsqueeze(1).repeat((1, self.n_pred, 1, 1))  # bs * np * nj * 3
        return traj_pred_3d_nj, pose_last_nf

    def forward(self, poses_inp):
        # pred_info = {'pose': pose_pred, 'traj_map': traj_map_pred, 'traj': traj_raw}
        pred_info = {}
        pose = poses_inp['pose']['pose'].to(self.device)
        occ_map = poses_inp['env']['map'].to(self.device).float()
        traj_map_hist = poses_inp['pose']['traj_map'].to(self.device).float()

        if self.test_mode:
            traj_pred, traj_map_pred = self.pred_pathnet(pose, occ_map, traj_map_hist)
            pose_pred = self.pred_posenet(pose, traj_pred)
            pred_info = {'pose': pose_pred, 'traj_map': traj_map_pred, 'traj': traj_pred}
        else:
            traj_gt = poses_inp['pose']['pose_gt'][:, :, self.root_idx, :2].to(self.device)
            if self.train_pathnet:
                traj_pred, traj_map_pred = self.pred_pathnet(pose, occ_map, traj_map_hist)
                pred_info['traj'] = traj_pred.float()
                pred_info['traj_map'] = traj_map_pred.float()
            else:
                pred_info['traj'] = traj_gt.float()
                resol = 2 * self.max_dist_from_human / self.map_size
                map_shape = (traj_gt.shape[0], self.n_pred, self.map_size, self.map_size, 1)
                pred_info['traj_map'] = traj2map(traj_gt.float(), self.max_dist_from_human, resol, map_shape,
                                                 device=None, binary=True).float()
                # pred_info['traj_map'] = traj_map_pred.float()

            if self.train_posenet:
                pose_pred = self.pred_posenet(pose, traj_gt)
                pred_info['pose'] = pose_pred.float()
            else:
                traj_pred_3d_nj, pose_last_nf = self.prepare_pose_raw(pose, traj_gt)
                pose_pred_raw = traj_pred_3d_nj + pose_last_nf
                pred_info['pose'] = pose_pred_raw.float()

        pred = {'env': poses_inp['env'], 'pose': pred_info}
        return pred

    def load_model(self, pathnet_path, posenet_path):
        self.pathnet.load_state_dict(torch.load(pathnet_path))
        self.posenet.load_state_dict(torch.load(posenet_path))

    def save_model(self, pathnet_path, posenet_path):
        torch.save(self.pathnet.state_dict(), pathnet_path)
        torch.save(self.posenet.state_dict(), posenet_path)

    def get_rel_pose(self, pose):
        pos_root = pose[:, :, self.root_idx, :]  # bs * nf * 3
        pos_root_nj = pos_root.unsqueeze(2).repeat((1, 1, self.n_joints, 1))  # bs * nf * nj * 3
        pose_rel = pose - pos_root_nj  # Relative poses. bs * nf * nj * 3
        return pose_rel


# -----------------------------------------------------------------------------------------------------
# PathNet_v2: early trajectory-only baseline. NOT USED (selectable as model_name: PathNet_v2).
# -----------------------------------------------------------------------------------------------------
# Instead of drawing the future path as heat-maps, this version:
#   1. runs the occupancy map (copied to 3 channels, like an RGB image) through an untrained ResNet18
#      and gets one feature vector for the whole map,
#   2. concatenates it with the past root (x, y) and regresses future (x, y) numbers directly with an MLP,
#   3. builds "future poses" by moving the LAST observed pose along that path. The body pose itself is
#      never predicted (no learned pose network).
# It has no spatial output, so it can't use the trajectory-map or collision losses. Path_MLP / Path_GRU
# replaced it with the UNet + heat-map approach above.
#
# Compatibility with the current scripts: NOT runnable as-is.
#   * forward() returns no 'traj_map', which the cfg losses (use_traj_map / use_collision) need.
#   * It has no load_model / save_model, which train.py and eval.py call.
#   * fc3 outputs n_hist * 2 numbers that are reshaped as n_pred frames, so it only works while
#     n_hist == n_pred.
# -----------------------------------------------------------------------------------------------------
class PathNet_v2(nn.Module):

    def __init__(self, cfg):
        super(PathNet_v2, self).__init__()
        self.cfg = cfg
        self.n_hist = cfg['dataset_specs']['n_hist']
        self.n_pred = cfg['dataset_specs']['n_pred']
        self.n_joints = cfg['dataset_specs']['n_joints']
        self.n_feats_mlp = cfg['model_specs']['n_feats_mlp']
        self.fc1 = torch.nn.Linear(self.n_feats_mlp + self.n_hist * 2, 2 * self.n_feats_mlp)  # 5*5 from image dimension
        self.fc2 = nn.Linear(self.n_feats_mlp * 2, self.n_feats_mlp)
        self.fc3 = torch.nn.Linear(self.n_feats_mlp, self.n_hist * 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feat_extractor = torchvision.models.resnet18(pretrained=False)
        num_features = self.feat_extractor.fc.in_features
        self.feat_extractor.fc = nn.Linear(num_features, self.n_feats_mlp)

        self.root_idx = cfg['dataset_specs']['root_idx']

    def forward(self, poses_inp):
        pose = poses_inp['pose']['pose'].to(self.device)
        occ_map = poses_inp['env']['map'].to(self.device).float()
        bs = pose.shape[0]

        # Predict trajectory map first and then predict pose.
        occ_map_inp = occ_map[:, :, :, :, 0].repeat(1, 3, 1, 1)
        occ_map_feat = self.feat_extractor(occ_map_inp)
        traj_hist = pose[:, :, self.root_idx, :2]
        hist_feat = torch.cat((traj_hist.reshape(bs, -1), occ_map_feat), dim=1)
        x = F.relu(self.fc1(hist_feat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        traj_pred = x.reshape(bs, self.n_pred, 2)

        # # Predict poses.
        # pose_f = pose.reshape(bs, -1)
        pose_last = pose[:, -1, :, :]
        pos_root = pose_last[:, self.root_idx, :].unsqueeze(1).repeat((1, self.n_joints, 1))
        pos_root_nf = pos_root.unsqueeze(1).repeat((1, self.n_pred, 1, 1))
        pose_rel = pose_last - pos_root
        pose_rel_nf = pose_rel.unsqueeze(1).repeat((1, self.n_pred, 1, 1))

        traj_pred_3d = pos_root_nf
        traj_pred_3d[:, :, :, :2] = traj_pred.unsqueeze(2).repeat((1, 1, self.n_joints, 1))
        pose_pred_raw = pose_rel_nf + traj_pred_3d
        # x_cat = torch.cat((pose_f, traj.reshape(bs, -1)), dim=1)
        # x = F.relu(self.fc1(x_cat))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = x.reshape(bs, -1, self.n_joints, 3)

        # pose_pred = torch.rand(pose.shape).to(self.device)
        # pose_pred[:, :, self.root_idx, :2] = traj_pred
        pred_info = {'pose': pose_pred_raw, 'traj': traj_pred}
        pred = {'env': poses_inp['env'], 'pose': pred_info}
        return pred
