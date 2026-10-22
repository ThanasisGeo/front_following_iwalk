# =====================================================================================================
# utils/augment.py: training-time data augmentation for path models (train_path.py)
# =====================================================================================================
# Mirror (left <-> right). In the actor frame (x = walking direction, y = left, z = up), a mirrored walk is
# an equally plausible walk: a left turn becomes a right turn, a wall on the left moves to the right.
# Mirroring one window means:
#   * skeletons (past and future): y -> -y, and swap every left/right joint pair (a mirrored left shoulder is
#     the person's right shoulder)
#   * the 40x40 maps: flip the rows (rows index y, cells of 0.125 m covering +-2.5 m). The two kinds of map use
#     different cell conventions (measured on the data), so their mirror axes differ by half a cell:
#       - occupancy and future-path maps fill the cell containing the point (floor): y = 0 is the boundary
#         between rows 19 and 20, so row r <-> 39 - r is exact.
#       - past-path maps are Gaussian blobs centred at the continuous index (y + 2.5) / 0.125 (peak = round):
#         y = 0 is row 20, so row r <-> 40 - r, i.e. flip and shift by one row (row 0 has no partner: zero).
# Enabled with cfg `augment: {mirror: p}` (or --set augment.mirror=0.5): each training window is mirrored with
# probability p. Validation and test data are never augmented.
# =====================================================================================================
import torch

# COCO-17 + hip centre (Real-IM): (left, right) joint pairs. Nose (0) and hip centre (17) map to themselves.
COCO_LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _mirror_joints(pose, flip, perm):
    # pose (B, T, J, 3); flip (B,) bool
    m = pose[flip][:, :, perm].clone()
    m[..., 1] = -m[..., 1]
    pose[flip] = m


def _flip_rows(maps, flip, centred=False):
    # maps (B, N, H, W, 1): rows are dim -3. centred=True for blob maps whose cell i sits at y = i*res - d.
    m = torch.flip(maps[flip], dims=[-3])
    if centred:
        m = torch.roll(m, shifts=1, dims=-3)
        m[..., 0, :, :] = 0
    maps[flip] = m


def mirror_batch(poses_inp, poses_label, p, n_joints=18, pairs=COCO_LR_PAIRS):
    """Mirror each window of a collated batch with probability p (in place, returns the same dicts)."""
    bs = poses_inp['pose']['pose'].shape[0]
    flip = torch.rand(bs) < p
    if not flip.any():
        return poses_inp, poses_label
    perm = list(range(n_joints))
    for a, b in pairs:
        perm[a], perm[b] = b, a
    joints, maps = {}, {}  # keyed by id(): a tensor shared by input and label must be flipped only once
    centred = set()        # ids of the past-path blob maps (input traj_map)
    for d in (poses_inp, poses_label):
        for key in ('pose', 'pose_gt'):
            if key in d['pose']:
                joints[id(d['pose'][key])] = d['pose'][key]
        if 'traj_map' in d['pose']:
            maps[id(d['pose']['traj_map'])] = d['pose']['traj_map']
            if d is poses_inp:
                centred.add(id(d['pose']['traj_map']))
        if 'map' in d['env']:
            maps[id(d['env']['map'])] = d['env']['map']
    for t in joints.values():
        _mirror_joints(t, flip, perm)
    for k, t in maps.items():
        _flip_rows(t, flip, centred=k in centred)
    return poses_inp, poses_label
