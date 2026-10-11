# =====================================================================================================
# utils/sequence_order.py: recover the time order of the windows in a Real-IM .npz file
# =====================================================================================================
# The preprocessed files store each sequence's 30-frame windows in SHUFFLED order: window i+1 is not the
# window after window i (consecutive entries are ~20 m apart). Each file is one continuous walk sampled with
# a stride of 1 frame, so window j follows window i exactly when j's first frames equal i's frames 1, 2, ...
# Matching those overlaps (in world coordinates, data['joints']) rebuilds the timeline.
#
# Used for leak-free validation splits (train_path.py) and for per-window time stamps in eval outputs.
# =====================================================================================================
import numpy as np
from scipy.spatial import cKDTree


def temporal_rank(joints_world, root_idx, n_match=5, tol=1e-3):
    # joints_world: (N, T, J, 3) world-frame windows. Returns (rank, n_chains):
    #   rank[i]   time position of window i (0 = earliest). Consecutive ranks are 1 frame (0.2 s) apart, so
    #             windows whose ranks differ by less than T share frames.
    #   n_chains  number of separate continuous pieces found (1 for every Real-IM file).
    root = np.asarray(joints_world)[:, :, root_idx, :]
    n = len(root)
    src = root[:, 1:1 + n_match].reshape(n, -1)  # frames 1..n_match of window i ...
    dst = root[:, :n_match].reshape(n, -1)       # ... equal frames 0..n_match-1 of its successor
    dist, idx = cKDTree(dst).query(src)
    succ = np.where((dist < tol) & (idx != np.arange(n)), idx, -1)

    has_pred = np.zeros(n, dtype=bool)
    has_pred[succ[succ >= 0]] = True
    starts = np.where(~has_pred)[0]

    rank = np.full(n, -1)
    t = 0
    for s in starts:
        i = s
        while i >= 0 and rank[i] < 0:
            rank[i] = t
            t += 1
            i = succ[i]
    for i in np.where(rank < 0)[0]:  # only reachable through a cycle, which a real walk can't produce
        rank[i] = t
        t += 1
    return rank, len(starts)
