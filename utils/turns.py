# =====================================================================================================
# utils/turns.py: automatic turn detection for path evaluation (CLAUDE/pose_conditioned_path.md §4)
# =====================================================================================================
# Every quantity is computed from a window's past root path `h` and its TRUE future root path `g`, both
# (N, T, 2) in the actor frame (metres). Because `g` is the ground-truth future, these labels are for
# evaluation and analysis only. Never feed them to a model.
#
# One shared implementation for eval_path.py, train_path.py (validation) and later analysis tools, so the
# definitions can't drift apart. Thresholds can be overridden by a `turns:` block in the cfg.
# =====================================================================================================
import numpy as np

DEFAULT_THRESHOLDS = {
    'dt': 0.2,                 # s per frame (5 Hz)
    'vel_frames': 4,           # current velocity = h[-1] - h[-1 - vel_frames]  (0.8 s)
    'turn_lat': 0.75,          # m: lateral offset at the horizon end -> 'turn' (primary label)
    'turn_heading': 30.0,      # deg: heading change -> 'turn_heading' (secondary label)
    'straight_lat_max': 0.5,   # m: max lateral offset over the whole horizon for 'straight'
    'hist_curv_new': 15.0,     # deg: past path bent less than this -> the turn is 'new' (not visible yet)
    'onset_lat': 0.25,         # m: lateral offset that marks the start of a turn
    'lat_bins': [0.0, 0.25, 0.75, 1.5],  # difficulty bins on lat_end (last bin is open-ended)
}

BASE_SUBSETS = ['all', 'straight', 'turn', 'turn_heading', 'turn_new', 'turn_ongoing']


def get_thresholds(cfg=None):
    return {**DEFAULT_THRESHOLDS, **((cfg or {}).get('turns') or {})}


def _angle(a, b):
    # Signed angle from a to b in degrees (counter-clockwise positive). Zero vectors give 0.
    return np.degrees(np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1)))


def _unit(v, fallback, eps=1e-6):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / np.maximum(n, eps), fallback)


def turn_features(h, g, th=None):
    th = th or DEFAULT_THRESHOLDS
    h, g = np.asarray(h, dtype=np.float64), np.asarray(g, dtype=np.float64)
    k = th['vel_frames']

    v = h[:, -1] - h[:, -1 - k]
    u = _unit(v, np.array([1.0, 0.0]))  # direction of motion; actor-frame x (≈ forward) if standing still
    rel = g - h[:, -1:, :]               # future positions relative to the current one
    # Signed distance from the straight-line extrapolation (+ = to the left of the walking direction).
    lat = u[:, None, 0] * rel[..., 1] - u[:, None, 1] * rel[..., 0]

    # Past curvature: direction over the first vel_frames of the past vs the last vel_frames.
    hist_curv = np.abs(_angle(h[:, k] - h[:, 0], v))

    over = np.abs(lat) > th['onset_lat']
    onset = np.where(over.any(axis=1), over.argmax(axis=1), -1)  # first future frame off the line, -1 = never

    path = np.concatenate((h[:, -1:], g), axis=1)
    return {
        'lat_end': np.abs(lat[:, -1]),
        'lat_end_signed': lat[:, -1],
        'lat_max': np.abs(lat).max(axis=1),
        'head_chg': np.abs(_angle(v, rel[:, -1])),
        'hist_curv': hist_curv,
        'onset': onset,
        'speed_past': np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(axis=1) / th['dt'],
        'len_future': np.linalg.norm(np.diff(path, axis=1), axis=-1).sum(axis=1),
    }


def lat_bin_names(th=None):
    edges = (th or DEFAULT_THRESHOLDS)['lat_bins']
    return [f'lat_{lo:.2f}-{hi:.2f}' for lo, hi in zip(edges[:-1], edges[1:])] + [f'lat_{edges[-1]:.2f}+']


def turn_labels(f, th=None):
    # Boolean masks per subset. 'turn_new' / 'turn_ongoing' split the turns by whether the past root path
    # already bends: a new turn can only be anticipated from cues other than the past path (e.g. body pose).
    th = th or DEFAULT_THRESHOLDS
    turn = f['lat_end'] > th['turn_lat']
    labels = {
        'all': np.ones_like(turn),
        'straight': ~turn & (f['lat_max'] < th['straight_lat_max']),
        'turn': turn,
        'turn_heading': f['head_chg'] > th['turn_heading'],
        'turn_new': turn & (f['hist_curv'] < th['hist_curv_new']),
        'turn_ongoing': turn & (f['hist_curv'] >= th['hist_curv_new']),
    }
    edges = list(th['lat_bins']) + [np.inf]
    for name, lo, hi in zip(lat_bin_names(th), edges[:-1], edges[1:]):
        labels[name] = (f['lat_end'] >= lo) & (f['lat_end'] < hi)
    return labels
