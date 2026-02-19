"""
Split conformal prediction for guide trajectory in 2D space.
Uses calibration trajectories (lists of positions) to build a tube: at each normalized
step t the prediction region is disk(centroid_t, q). Nonconformity score per trajectory
= max_t distance(position_t, centroid_t); q = calibrated quantile.
"""

import numpy as np


def _positions_from_trajectory(start_pos, traj):
    """Extract (x,y) list from start_pos and list of (s, r, pos). Filters None positions."""
    out = []
    if start_pos is not None and len(start_pos) >= 2:
        out.append([float(start_pos[0]), float(start_pos[1])])
    for _, _, pos in (traj or []):
        if pos is not None and len(pos) >= 2:
            out.append([float(pos[0]), float(pos[1])])
    return out


def _interpolate_to_n_points(path_xy, n):
    """Resample path (list of [x,y]) to n points with linear interpolation by index."""
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if len(path_xy) < 2:
        return np.tile(path_xy[0] if len(path_xy) == 1 else [0, 0], (n, 1))
    indices = np.linspace(0, len(path_xy) - 1, n, endpoint=True)
    xs = np.interp(indices, np.arange(len(path_xy)), path_xy[:, 0])
    ys = np.interp(indices, np.arange(len(path_xy)), path_xy[:, 1])
    return np.column_stack([xs, ys])


class ConformalSpace:
    """
    Conformal prediction tube for 2D guide trajectory.
    Calibration: multiple trajectories (each list of (s, r, pos)); normalize to same length,
    compute centroid path, score each trajectory as max_t distance to centroid; take quantile.
    Prediction: at each normalized step t, region = disk(centroid_t, q).
    """

    def __init__(self, n_steps=100):
        """
        Args:
            n_steps: Number of normalized steps to resample each trajectory to.
        """
        self.n_steps = n_steps
        self._quantile = None
        self._alpha = None
        self._n_calibration = 0
        self._centroid_path = None  # (n_steps, 2)

    def calibrate(self, calibration_trajectories_with_start, alpha=0.1):
        """
        Run conformal calibration from trajectories.

        Args:
            calibration_trajectories_with_start: List of (start_pos, trajectory) where
                trajectory is list of (s, r, pos). Or list of trajectory lists where
                we use first point as start: then each item is list of (s, r, pos) and
                we need start_pos from caller. So we accept list of (start_pos, traj) or
                list of list of (s,r,pos) and we take start from first (s,r,pos) by getting
                pos from env - actually the caller has start_pos separately. So signature:
                list of tuples (start_pos, traj) with traj = [(s,r,pos), ...].
            alpha: Miscoverage level (target coverage 1 - alpha).

        Returns:
            self (for chaining).
        """
        n = self.n_steps
        paths = []
        for item in calibration_trajectories_with_start:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                start_pos, traj = item
            else:
                continue
            path_xy = _positions_from_trajectory(start_pos, traj)
            if len(path_xy) < 2:
                continue
            path_resampled = _interpolate_to_n_points(path_xy, n)
            paths.append(path_resampled)
        paths = np.array(paths)  # (K, n_steps, 2)
        if len(paths) == 0:
            raise ValueError("ConformalSpace: no valid calibration trajectories.")
        centroid = np.mean(paths, axis=0)  # (n_steps, 2)
        # Per-trajectory score: max_t ||p_i(t) - c_t||
        scores = np.max(np.linalg.norm(paths - centroid[np.newaxis, :, :], axis=2), axis=1)
        idx = min(int(np.ceil((len(scores) + 1) * (1.0 - alpha))), len(scores)) - 1
        idx = max(0, idx)
        self._quantile = float(np.sort(scores)[idx])
        self._alpha = alpha
        self._n_calibration = len(paths)
        self._centroid_path = centroid
        return self

    def radius(self):
        """Calibrated tube radius (same at all steps)."""
        return self._quantile

    def centroid_path(self):
        """Centroid path shape (n_steps, 2). None if not calibrated."""
        return self._centroid_path

    def region_at_step(self, t):
        """
        Prediction region at normalized step t: (centroid_t, radius).
        Returns ((x,y), r) or (None, None) if not calibrated.
        """
        if self._centroid_path is None or self._quantile is None:
            return (None, None)
        t = int(np.clip(t, 0, self.n_steps - 1))
        return (tuple(self._centroid_path[t].tolist()), self._quantile)

    def trajectory_score(self, start_pos, traj):
        """
        Nonconformity score for one trajectory: max_t distance to centroid.
        Traj is list of (s, r, pos). Returns float or None if not calibrated.
        """
        if self._centroid_path is None:
            return None
        path_xy = _positions_from_trajectory(start_pos, traj)
        if len(path_xy) < 2:
            return None
        path_resampled = _interpolate_to_n_points(path_xy, self.n_steps)
        d = np.linalg.norm(path_resampled - self._centroid_path, axis=1)
        return float(np.max(d))

    @property
    def quantile(self):
        return self._quantile

    @property
    def alpha(self):
        return self._alpha

    @property
    def n_calibration(self):
        return self._n_calibration

    def is_calibrated(self):
        return self._quantile is not None
