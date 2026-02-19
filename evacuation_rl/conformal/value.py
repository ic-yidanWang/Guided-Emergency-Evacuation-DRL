"""
Split conformal prediction for the critic value V(s).
Uses calibration data (state, empirical return) to compute a quantile of |G - V(s)|,
then prediction interval at new state s is [V(s) - q, V(s) + q].
"""

import numpy as np


class ConformalValue:
    """
    Conformal prediction intervals for the value function V(s).
    Calibration uses (state, empirical_return) pairs; nonconformity score = |G - V(s)|.
    """

    def __init__(self, agent, gamma=0.99):
        """
        Args:
            agent: ActorCritic (or any with get_value(s) returning scalar).
            gamma: Discount for computing returns (should match training).
        """
        self.agent = agent
        self.gamma = gamma
        self._quantile = None
        self._alpha = None
        self._n_calibration = 0

    def calibrate(self, calibration_trajectories, alpha=0.1):
        """
        Run conformal calibration from a list of trajectories.

        Args:
            calibration_trajectories: List of episodes. Each episode is a list of
                (state, reward) tuples: [(s_0, r_0), (s_1, r_1), ...].
            alpha: Miscoverage level (target coverage is 1 - alpha, e.g. alpha=0.1 -> 90%).

        Returns:
            self (for chaining).
        """
        states, returns = [], []
        for episode in calibration_trajectories:
            if not episode:
                continue
            rewards = [r for _, r in episode]
            sts = [s for s, _ in episode]
            # Empirical return from step t: G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k
            T = len(rewards)
            for t in range(T):
                G_t = 0.0
                for k in range(t, T):
                    G_t += (self.gamma ** (k - t)) * rewards[k]
                states.append(sts[t])
                returns.append(G_t)
        returns = np.asarray(returns, dtype=np.float64)
        n = len(states)
        if n == 0:
            raise ValueError("Calibration set is empty.")
        # Nonconformity scores E_i = |G_i - V(s_i)|
        scores = np.zeros(n)
        for i in range(n):
            v_s = self.agent.get_value(states[i])
            scores[i] = abs(returns[i] - v_s)
        # Split conformal: q = (1-alpha)(1+1/n) quantile of scores
        idx = min(int(np.ceil((n + 1) * (1.0 - alpha))), n) - 1
        idx = max(0, idx)
        self._quantile = float(np.sort(scores)[idx])
        self._alpha = alpha
        self._n_calibration = n
        return self

    def interval(self, s):
        """
        Prediction interval for value at state s.
        Returns (lower, upper) or None if not calibrated.
        """
        if self._quantile is None:
            return None
        v = self.agent.get_value(s)
        return (v - self._quantile, v + self._quantile)

    def predict(self, s):
        """Point prediction V(s)."""
        return self.agent.get_value(s)

    @property
    def quantile(self):
        """Calibrated quantile (radius of interval)."""
        return self._quantile

    @property
    def alpha(self):
        """Miscoverage level (1-alpha = target coverage)."""
        return self._alpha

    @property
    def n_calibration(self):
        """Number of calibration points used."""
        return self._n_calibration

    def is_calibrated(self):
        return self._quantile is not None
