"""
Actor-Critic module for training the guide agent.

- State: [distance_to_exit_norm, astar_dir_x, astar_dir_y, n_in_guide_range] (4-dim, relative only).
- Action: [vx, vy] in [-1, 1], interpreted as direction (normalized) and magnitude; env scales by max_guide_speed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy: state -> mean action (2), with learnable log_std for (vx, vy)."""

    def __init__(self, state_dim, action_dim=2, hidden_sizes=(64, 64), log_std_init=-0.5):
        super().__init__()
        self.action_dim = action_dim
        self.net = mlp([state_dim] + list(hidden_sizes), output_activation=nn.Identity)
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, s):
        x = self.net(s)
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, s, deterministic=False):
        mean, std = self.forward(s)
        if deterministic:
            return mean
        dist = Normal(mean, std)
        a = dist.sample()
        return torch.clamp(a, -1.0, 1.0)


class Critic(nn.Module):
    """State value function: state -> scalar."""

    def __init__(self, state_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([state_dim] + list(hidden_sizes) + [1], output_activation=nn.Identity)

    def forward(self, s):
        return self.net(s).squeeze(-1)


class ActorCritic(nn.Module):
    """Actor + Critic for guide agent. state_dim = 4 (dist_to_exit, astar_dir_xy, n_in_range)."""

    def __init__(self, state_dim, action_dim=2, hidden_sizes=(64, 64), lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, log_std_init=0.0):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, hidden_sizes, log_std_init=log_std_init)
        self.critic = Critic(state_dim, hidden_sizes)
        self.gamma = gamma
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_action(self, s, deterministic=False):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            a = self.actor.get_action(s, deterministic=deterministic)
        return a.squeeze(0).numpy()

    def get_value(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            return self.critic(s).squeeze(0).item()

    def update(self, s, a, r, s_next, done):
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        r = float(r)
        s_next = torch.as_tensor(s_next, dtype=torch.float32).unsqueeze(0)
        done = bool(done)

        v = self.critic(s).squeeze()
        with torch.no_grad():
            v_next = self.critic(s_next).squeeze() if not done else 0.0
            td_target = r + self.gamma * v_next
        loss_critic = F.mse_loss(v, td_target)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        mean, std = self.actor(s)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(a).sum(dim=-1)
        advantage = (td_target - v).detach()
        loss_actor = -(log_prob * advantage).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        return loss_actor.item(), loss_critic.item()
