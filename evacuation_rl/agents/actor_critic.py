"""
Actor-Critic module for training the guide agent.

- State: [dir_to_avg_pos_xy, avg_vel_dir_xy, astar_dir_xy, x_norm, y_norm] (8-dim): crowd centroid/velocity direction, A* to exit, and guide's normalized position in the room.
- Action: [vx, vy, confidence_logit]. vx, vy in [-1, 1]; confidence = sigmoid(confidence_logit) in (0, 1).
  When confidence > threshold, trainer uses dedicated "go find" behavior instead of (vx, vy).

Two-actor design: ActorMove (how to move) and ActorGoFind (whether to random pathfind).
- ActorMove trained only on steps where (vx, vy) was used.
- ActorGoFind trained on every step so the go_find threshold learns when to trigger.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Experience replay buffer for off-policy training.
    
    WARNING: Off-policy learning has fundamental limitations:
    - Stored reward r and s_next are based on old action and old environment state
    - If current policy would choose different action, r and s_next may not match
    - Importance sampling can only correct action distribution, not reward/state mismatch
    - Solutions: limit buffer age, use on-policy learning, or accept the limitation
    """

    def __init__(self, capacity=100000, max_age=None):
        """
        Args:
            capacity: Maximum buffer size
            max_age: Maximum age of experiences (in number of steps). If None, no age limit.
                     Older experiences will be discarded when sampling.
        """
        self.buffer = deque(maxlen=capacity)
        self.max_age = max_age
        self.step_counter = 0  # Track current step for age-based filtering

    def push(self, s, a, r, s_next, done, update_actor_flag, used_go_find=None, old_policy_prob=None):
        """Store a transition: (s, a, r, s_next, done, update_actor_flag, used_go_find, step_id, old_policy_prob).
        
        Args:
            old_policy_prob: log probability of action a under the policy at storage time.
                           Used for importance sampling correction.
        """
        self.step_counter += 1
        self.buffer.append((s.copy(), a.copy(), r, s_next.copy() if s_next is not None else None, done, update_actor_flag, used_go_find, self.step_counter, old_policy_prob))

    def sample(self, batch_size):
        """Sample a batch of transitions, optionally filtering by age."""
        if self.max_age is not None:
            # Only sample from recent experiences
            current_step = self.step_counter
            valid_experiences = [
                exp for exp in self.buffer
                if current_step - exp[7] <= self.max_age  # exp[7] is step_id (8th element, 0-indexed)
            ]
            if len(valid_experiences) == 0:
                # Fallback to all experiences if none are recent enough
                valid_experiences = list(self.buffer)
            batch = random.sample(valid_experiences, min(batch_size, len(valid_experiences)))
        else:
            batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Unpack: (s, a, r, s_next, done, update_actor_flag, used_go_find, step_id, old_policy_prob)
        s_batch, a_batch, r_batch, s_next_batch, done_batch, update_actor_batch, used_go_find_batch, _, old_policy_prob_batch = zip(*batch)
        return s_batch, a_batch, r_batch, s_next_batch, done_batch, update_actor_batch, used_go_find_batch, old_policy_prob_batch

    def __len__(self):
        return len(self.buffer)


class ActorMove(nn.Module):
    """Gaussian policy: state -> (vx, vy). Trained only when (vx, vy) was used for the step."""

    def __init__(self, state_dim, hidden_sizes=(64, 64), log_std_init=-0.5):
        super().__init__()
        self.net = mlp([state_dim] + list(hidden_sizes), output_activation=nn.Identity)
        self.mean_layer = nn.Linear(hidden_sizes[-1], 2)
        self.log_std = nn.Parameter(torch.full((2,), log_std_init))

    def forward(self, s):
        x = self.net(s)
        mean_vel = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean_vel)
        return mean_vel, std

    def get_action(self, s, deterministic=False):
        mean, std = self.forward(s)
        if deterministic:
            return torch.clamp(mean, -1.0, 1.0)
        a = Normal(mean, std).sample()
        return torch.clamp(a, -1.0, 1.0)


class ActorGoFind(nn.Module):
    """Gaussian policy: state -> scalar logit (sigmoid = prob to use go_find). Trained on every step."""

    def __init__(self, state_dim, hidden_sizes=(64, 64), log_std_init=0.0):
        super().__init__()
        self.net = mlp([state_dim] + list(hidden_sizes), output_activation=nn.Identity)
        self.mean_layer = nn.Linear(hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(torch.full((1,), log_std_init))

    def forward(self, s):
        x = self.net(s)
        mean = self.mean_layer(x).squeeze(-1)
        std = torch.exp(self.log_std).expand(mean.shape)
        return mean, std

    def get_action(self, s, deterministic=False):
        mean, std = self.forward(s)
        if deterministic:
            return mean
        return Normal(mean, std).sample()


class Critic(nn.Module):
    """State-action value function Q(s, a, extras): (state, extras, action) -> scalar. extras are passed to critic only."""

    def __init__(self, state_dim, action_dim, critic_extra_dim=0, hidden_sizes=(64, 64)):
        super().__init__()
        input_dim = state_dim + critic_extra_dim + action_dim
        self.net = mlp([input_dim] + list(hidden_sizes) + [1], output_activation=nn.Identity)

    def forward(self, s, a, extras):
        x = torch.cat([s, extras, a], dim=-1)
        return self.net(x).squeeze(-1)


class ValueCritic(nn.Module):
    """State value function V(s): state -> scalar. Used for off-policy learning."""

    def __init__(self, state_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([state_dim] + list(hidden_sizes) + [1], output_activation=nn.Identity)

    def forward(self, s):
        return self.net(s).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Single-actor Actor-Critic for the guide:
    - Actor: ActorMove, outputs (vx, vy) in [-1, 1]^2
    - Critic: Q(s, a) with a = (vx, vy)
    Uses on-policy TD with advantage A = td_target - Q(s,a).
    """

    def __init__(
        self,
        state_dim,
        action_dim=2,
        critic_extra_dim=6,
        hidden_sizes=(64, 64),
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        log_std_init=0.0,
        optimizer_type="adamw",
        weight_decay=1e-2,
    ):
        super().__init__()
        self.actor = ActorMove(state_dim, hidden_sizes, log_std_init=log_std_init)
        self.critic = Critic(state_dim, action_dim, critic_extra_dim=critic_extra_dim, hidden_sizes=hidden_sizes)
        self.gamma = gamma
        opt_type = (optimizer_type or "adam").lower()
        wd = float(weight_decay) if weight_decay is not None else 0.0
        if opt_type == "adamw":
            self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=wd)
            self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=wd)
        else:
            self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=wd)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=wd)

    def get_action(self, s, deterministic=False):
        """Return numpy action (vx, vy)."""
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            a = self.actor.get_action(s, deterministic=deterministic)
        return a.squeeze(0).numpy()

    def get_action_tensor(self, s, deterministic=True):
        """Return action tensor (batch, action_dim) for critic target."""
        return self.actor.get_action(s, deterministic=deterministic)

    def get_value(self, s, extras):
        """V(s) = Q(s, extras, π(s)): value at s under current policy (for conformal / display). extras: (6,) array (env 5 + effective_speed)."""
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().unsqueeze(0)
        if isinstance(extras, np.ndarray):
            extras = torch.from_numpy(extras).float().unsqueeze(0)
        with torch.no_grad():
            a = self.get_action_tensor(s, deterministic=True)
            return self.critic(s, a, extras).squeeze(0).item()

    def update(self, s, a, r, s_next, done, extras, extras_next):
        """
        On-policy TD update:
        - Critic: minimize (Q(s,extras,a) - (r + γ Q(s',extras_next,π(s'))))^2
        - Actor: policy gradient with advantage A = td_target - Q(s,extras,a)
        """
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        extras = torch.as_tensor(extras, dtype=torch.float32).unsqueeze(0)
        extras_next = torch.as_tensor(extras_next, dtype=torch.float32).unsqueeze(0)
        r = float(r)
        s_next = torch.as_tensor(s_next, dtype=torch.float32).unsqueeze(0)
        done = bool(done)

        # Critic update
        q = self.critic(s, a, extras).squeeze()
        with torch.no_grad():
            a_next = self.get_action_tensor(s_next, deterministic=True)
            q_next = self.critic(s_next, a_next, extras_next).squeeze() if not done else 0.0
            td_target = r + self.gamma * q_next
        loss_critic = F.mse_loss(q, td_target)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # Advantage for actor
        advantage = (td_target - q).detach()

        # Actor update
        mean, std = self.actor(s)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(a).sum(dim=-1)
        loss_actor = -(log_prob * advantage).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        return loss_actor.item(), loss_critic.item()
