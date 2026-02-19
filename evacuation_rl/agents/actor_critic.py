"""
Actor-Critic module for training the guide agent.

- State: [distance_to_exit_norm, astar_dir_xy, n_in_guide_range, x_norm, y_norm] (6-dim).
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


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


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
    """State-action value function Q(s, a): (state, action) -> scalar."""

    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = mlp([input_dim] + list(hidden_sizes) + [1], output_activation=nn.Identity)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Two actors + Critic: ActorMove (vx, vy) and ActorGoFind (logit for go_find).
    Critic is Q(s, a); get_value(s) returns Q(s, π(s)) for conformal/display.
    get_action returns (vx, vy, logit). update(s, a, r, s_next, done, update_actor):
    - Critic: always (TD on Q(s,a)).
    - ActorMove: only when update_actor=True (step used (vx, vy)).
    - ActorGoFind: always when update_actor is not False.
    """

    def __init__(
        self,
        state_dim,
        action_dim=3,
        hidden_sizes=(64, 64),
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        log_std_init=0.0,
        optimizer_type="adamw",
        weight_decay=1e-2,
    ):
        super().__init__()
        self.actor_move = ActorMove(state_dim, hidden_sizes, log_std_init=log_std_init)
        self.actor_go_find = ActorGoFind(state_dim, hidden_sizes, log_std_init=0.0)
        self.critic = Critic(state_dim, action_dim, hidden_sizes)
        self.gamma = gamma
        opt_type = (optimizer_type or "adam").lower()
        wd = float(weight_decay) if weight_decay is not None else 0.0
        if opt_type == "adamw":
            self.opt_actor_move = torch.optim.AdamW(self.actor_move.parameters(), lr=lr_actor, weight_decay=wd)
            self.opt_actor_go_find = torch.optim.AdamW(self.actor_go_find.parameters(), lr=lr_actor, weight_decay=wd)
            self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=wd)
        else:
            self.opt_actor_move = torch.optim.Adam(self.actor_move.parameters(), lr=lr_actor, weight_decay=wd)
            self.opt_actor_go_find = torch.optim.Adam(self.actor_go_find.parameters(), lr=lr_actor, weight_decay=wd)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=wd)
        # For scheduler / backward compat: single "actor" optimizer steps both (train_guide steps scheduler_actor once per episode)
        self.opt_actor = self.opt_actor_move

    def get_action(self, s, deterministic=False):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            vxvy = self.actor_move.get_action(s, deterministic=deterministic)
            logit = self.actor_go_find.get_action(s, deterministic=deterministic)
            if logit.dim() == 0:
                logit = logit.unsqueeze(0)
            a = torch.cat([vxvy, logit.unsqueeze(-1) if logit.dim() == 1 else logit], dim=-1)
        return a.squeeze(0).numpy()

    def get_value(self, s):
        """V(s) = Q(s, π(s)): value at s under current policy (for conformal / display)."""
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().unsqueeze(0)
        with torch.no_grad():
            a = self.get_action_tensor(s, deterministic=True)
            return self.critic(s, a).squeeze(0).item()

    def get_action_tensor(self, s, deterministic=True):
        """Return action tensor (batch, action_dim) for use inside the module (e.g. critic target)."""
        vxvy = self.actor_move.get_action(s, deterministic=deterministic)
        logit = self.actor_go_find.get_action(s, deterministic=deterministic)
        if logit.dim() == 0:
            logit = logit.unsqueeze(0)
        if logit.dim() == 1:
            logit = logit.unsqueeze(-1)
        return torch.cat([vxvy, logit], dim=-1)

    def update(self, s, a, r, s_next, done, update_actor=True):
        """
        update_actor: True = update both ActorMove and ActorGoFind (only ActorMove when step used (vx,vy));
                     "confidence_only" = update only ActorGoFind (step used go_find);
                     False = no actor updates.
        Pass used_go_find from trainer: update_actor=False when used_go_find, "confidence_only" when used_go_find.
        """
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
        r = float(r)
        s_next = torch.as_tensor(s_next, dtype=torch.float32).unsqueeze(0)
        done = bool(done)
        a_move, a_logit = a[:, :2], a[:, 2:3]

        q = self.critic(s, a).squeeze()
        with torch.no_grad():
            a_next = self.get_action_tensor(s_next, deterministic=True)
            q_next = self.critic(s_next, a_next).squeeze() if not done else 0.0
            td_target = r + self.gamma * q_next
        loss_critic = F.mse_loss(q, td_target)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        # Advantage = td_target - Q(s,a). Positive = this (s,a) did better than expected.
        advantage = (td_target - q).detach()
        loss_move_val, loss_go_val = 0.0, 0.0

        # ActorGoFind: policy gradient on the sampled logit. Reinforce logit when advantage > 0 (decision was good), penalize when advantage < 0.
        if update_actor is not False:
            mean_g, std_g = self.actor_go_find(s)
            dist_g = Normal(mean_g.squeeze(-1), std_g.squeeze(-1))
            log_prob_g = dist_g.log_prob(a_logit.squeeze(-1))
            loss_go = -(log_prob_g * advantage).mean()
            loss_go_val = loss_go.item()
            self.opt_actor_go_find.zero_grad()
            loss_go.backward()
            self.opt_actor_go_find.step()

        # ActorMove: only when (vx, vy) was actually used (not go_find)
        if update_actor is True:
            mean_m, std_m = self.actor_move(s)
            dist_m = Normal(mean_m, std_m)
            log_prob_m = dist_m.log_prob(a_move).sum(dim=-1)
            loss_move = -(log_prob_m * advantage).mean()
            loss_move_val = loss_move.item()
            self.opt_actor_move.zero_grad()
            loss_move.backward()
            self.opt_actor_move.step()

        return loss_move_val + loss_go_val, loss_critic.item()
