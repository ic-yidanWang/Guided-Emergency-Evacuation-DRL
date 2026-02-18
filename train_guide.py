"""
Train the guide agent with Actor-Critic.

Usage:
  python train_guide.py [--config config/simulation_config.json] [--no-viz]
"""

import argparse
import os
import numpy as np

from evacuation_rl.utils.config_loader import load_config
from evacuation_rl.utils.simulation import setup_environment
from evacuation_rl.agents.actor_critic import ActorCritic
from evacuation_rl.utils import visualization


def main():
    parser = argparse.ArgumentParser(description='Train guide agent with Actor-Critic')
    parser.add_argument('--config', '-c', type=str, default='config/simulation_config.json')
    parser.add_argument('--no-viz', action='store_true', help='Disable real-time visualization')
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get('train', {})
    if not train_cfg:
        train_cfg = {
            'visualize': True, 'refresh_interval': 5, 'max_guide_speed': 2.0,
            'reward_scale': -0.1, 'episodes': 50, 'steps_per_episode': 200,
            'lr_actor': 3e-4, 'lr_critic': 1e-3, 'gamma': 0.99,
        }

    do_visualize = train_cfg.get('visualize', True) and not args.no_viz
    refresh_interval = int(train_cfg.get('refresh_interval', 10))
    max_guide_speed = float(train_cfg.get('max_guide_speed', 2.0))
    reward_scale = float(train_cfg.get('reward_scale', -0.1))
    exit_reward_scale = float(train_cfg.get('exit_reward_scale', 2.0))
    guide_boundary_margin = float(train_cfg.get('guide_boundary_margin', 0.8))
    guide_boundary_penalty_scale = float(train_cfg.get('guide_boundary_penalty_scale', 0.5))
    guide_corner_penalty_scale = float(train_cfg.get('guide_corner_penalty_scale', 0.3))
    n_in_range_threshold = float(train_cfg.get('n_in_range_threshold', 0.2))
    reward_toward_exit_scale = float(train_cfg.get('reward_toward_exit_scale', 0.1))
    reward_toward_crowd_scale = float(train_cfg.get('reward_toward_crowd_scale', 0.1))
    episodes = int(train_cfg.get('episodes', 50))
    steps_per_episode = int(train_cfg.get('steps_per_episode', 200))
    lr_actor = float(train_cfg.get('lr_actor', 3e-4))
    lr_critic = float(train_cfg.get('lr_critic', 1e-3))
    gamma = float(train_cfg.get('gamma', 0.99))
    log_std_init = float(train_cfg.get('log_std_init', 0.0))
    exploration_noise_std = float(train_cfg.get('exploration_noise_std', 0.25))
    exploration_decay = float(train_cfg.get('exploration_decay', 0.995))

    if not config['agents'].get('add_guide_agent', False):
        print("Config has add_guide_agent=false. Enable it to train the guide.")
        return

    # One env to get state_dim (depends on num exits)
    env = setup_environment(config)
    state = env.get_guide_state()
    if state is None:
        print("No guide or no exits in env. Cannot train.")
        return
    state_dim = state.shape[0]
    print(f"State dim: {state_dim} (dist_to_exit, astar_dir_xy, n_in_range)")

    agent = ActorCritic(
        state_dim=state_dim,
        action_dim=2,
        hidden_sizes=(64, 64),
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        log_std_init=log_std_init,
    )

    domain = {
        'x': env.L[0, 1] - env.L[0, 0],
        'y': env.L[1, 1] - env.L[1, 0],
        'z': env.L[2, 1] - env.L[2, 0],
    }
    obstacle_configs = config.get('obstacles', [])
    agent_size = config.get('exit_parameters', {}).get('agent_size', 0.18)
    guide_size = config.get('guide_parameters', {}).get('guide_size', 0.25)
    guide_radius = config.get('guide_parameters', {}).get('guide_radius')

    fig, ax_evac, ax_reward = None, None, None
    if do_visualize:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, (ax_evac, ax_reward) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Train Guide — Actor-Critic", fontsize=12)
        visualization.draw_reward_curve(ax_reward, [], max_episodes=episodes, fig=fig)
        fig.show()
        fig.canvas.draw()

    total_reward_per_episode = []

    for ep in range(episodes):
        if ep == 0:
            env = setup_environment(config)
        else:
            env.reset_guided(quiet=True)
        ep_reward = 0.0
        s = env.get_guide_state()
        if s is None:
            continue

        # Exploration noise decays per episode (so later episodes are more exploitative)
        noise_std = exploration_noise_std * (exploration_decay ** ep)
        for step in range(steps_per_episode):
            a = agent.get_action(s, deterministic=False)
            a = a + noise_std * np.random.randn(2)
            a = np.clip(a, -1.0, 1.0)
            guide_actions = [[float(a[0]), float(a[1])]]
            done = env.step_guided(guide_actions=guide_actions, max_guide_speed=max_guide_speed)
            r = (env.get_evacuation_reward(scale=reward_scale) + exit_reward_scale * env.get_exit_reward()
                - env.get_guide_boundary_penalty(margin=guide_boundary_margin, penalty_scale=guide_boundary_penalty_scale, corner_extra_scale=guide_corner_penalty_scale)
                + env.get_guide_dense_reward(n_in_range_threshold=n_in_range_threshold,
                    reward_toward_exit_scale=reward_toward_exit_scale,
                    reward_toward_crowd_scale=reward_toward_crowd_scale))
            s_next = env.get_guide_state()
            ep_reward += r

            s_next_valid = s_next if s_next is not None else s
            agent.update(s, a, r, s_next_valid, done=done)
            if s_next is not None:
                s = s_next
            if done:
                break

            if do_visualize and (step % refresh_interval == 0):
                if ax_evac is not None:
                    visualization.draw_training_frame(
                        ax_evac, env, domain, obstacle_configs,
                        agent_size=agent_size, guide_size=guide_size, guide_radius=guide_radius,
                        episode=ep + 1, total_episodes=episodes, step=step, ep_reward=ep_reward,
                        fig=fig,
                    )
                if ax_reward is not None:
                    visualization.draw_reward_curve(ax_reward, total_reward_per_episode, max_episodes=episodes, fig=fig)

        total_reward_per_episode.append(ep_reward)
        # Update reward curve after each episode
        if do_visualize and ax_reward is not None:
            visualization.draw_reward_curve(ax_reward, total_reward_per_episode, max_episodes=episodes, fig=fig)
        agents_xy, guide_agents_xy = env.get_all_positions_for_vis()
        n_agents, n_guides = len(agents_xy), len(guide_agents_xy)
        print(f"Episode {ep+1}/{episodes}  total_reward={ep_reward:.2f}  steps={step+1}  Agents: {n_agents}  Guides: {n_guides}")

    if do_visualize and fig is not None:
        import matplotlib.pyplot as plt
        if ax_reward is not None and total_reward_per_episode:
            visualization.draw_reward_curve(ax_reward, total_reward_per_episode, max_episodes=episodes, fig=fig)
        plt.ioff()
        # Only block on show if the window is still open (user may have closed it)
        if fig.number in plt.get_fignums():
            plt.show(block=True)
        else:
            plt.close(fig)

    print("\nTraining finished.")
    if total_reward_per_episode:
        n = min(10, len(total_reward_per_episode))
        print(f"Mean episode reward (last {n}): {sum(total_reward_per_episode[-n:]) / n:.2f}")


if __name__ == '__main__':
    main()
