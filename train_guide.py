"""
Train the guide agent with Actor-Critic.

Usage:
  python train_guide.py [--config config/simulation_config.json] [--no-viz]

Outputs (from this script):
  - Model checkpoints: when save_model=true, to train.save_path (and every N episodes if save_every_n_episodes > 0).
  - Conformal figures: when train.do_value_conformal or train.do_space_conformal, to respective
    output_dir; value figure: conformal_value_interval_ep{N}_*.png, space: conformal_space_tube_ep{N}_*.png.
  those come from run_guided_visualize.py (output/guided/frames/).
"""

import argparse
import os
import numpy as np
import torch

from evacuation_rl.utils.config_loader import load_config
from evacuation_rl.utils.simulation import setup_environment
from evacuation_rl.agents.actor_critic import ActorCritic
from evacuation_rl.conformal import ConformalValue, ConformalSpace
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
    exit_reward_scale = float(train_cfg.get('exit_reward_scale', 2.0))
    guide_boundary_margin = float(train_cfg.get('guide_boundary_margin', 0.8))
    guide_boundary_penalty_scale = float(train_cfg.get('guide_boundary_penalty_scale', 0.5))
    guide_corner_penalty_scale = float(train_cfg.get('guide_corner_penalty_scale', 0.3))
    n_in_range_count_threshold = int(train_cfg.get('n_in_range_count_threshold', 1))
    reward_toward_exit_scale = float(train_cfg.get('reward_toward_exit_scale', 0.1))
    reward_toward_crowd_scale = float(train_cfg.get('reward_toward_crowd_scale', 0.1))
    go_find_confidence_threshold = float(train_cfg.get('go_find_confidence_threshold', 0.5))
    go_find_min_distance = float(train_cfg.get('go_find_min_distance', 3.0))
    go_find_max_distance = float(train_cfg.get('go_find_max_distance', 5.0))
    go_find_stick_steps = int(train_cfg.get('go_find_stick_steps', 5))
    go_find_alone_bonus_scale = float(train_cfg.get('go_find_alone_bonus_scale', 2.0))
    episodes = int(train_cfg.get('episodes', 50))
    steps_per_episode = int(train_cfg.get('steps_per_episode', 200))
    save_model = train_cfg.get('save_model', False)
    save_path = train_cfg.get('save_path', 'output/guided/guide_agent.pt')
    save_every_n_episodes = int(train_cfg.get('save_every_n_episodes', 0))
    do_value_conformal = train_cfg.get('do_value_conformal', False)
    do_space_conformal = train_cfg.get('do_space_conformal', False)
    do_conformal = do_value_conformal or do_space_conformal
    value_conformal_cfg = config.get('value_conformal', {})
    space_conformal_cfg = config.get('space_conformal', {})
    lr_actor = float(train_cfg.get('lr_actor', 3e-4))
    lr_critic = float(train_cfg.get('lr_critic', 1e-3))
    gamma = float(train_cfg.get('gamma', 0.99))
    log_std_init = float(train_cfg.get('log_std_init', 0.0))
    optimizer_type = train_cfg.get('optimizer', 'adamw')
    weight_decay = float(train_cfg.get('weight_decay', 1e-2))
    lr_scheduler_type = (train_cfg.get('lr_scheduler') or 'cosine').strip().lower() or 'none'
    min_lr_ratio = float(train_cfg.get('min_lr_ratio', 0.1))
    exploration_noise_std = float(train_cfg.get('exploration_noise_std', 0.25))
    exploration_decay = float(train_cfg.get('exploration_decay', 0.995))

    if not config['agents'].get('add_guide_agent', False):
        print("Config has add_guide_agent=false. Enable it to train the guide.")
        return

    env = setup_environment(config)
    state = env.get_guide_state()
    if state is None:
        print("No guide or no exits in env. Cannot train.")
        return
    state_dim = state.shape[0]
    print(f"State dim: {state_dim} (dist_to_exit, astar_dir_xy, n_in_range, x_norm, y_norm)")
    print(f"Action: (vx, vy, confidence_logit); go_find if sigmoid(confidence) > {go_find_confidence_threshold}")

    agent = ActorCritic(
        state_dim=state_dim,
        action_dim=3,
        hidden_sizes=(64, 64),
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        log_std_init=log_std_init,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
    )
    scheduler_actor = None
    scheduler_actor_go_find = None
    scheduler_critic = None
    if lr_scheduler_type == 'cosine' and episodes > 0:
        scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.opt_actor_move, T_max=episodes, eta_min=lr_actor * min_lr_ratio
        )
        scheduler_actor_go_find = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.opt_actor_go_find, T_max=episodes, eta_min=lr_actor * min_lr_ratio
        )
        scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.opt_critic, T_max=episodes, eta_min=lr_critic * min_lr_ratio
        )
    print(f"Optimizer: {optimizer_type}, weight_decay={weight_decay}, lr_scheduler={lr_scheduler_type or 'none'}")

    domain = {
        'x': env.L[0, 1] - env.L[0, 0],
        'y': env.L[1, 1] - env.L[1, 0],
        'z': env.L[2, 1] - env.L[2, 0],
    }
    obstacle_configs = config.get('obstacles', [])
    agent_size = config.get('exit_parameters', {}).get('agent_size', 0.18)
    guide_size = config.get('guide_parameters', {}).get('guide_size', 0.25)
    guide_radius = config.get('guide_parameters', {}).get('guide_radius')

    conformal_every_n = 0
    if do_value_conformal:
        conformal_every_n = max(conformal_every_n, int(value_conformal_cfg.get('every_n_episodes', 0)))
    if do_space_conformal:
        conformal_every_n = max(conformal_every_n, int(space_conformal_cfg.get('every_n_episodes', 0)))

    def _run_conformal_snapshot(
        agent, config, episode_label, label_suffix,
        exit_reward_scale, guide_boundary_margin, guide_boundary_penalty_scale, guide_corner_penalty_scale,
        n_in_range_count_threshold, reward_toward_exit_scale,
        go_find_confidence_threshold, go_find_min_distance, go_find_max_distance, max_guide_speed,
        steps_per_episode, gamma,
        obstacle_configs, domain,
    ):
        import matplotlib.pyplot as plt
        v_cfg = config.get('value_conformal', {})
        s_cfg = config.get('space_conformal', {})
        do_value = do_value_conformal
        do_space = do_space_conformal
        if not do_value and not do_space:
            return
        calibration_episodes = max(
            int(v_cfg.get('calibration_episodes', 5)) if do_value else 0,
            int(s_cfg.get('calibration_episodes', 5)) if do_space else 0,
        )
        sim_out = config.get('simulation', {}).get('output_dir', 'output/guided')

        def _reward_fn(env):
            return (exit_reward_scale * env.get_exit_reward()
                    - env.get_guide_boundary_penalty(margin=guide_boundary_margin, penalty_scale=guide_boundary_penalty_scale, corner_extra_scale=guide_corner_penalty_scale)
                    + env.get_guide_dense_reward(n_in_range_count_threshold=n_in_range_count_threshold, reward_toward_exit_scale=reward_toward_exit_scale)
                    + env.get_guide_reward_toward_crowd(reward_toward_crowd_scale=reward_toward_crowd_scale, n_in_range_count_threshold=n_in_range_count_threshold, go_find_alone_bonus_scale=go_find_alone_bonus_scale))

        def _run_episode_trajectory(env, deterministic):
            start_pos = env.get_guide_position()
            traj = []
            s = env.get_guide_state()
            if s is None:
                return traj, start_pos
            for _ in range(steps_per_episode):
                a = agent.get_action(s, deterministic=deterministic)
                confidence = 1.0 / (1.0 + np.exp(-float(a[2])))
                if confidence > go_find_confidence_threshold:
                    dx, dy = env.get_guide_go_find_direction(min_distance=go_find_min_distance, max_distance=go_find_max_distance, stick_steps=go_find_stick_steps)
                    guide_actions = [[float(dx), float(dy)]]
                else:
                    guide_actions = [[float(a[0]), float(a[1])]]
                done = env.step_guided(guide_actions=guide_actions, max_guide_speed=max_guide_speed)
                r = _reward_fn(env)
                pos = env.get_guide_position()
                traj.append((s.copy(), r, pos))
                s_next = env.get_guide_state()
                s = s_next if s_next is not None else s
                if done:
                    break
            return traj, start_pos

        print(f"Conformal: snapshot for ep {episode_label} — calibration {calibration_episodes} ep + 1 eval")
        env_cal = setup_environment(config)
        calibration_with_start = []
        for ep in range(calibration_episodes):
            if ep > 0:
                env_cal.reset_guided(quiet=True)
            traj, start_pos = _run_episode_trajectory(env_cal, deterministic=False)
            if traj:
                calibration_with_start.append((start_pos, traj))
            print(f"  Conformal: calibration ep {ep + 1}/{calibration_episodes} done ({len(traj) if traj else 0} steps)")
        if not calibration_with_start:
            print(f"Conformal ep{episode_label}: no calibration data, skip.")
            return
        print(f"  Conformal: running eval episode...")
        env_cal.reset_guided(quiet=True)
        eval_traj, eval_start_pos = _run_episode_trajectory(env_cal, deterministic=True)
        print(f"  Conformal: eval done ({len(eval_traj)} steps), plotting...")
        if not eval_traj:
            print(f"Conformal ep{episode_label}: empty eval trajectory, skip.")
            return

        # ----- Value conformal -----
        if do_value:
            calibration_trajectories_value = [[(s, r) for s, r, _ in t] for _, t in calibration_with_start]
            alpha_v = float(v_cfg.get('alpha', 0.1))
            cf = ConformalValue(agent, gamma=gamma).calibrate(calibration_trajectories_value, alpha=alpha_v)
            trajectory_xy = [eval_start_pos] + [pos for (_, _, pos) in eval_traj if pos is not None]
            trajectory_xy = [p for p in trajectory_xy if p is not None]
            steps, values, lowers, uppers, empirical_returns = [], [], [], [], []
            T = len(eval_traj)
            for t in range(T):
                s, r, _ = eval_traj[t]
                v = cf.predict(s)
                lo, hi = cf.interval(s)
                G_t = sum((gamma ** (k - t)) * eval_traj[k][1] for k in range(t, T))
                steps.append(t)
                values.append(v)
                lowers.append(lo)
                uppers.append(hi)
                empirical_returns.append(G_t)
            out_dir = v_cfg.get('output_dir') or sim_out
            base_name = (v_cfg.get('figure_name') or 'conformal_value_interval.png').replace('.png', '')
            figure_name = f"{base_name}_ep{episode_label}_{label_suffix}.png"
            fig, (ax_traj, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
            ax_traj.set_aspect('equal')
            ax_traj.set_xlim(float(env_cal.L[0, 0]), float(env_cal.L[0, 1]))
            ax_traj.set_ylim(float(env_cal.L[1, 0]), float(env_cal.L[1, 1]))
            exits_list = [[e['x'], e['y']] for e in config.get('exits', [])]
            visualization.draw_exits(ax_traj, np.array(exits_list) if exits_list else [], domain)
            visualization.draw_obstacles(ax_traj, obstacle_configs=obstacle_configs, domain=domain)
            if len(trajectory_xy) >= 2:
                xs, ys = [p[0] for p in trajectory_xy], [p[1] for p in trajectory_xy]
                ax_traj.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=20, zorder=5)
                ax_traj.plot(xs, ys, 'k-', lw=0.8, alpha=0.6, zorder=4)
                ax_traj.scatter([xs[0]], [ys[0]], c='green', s=120, marker='o', label='Start', zorder=6, edgecolors='black')
                ax_traj.scatter([xs[-1]], [ys[-1]], c='red', s=120, marker='s', label='End', zorder=6, edgecolors='black')
            ax_traj.set_xlabel('x')
            ax_traj.set_ylabel('y')
            ax_traj.set_title(f'Guide trajectory (ep {episode_label})')
            ax_traj.legend(loc='best')
            ax_traj.grid(True, alpha=0.3)
            ax_val.fill_between(steps, lowers, uppers, alpha=0.3, color='steelblue', label=f'Conformal Interval (1-α={1-alpha_v:.1f})')
            ax_val.plot(steps, values, color='navy', lw=2, label='V(s) (Critic Prediction)')
            ax_val.plot(steps, empirical_returns, color='darkorange', lw=1.5, ls='--', label='G_t (True Return)')
            ax_val.set_xlabel('Step')
            ax_val.set_ylabel('Value / Return')
            ax_val.set_title('Conformal value interval by step')
            ax_val.legend(loc='best')
            ax_val.grid(True, alpha=0.3)
            fig.suptitle(f'Value conformal — episode {episode_label}', fontsize=12)
            fig.tight_layout()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, figure_name)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Value conformal: saved {out_path} (quantile={cf.quantile:.4f})")

        # ----- Space conformal -----
        if do_space:
            alpha_s = float(s_cfg.get('alpha', 0.1))
            n_steps = int(s_cfg.get('n_steps', 100))
            cf_s = ConformalSpace(n_steps=n_steps).calibrate(calibration_with_start, alpha=alpha_s)
            centroid = cf_s.centroid_path()
            radius = cf_s.radius()
            trajectory_xy = [eval_start_pos] + [pos for (_, _, pos) in eval_traj if pos is not None]
            trajectory_xy = [p for p in trajectory_xy if p is not None]
            out_dir = s_cfg.get('output_dir') or sim_out
            base_name = (s_cfg.get('figure_name') or 'conformal_space_tube.png').replace('.png', '')
            figure_name = f"{base_name}_ep{episode_label}_{label_suffix}.png"
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_aspect('equal')
            ax.set_xlim(float(env_cal.L[0, 0]), float(env_cal.L[0, 1]))
            ax.set_ylim(float(env_cal.L[1, 0]), float(env_cal.L[1, 1]))
            exits_list = [[e['x'], e['y']] for e in config.get('exits', [])]
            visualization.draw_exits(ax, np.array(exits_list) if exits_list else [], domain)
            visualization.draw_obstacles(ax, obstacle_configs=obstacle_configs, domain=domain)
            if centroid is not None:
                cx, cy = centroid[:, 0], centroid[:, 1]
                ax.plot(cx, cy, 'b-', lw=1.5, alpha=0.7, label='Centroid path')
                for i in range(0, len(cx), max(1, len(cx) // 25)):
                    circle = plt.Circle((cx[i], cy[i]), radius, fill=True, facecolor=(0, 0, 1, 0.08), edgecolor='blue', linestyle='--', linewidth=0.8)
                    ax.add_patch(circle)
            if len(trajectory_xy) >= 2:
                xs, ys = [p[0] for p in trajectory_xy], [p[1] for p in trajectory_xy]
                ax.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=20, zorder=5)
                ax.plot(xs, ys, 'k-', lw=0.8, alpha=0.6, zorder=4, label='Eval trajectory')
                ax.scatter([xs[0]], [ys[0]], c='green', s=120, marker='o', label='Start', zorder=6, edgecolors='black')
                ax.scatter([xs[-1]], [ys[-1]], c='red', s=120, marker='s', label='End', zorder=6, edgecolors='black')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Space conformal tube (r={radius:.3f}, 1-α={1-alpha_s:.1f}) — ep {episode_label}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            fig.suptitle(f'Space conformal — episode {episode_label}', fontsize=12)
            fig.tight_layout()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, figure_name)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Space conformal: saved {out_path} (radius={radius:.4f})")

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
        go_find_steps_this_ep = 0
        for step in range(steps_per_episode):
            a = agent.get_action(s, deterministic=False)
            a = a + noise_std * np.random.randn(3)
            a[0] = np.clip(a[0], -1.0, 1.0)
            a[1] = np.clip(a[1], -1.0, 1.0)
            confidence = 1.0 / (1.0 + np.exp(-float(a[2])))
            used_go_find = False
            if confidence > go_find_confidence_threshold:
                go_find_steps_this_ep += 1
                dx, dy = env.get_guide_go_find_direction(min_distance=go_find_min_distance, max_distance=go_find_max_distance, stick_steps=go_find_stick_steps)
                guide_actions = [[float(dx), float(dy)]]
                used_go_find = True
            else:
                guide_actions = [[float(a[0]), float(a[1])]]
            done = env.step_guided(guide_actions=guide_actions, max_guide_speed=max_guide_speed)
            r = (exit_reward_scale * env.get_exit_reward()
                - env.get_guide_boundary_penalty(margin=guide_boundary_margin, penalty_scale=guide_boundary_penalty_scale, corner_extra_scale=guide_corner_penalty_scale)
                + env.get_guide_dense_reward(n_in_range_count_threshold=n_in_range_count_threshold,
                    reward_toward_exit_scale=reward_toward_exit_scale)
                + env.get_guide_reward_toward_crowd(reward_toward_crowd_scale=reward_toward_crowd_scale,
                    n_in_range_count_threshold=n_in_range_count_threshold,
                    go_find_alone_bonus_scale=go_find_alone_bonus_scale))
            s_next = env.get_guide_state()
            ep_reward += r

            s_next_valid = s_next if s_next is not None else s
            # When we used A* go_find: (vx,vy) didn't cause the transition → only update confidence (3rd dim) so threshold learns
            agent.update(s, a, r, s_next_valid, done=done, update_actor="confidence_only" if used_go_find else True)
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
        total_steps_ep = step + 1
        go_find_pct = 100.0 * go_find_steps_this_ep / total_steps_ep if total_steps_ep else 0.0
        print(f"Episode {ep+1}/{episodes}  total_reward={ep_reward:.2f}  steps={total_steps_ep}  Agents: {n_agents}  Guides: {n_guides}  |  go_find: {go_find_steps_this_ep}/{total_steps_ep} ({go_find_pct:.1f}%)")

        # Save checkpoint every N episodes
        if save_model and save_path and save_every_n_episodes > 0 and (ep + 1) % save_every_n_episodes == 0:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            base, ext = os.path.splitext(save_path)
            path_n = f"{base}_ep{ep + 1}{ext}"
            torch.save({"agent": agent.state_dict()}, path_n)
            print(f"Saved agent to {path_n}")

        # Conformal snapshot every N episodes
        if do_conformal and conformal_every_n > 0 and (ep + 1) % conformal_every_n == 0:
            _run_conformal_snapshot(
                agent, config, episode_label=ep + 1, label_suffix='snapshot',
                exit_reward_scale=exit_reward_scale, guide_boundary_margin=guide_boundary_margin,
                guide_boundary_penalty_scale=guide_boundary_penalty_scale, guide_corner_penalty_scale=guide_corner_penalty_scale,
                n_in_range_count_threshold=n_in_range_count_threshold, reward_toward_exit_scale=reward_toward_exit_scale,
                go_find_confidence_threshold=go_find_confidence_threshold, go_find_min_distance=go_find_min_distance,
                go_find_max_distance=go_find_max_distance, max_guide_speed=max_guide_speed,
                steps_per_episode=steps_per_episode, gamma=gamma,
                obstacle_configs=obstacle_configs, domain=domain,
            )

        # LR scheduler step (per episode)
        if scheduler_actor is not None:
            scheduler_actor.step()
        if scheduler_actor_go_find is not None:
            scheduler_actor_go_find.step()
        if scheduler_critic is not None:
            scheduler_critic.step()

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
    # Save final checkpoint (if not already saved this episode by save_every_n)
    if save_model and save_path:
        if save_every_n_episodes <= 0 or episodes % save_every_n_episodes != 0:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({"agent": agent.state_dict()}, save_path)
            print(f"Saved agent to {save_path}")

    # Conformal snapshot at end (if do_conformal and not already run at this episode)
    if do_conformal:
        run_final = (conformal_every_n <= 0) or (episodes % conformal_every_n != 0)
        if run_final:
            _run_conformal_snapshot(
                agent, config, episode_label=episodes, label_suffix='final',
                exit_reward_scale=exit_reward_scale, guide_boundary_margin=guide_boundary_margin,
                guide_boundary_penalty_scale=guide_boundary_penalty_scale, guide_corner_penalty_scale=guide_corner_penalty_scale,
                n_in_range_count_threshold=n_in_range_count_threshold, reward_toward_exit_scale=reward_toward_exit_scale,
                go_find_confidence_threshold=go_find_confidence_threshold, go_find_min_distance=go_find_min_distance,
                go_find_max_distance=go_find_max_distance, max_guide_speed=max_guide_speed,
                steps_per_episode=steps_per_episode, gamma=gamma,
                obstacle_configs=obstacle_configs, domain=domain,
            )


if __name__ == '__main__':
    main()
