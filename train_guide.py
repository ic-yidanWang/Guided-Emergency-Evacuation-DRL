"""
Train the guide agent with Actor-Critic.

Usage:
  python train_guide.py [--config config/simulation_config.json] [--no-viz]

Outputs (from this script):
  - Model checkpoints: when save_model=true, to train.save_path (and every N episodes if save_every_n_episodes > 0).
  - Conformal figures: when train.do_value_conformal, to value_conformal.output_dir;
    value figure: conformal_value_interval_ep{N}_*.png (critic return intervals only).
  those come from run_guided_visualize.py (output/guided/frames/).
"""

import argparse
import os
import time
import numpy as np
import torch

from evacuation_rl.utils.config_loader import load_config
from evacuation_rl.utils.simulation import setup_environment
from evacuation_rl.agents.actor_critic import ActorCritic
from evacuation_rl.conformal import ConformalValue
from evacuation_rl.utils import visualization

import matplotlib.pyplot as plt

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
    guide_boundary_margin = float(train_cfg.get('guide_boundary_margin', 0.8))
    use_boundary_penalty = train_cfg.get('use_boundary_penalty', True)
    use_corner_penalty = train_cfg.get('use_corner_penalty', True)
    guide_boundary_penalty_scale = float(train_cfg.get('guide_boundary_penalty_scale', 0.5)) if use_boundary_penalty else 0.0
    guide_corner_penalty_scale = float(train_cfg.get('guide_corner_penalty_scale', 0.3)) if use_corner_penalty else 0.0
    time_penalty_scale = float(train_cfg.get('time_penalty_scale', 0.01))
    memory_step_reward_scale = float(train_cfg.get('memory_step_reward_scale', 0.01))
    memory_first_reward_scale = float(train_cfg.get('memory_first_reward_scale', 0.1))
    memory_exit_reward_scale = float(train_cfg.get('memory_exit_reward_scale', 0.05))
    episodes = int(train_cfg.get('episodes', 50))
    steps_per_episode = int(train_cfg.get('steps_per_episode', 200))
    save_model = train_cfg.get('save_model', False)
    save_path = train_cfg.get('save_path', 'output/guided/guide_agent.pt')
    save_every_n_episodes = int(train_cfg.get('save_every_n_episodes', 0))
    do_value_conformal = train_cfg.get('do_value_conformal', False)
    do_conformal = do_value_conformal
    value_conformal_cfg = config.get('value_conformal', {})
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
    use_visit_pathfinding_when_alone = config.get('guide_parameters', {}).get('use_visit_pathfinding_when_alone', False)
    # Using on-policy learning: update immediately with current policy's experiences
    # This ensures reward and state transitions are always consistent with the current policy

    if not config['agents'].get('add_guide_agent', False):
        print("Config has add_guide_agent=false. Enable it to train the guide.")
        return

    env = setup_environment(config)
    state = env.get_guide_state()
    if state is None:
        print("No guide in env. Cannot train.")
        return
    state_dim = state.shape[0]
    print(f"State dim: {state_dim} (dir_cos, dir_sin, vel_cos, vel_sin, dist_to_centroid_norm, astar_cos, astar_sin, x_norm, y_norm, n_remaining_norm); critic extras: 4 (n_escaped_norm, n_first_guided_norm, memory_sum_norm, control_mode)")
    if use_visit_pathfinding_when_alone:
        print("Visit pathfinding when alone: enabled (guide uses least-visited cell + A* when no evacuees in perception)")
    print("Action: (vx, vy) continuous in [-1, 1]^2")
    print(f"Boundary penalty: {'on' if use_boundary_penalty else 'off'} (scale={guide_boundary_penalty_scale}), Corner penalty: {'on' if use_corner_penalty else 'off'} (scale={guide_corner_penalty_scale})")

    # Using Q(s, extras, a) critic for advantage calculation (on-policy learning), with a = (vx, vy)
    agent = ActorCritic(
        state_dim=state_dim,
        action_dim=2,
        critic_extra_dim=4,
        hidden_sizes=(64, 64),
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        log_std_init=log_std_init,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
    )
    # On-policy learning: no replay buffer needed
    scheduler_actor = None
    scheduler_critic = None
    if lr_scheduler_type == 'cosine' and episodes > 0:
        scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.opt_actor, T_max=episodes, eta_min=lr_actor * min_lr_ratio
        )
        scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.opt_critic, T_max=episodes, eta_min=lr_critic * min_lr_ratio
        )
    print(f"Optimizer: {optimizer_type}, weight_decay={weight_decay}, lr_scheduler={lr_scheduler_type or 'none'}")
    print("Training mode: On-policy learning with single Actor (vx, vy) and Q(s,a) critic")

    domain = {
        'x': env.L[0, 1] - env.L[0, 0],
        'y': env.L[1, 1] - env.L[1, 0],
        'z': env.L[2, 1] - env.L[2, 0],
    }
    obstacle_configs = config.get('obstacles', [])
    agent_size = config.get('agents', {}).get('agent_size', config.get('exit_parameters', {}).get('agent_size', 0.18))
    guide_size = config.get('guide_parameters', {}).get('guide_size', 0.25)
    guide_radius = config.get('guide_parameters', {}).get('guide_radius')
    perception_radius = config.get('guide_parameters', {}).get('perception_radius', 2.5)

    conformal_every_n = int(value_conformal_cfg.get('every_n_episodes', 0)) if do_value_conformal else 0

    def _run_conformal_snapshot(
        agent, config, episode_label, label_suffix,
        guide_boundary_margin, guide_boundary_penalty_scale, guide_corner_penalty_scale,
        steps_per_episode, gamma,
        obstacle_configs, domain,
        memory_step_reward_scale, memory_first_reward_scale, memory_exit_reward_scale,
    ):
        """Run conformal snapshot. Always draw in a headless Figure and save to file (no main window)."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        v_cfg = config.get('value_conformal', {})
        do_value = do_value_conformal
        if not do_value:
            return
        calibration_episodes = int(v_cfg.get('calibration_episodes', 5))
        sim_out = config.get('simulation', {}).get('output_dir', 'output/guided')
        fix_seed = v_cfg.get('fix_seed', False)
        conformal_seed = int(v_cfg.get('seed', 42))
        saved_np_state = None
        saved_torch_state = None
        if fix_seed:
            saved_np_state = np.random.get_state()
            saved_torch_state = torch.get_rng_state()
            np.random.seed(conformal_seed)
            torch.manual_seed(conformal_seed)
            print(f"  Conformal: fix_seed=True, seed={conformal_seed} (reproducible calibration & eval)")

        def _restore_rng():
            if fix_seed and saved_np_state is not None:
                np.random.set_state(saved_np_state)
                torch.set_rng_state(saved_torch_state)

        def _reward_fn(env):
            return (
                - env.get_guide_boundary_penalty(
                    margin=guide_boundary_margin,
                    penalty_scale=guide_boundary_penalty_scale,
                    corner_extra_scale=guide_corner_penalty_scale,
                )
                + env.get_time_penalty_reward(time_penalty_scale=time_penalty_scale)
                + env.get_guide_memory_reward(
                    step_scale=memory_step_reward_scale,
                    first_scale=memory_first_reward_scale,
                    exit_scale=memory_exit_reward_scale,
                )
            )

        use_visit_when_alone = config.get('guide_parameters', {}).get('use_visit_pathfinding_when_alone', False)

        def _run_episode_trajectory(env, deterministic):
            start_pos = env.get_guide_position()
            traj = []
            has_evacuee = env.has_evacuees_in_guide_perception()
            use_scripted = (not has_evacuee) and use_visit_when_alone
            s = env.get_guide_state()
            if s is None:
                return traj, start_pos
            for _ in range(steps_per_episode):
                has_evacuee = env.has_evacuees_in_guide_perception()
                use_scripted = (not has_evacuee) and use_visit_when_alone
                control_mode = 0 if use_scripted else 1
                s = env.get_guide_state()
                if s is None:
                    break
                extras = env.get_guide_critic_extras(control_mode=control_mode)
                if use_scripted:
                    dx, dy = env.get_visit_pathfinding_direction()
                    a = np.array([dx, dy], dtype=np.float32)
                    a = np.clip(a, -1.0, 1.0)
                else:
                    a = agent.get_action(s, deterministic=deterministic)
                guide_actions = [[float(a[0]), float(a[1])]]
                done = env.step_guided(guide_actions=guide_actions, max_guide_speed=max_guide_speed)
                r = _reward_fn(env)
                pos = env.get_guide_position()
                traj.append((s.copy(), extras.copy(), r, pos))
                s_next = env.get_guide_state()
                s = s_next if s_next is not None else s
                if done:
                    break
            return traj, start_pos

        try:
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
                calibration_trajectories_value = [[(s, extras, r) for s, extras, r, _ in t] for _, t in calibration_with_start]
                alpha_v = float(v_cfg.get('alpha', 0.1))
                cf = ConformalValue(agent, gamma=gamma).calibrate(calibration_trajectories_value, alpha=alpha_v)
                trajectory_xy = [eval_start_pos] + [pos for (_, _, _, pos) in eval_traj if pos is not None]
                trajectory_xy = [p for p in trajectory_xy if p is not None]
                steps, values, lowers, uppers, empirical_returns = [], [], [], [], []
                T = len(eval_traj)
                for t in range(T):
                    s, extras, r, _ = eval_traj[t]
                    v = cf.predict(s, extras)
                    lo, hi = cf.interval(s, extras)
                    G_t = sum((gamma ** (k - t)) * eval_traj[k][2] for k in range(t, T))
                    steps.append(t)
                    values.append(v)
                    lowers.append(lo)
                    uppers.append(hi)
                    empirical_returns.append(G_t)
                out_dir = v_cfg.get('output_dir') or sim_out
                base_name = (v_cfg.get('figure_name') or 'conformal_value_interval.png').replace('.png', '')
                figure_name = f"{base_name}_ep{episode_label}_{label_suffix}.png"
                fig_to_save = Figure(figsize=(14, 5))
                FigureCanvasAgg(fig_to_save)
                ax_traj = fig_to_save.add_subplot(121)
                ax_val = fig_to_save.add_subplot(122)
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
                fig_to_save.suptitle(f'Value conformal — episode {episode_label}', fontsize=12)
                fig_to_save.tight_layout()
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, figure_name)
                fig_to_save.savefig(out_path, dpi=150)
                print(f"  Value conformal: saved {out_path} (quantile={cf.quantile:.4f})")
        finally:
            _restore_rng()

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
    train_start_time = time.perf_counter()

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
            has_evacuee = env.has_evacuees_in_guide_perception()
            use_scripted = (not has_evacuee) and use_visit_pathfinding_when_alone
            control_mode = 0 if use_scripted else 1
            s = env.get_guide_state()
            if s is None:
                break
            extras = env.get_guide_critic_extras(control_mode=control_mode)
            if use_scripted:
                dx, dy = env.get_visit_pathfinding_direction()
                a = np.array([dx, dy], dtype=np.float32)
                a = np.clip(a, -1.0, 1.0)
            else:
                a = agent.get_action(s, deterministic=False)
                a = a + noise_std * np.random.randn(2)
                a[0] = np.clip(a[0], -1.0, 1.0)
                a[1] = np.clip(a[1], -1.0, 1.0)

            guide_actions = [[float(a[0]), float(a[1])]]
            done = env.step_guided(guide_actions=guide_actions, max_guide_speed=max_guide_speed)
            s_next = env.get_guide_state()
            extras_next = env.get_guide_critic_extras(control_mode=1)
            r = (
                - env.get_guide_boundary_penalty(
                    margin=guide_boundary_margin,
                    penalty_scale=guide_boundary_penalty_scale,
                    corner_extra_scale=guide_corner_penalty_scale,
                )
                + env.get_time_penalty_reward(time_penalty_scale=time_penalty_scale)
                + env.get_guide_memory_reward(
                    step_scale=memory_step_reward_scale,
                    first_scale=memory_first_reward_scale,
                    exit_scale=memory_exit_reward_scale,
                )
            )
            ep_reward += r

            s_next_valid = s_next if s_next is not None else s
            extras_next_valid = extras_next if s_next is not None else extras

            # On-policy learning: only update when we used RL (not scripted pathfinding)
            if not use_scripted:
                agent.update(s, a, r, s_next_valid, done=done, extras=extras, extras_next=extras_next_valid)

            if s_next is not None:
                s = s_next
            if done:
                break

            if do_visualize and (step % refresh_interval == 0):
                if ax_evac is not None:
                    visualization.draw_training_frame(
                        ax_evac, env, domain, obstacle_configs,
                        agent_size=agent_size, guide_size=guide_size, guide_radius=guide_radius,
                        perception_radius=perception_radius,
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
        print(f"Episode {ep+1}/{episodes}  total_reward={ep_reward:.2f}  steps={total_steps_ep}  Agents: {n_agents}  Guides: {n_guides}")

        # Save checkpoint every N episodes
        if save_model and save_path and save_every_n_episodes > 0 and (ep + 1) % save_every_n_episodes == 0:
            print("Saving the model...")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            base, ext = os.path.splitext(save_path)
            path_n = f"{base}_ep{ep + 1}{ext}"
            torch.save({"agent": agent.state_dict()}, path_n)
            print(f"Saved agent to {path_n}")

        # Conformal snapshot every N episodes
        if do_conformal and conformal_every_n > 0 and (ep + 1) % conformal_every_n == 0:
            print("Running conformal snapshot...")
            if do_visualize and ax_evac is not None and fig is not None:
                ax_evac.clear()
                ax_evac.set_xlim(0, domain['x'])
                ax_evac.set_ylim(0, domain['y'])
                ax_evac.set_aspect('equal')
                ax_evac.text(0.5, 0.5, 'Evaluating Conformal Prediction...\nPlease Wait',
                             transform=ax_evac.transAxes, fontsize=14, ha='center', va='center')
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            _run_conformal_snapshot(
                agent, config, episode_label=ep + 1, label_suffix='snapshot',
                guide_boundary_margin=guide_boundary_margin,
                guide_boundary_penalty_scale=guide_boundary_penalty_scale, guide_corner_penalty_scale=guide_corner_penalty_scale,
                steps_per_episode=steps_per_episode, gamma=gamma,
                obstacle_configs=obstacle_configs, domain=domain,
                memory_step_reward_scale=memory_step_reward_scale,
                memory_first_reward_scale=memory_first_reward_scale,
                memory_exit_reward_scale=memory_exit_reward_scale,
            )

        # LR scheduler step (per episode)
        if scheduler_actor is not None:
            scheduler_actor.step()
        if scheduler_critic is not None:
            scheduler_critic.step()

    if do_visualize and fig is not None:
        
        total_elapsed = time.perf_counter() - train_start_time
        if ax_evac is not None:
            ax_evac.clear()
            ax_evac.set_xlim(0, domain['x'])
            ax_evac.set_ylim(0, domain['y'])
            ax_evac.set_aspect('equal')
            if total_elapsed >= 3600:
                time_str = f'{total_elapsed / 3600:.2f} h'
            elif total_elapsed >= 60:
                time_str = f'{total_elapsed / 60:.2f} min'
            else:
                time_str = f'{total_elapsed:.2f} s'
            ax_evac.text(0.5, 0.5, f'Training finished\nTotal time: {time_str}',
                         transform=ax_evac.transAxes, fontsize=14, ha='center', va='center')
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        if ax_reward is not None and total_reward_per_episode:
            visualization.draw_reward_curve(ax_reward, total_reward_per_episode, max_episodes=episodes, fig=fig)
        plt.ioff()
        # Only block on show if the window is still open (user may have closed it)
        if fig.number in plt.get_fignums():
            plt.show(block=True)
        else:
            plt.close(fig)

    print(f"\nTraining finished. Time taken: {time_str:.2f}.")

    
    if total_reward_per_episode:
        n = min(10, len(total_reward_per_episode))
        print(f"Mean episode reward (last {n}): {sum(total_reward_per_episode[-n:]) / n:.2f}")
    # Save final checkpoint (if not already saved this episode by save_every_n)
    if save_model and save_path:
        if save_every_n_episodes <= 0 or episodes % save_every_n_episodes != 0:
            print("Saving the lastmodel...")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({"agent": agent.state_dict()}, save_path)
            print(f"Saved agent to {save_path}")

    # Conformal snapshot at end (if do_conformal and not already run at this episode)
    if do_conformal:
        run_final = (conformal_every_n <= 0) or (episodes % conformal_every_n != 0)
        if run_final:
            print("Running final conformal snapshot...")
            _run_conformal_snapshot(
                agent, config, episode_label=episodes, label_suffix='final',
                guide_boundary_margin=guide_boundary_margin,
                guide_boundary_penalty_scale=guide_boundary_penalty_scale, guide_corner_penalty_scale=guide_corner_penalty_scale,
                steps_per_episode=steps_per_episode, gamma=gamma,
                obstacle_configs=obstacle_configs, domain=domain,
                memory_step_reward_scale=memory_step_reward_scale,
                memory_first_reward_scale=memory_first_reward_scale,
                memory_exit_reward_scale=memory_exit_reward_scale,
            )


if __name__ == '__main__':
    main()
