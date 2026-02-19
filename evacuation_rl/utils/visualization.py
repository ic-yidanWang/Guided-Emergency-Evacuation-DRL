"""
Visualization utilities for evacuation simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle


def draw_exits(ax, exits, domain, label='Exits', zorder=5):
    """
    Draw exit points on the axes
    
    Args:
        ax: Matplotlib axes object
        exits: List of exit positions (normalized [0,1] or absolute coordinates)
        domain: Domain boundaries dict with 'x', 'y', 'z' keys
        label: Label for legend
        zorder: Z-order for rendering (higher values appear on top)
    """
    exits = np.array(exits) if exits is not None else np.empty((0, 2))
    if exits.size == 0:
        return
    if exits.ndim == 1:
        exits = np.reshape(exits, (1, -1))
    # Check if exits are normalized (all values <= 1) or absolute
    if np.all(exits[:, :2] <= 1.0):
        # Normalized coordinates, convert to absolute
        ax.scatter(exits[:, 0] * domain['x'], exits[:, 1] * domain['y'],
                  c='yellow', marker='*', s=500, edgecolors='black',
                  linewidths=2, label=label, zorder=zorder)
    else:
        # Already absolute coordinates
        ax.scatter(exits[:, 0], exits[:, 1],
                  c='yellow', marker='*', s=500, edgecolors='black',
                  linewidths=2, label=label, zorder=zorder)


def draw_guides(ax, guides, domain, label='Guides', zorder=4):
    """
    Draw guide points (static red circles) on the axes
    
    Args:
        ax: Matplotlib axes object
        guides: List of guide positions (normalized [0,1] or absolute coordinates)
        domain: Domain boundaries dict with 'x', 'y', 'z' keys
        label: Label for legend
        zorder: Z-order for rendering
    """
    if guides:
        guides = np.array(guides)
        # Check if guides are normalized (all values <= 1) or absolute
        if guides.size > 0 and np.all(guides[:, :2] <= 1.0):
            # Normalized coordinates, convert to absolute
            ax.scatter(guides[:, 0] * domain['x'], guides[:, 1] * domain['y'], 
                      c='red', marker='o', s=300, edgecolors='darkred', 
                      linewidths=2, label=label, zorder=zorder)
        else:
            # Already absolute coordinates
            ax.scatter(guides[:, 0], guides[:, 1], 
                      c='red', marker='o', s=300, edgecolors='darkred', 
                      linewidths=2, label=label, zorder=zorder)


def draw_obstacles(ax, obstacle_configs=None, obstacles=None, domain=None, label='Obstacles', zorder=3):
    """
    Draw obstacles on the axes
    
    Args:
        ax: Matplotlib axes object
        obstacle_configs: List of obstacle configs with type info (circle/rectangle) in absolute coordinates
        obstacles: Optional list of raw obstacle positions (fallback, normalized [0,1])
        domain: Domain boundaries dict with 'x', 'y', 'z' keys (required if obstacles provided)
        label: Label for legend
        zorder: Z-order for rendering
    """
    if obstacle_configs:
        # Use original obstacle configs to draw rectangles/circles
        # Note: obstacle_configs are in ABSOLUTE coordinates from config file
        for obs in obstacle_configs:
            obs_type = obs.get('type', 'circle')
            
            if obs_type == 'circle':
                # Draw circle obstacle with proper radius
                # Coordinates are already absolute, no scaling needed
                center_x = obs['x']
                center_y = obs['y']
                radius = obs.get('size', 0.5)  # Size is in absolute units
                
                circle = Circle((center_x, center_y), radius, 
                               linewidth=2, edgecolor='black', 
                               facecolor='gray', alpha=0.4, zorder=zorder)
                ax.add_patch(circle)
            
            elif obs_type == 'rectangle':
                center_x = obs['x']
                center_y = obs['y']
                width = obs.get('width', 0.4)
                height = obs.get('height', 0.3)
                
                rect_x = center_x - width / 2
                rect_y = center_y - height / 2
                
                rect = Rectangle((rect_x, rect_y), width, height, 
                                linewidth=2, edgecolor='black', 
                                facecolor='gray', alpha=0.4, zorder=zorder)
                ax.add_patch(rect)
    
    elif obstacles and domain:
        # Fallback to plotting raw obstacle points if no configs provided
        obstacles_arr = np.array(obstacles)
        # Check if obstacles are normalized (all values <= 1) or absolute
        if obstacles_arr.size > 0 and np.all(obstacles_arr[:, :2] <= 1.0):
            # Normalized coordinates, convert to absolute
            ax.scatter(obstacles_arr[:, 0] * domain['x'], obstacles_arr[:, 1] * domain['y'], 
                      c='black', marker='s', s=200, label=label, zorder=zorder)
        else:
            # Already absolute coordinates
            ax.scatter(obstacles_arr[:, 0], obstacles_arr[:, 1], 
                      c='black', marker='s', s=200, label=label, zorder=zorder)


# Scale factor for drawing particle radii (smaller = less overlap, still proportional to config size)
VIS_RADIUS_SCALE = 0.5


def draw_agents(ax, agents, domain, label='Agents', zorder=2, alpha=0.6, agent_size=0.18):
    """
    Draw regular agents (blue circles). Radius = agent_size * VIS_RADIUS_SCALE to avoid overlap.
    
    Args:
        ax: Matplotlib axes object
        agents: List of agent positions (normalized [0,1] or absolute coordinates)
        domain: Domain boundaries dict with 'x', 'y', 'z' keys
        label: Label for legend
        zorder: Z-order for rendering
        alpha: Transparency level (0-1)
        agent_size: Radius of agent circles in world units (default 0.18); scales with config.
    """
    if agents:
        agents = np.array(agents)
        if agents.size > 0 and np.all(agents[:, :2] <= 1.0):
            xy = agents[:, :2] * np.array([domain['x'], domain['y']])
        else:
            xy = agents[:, :2]
        r = agent_size * VIS_RADIUS_SCALE
        for i in range(len(xy)):
            circle = Circle((xy[i, 0], xy[i, 1]), r,
                            facecolor='blue', edgecolor='darkblue', alpha=alpha,
                            linewidth=1, zorder=zorder, label=label if i == 0 else '')
            ax.add_patch(circle)


def draw_guide_agents(ax, guide_agents, domain, label='Guide Agents', zorder=4, guide_size=0.25, guide_radius=None):
    """
    Draw guide agents (yellow/gold circles) and optional dashed influence circle.
    
    Args:
        ax: Matplotlib axes object
        guide_agents: List of guide agent positions (normalized [0,1] or absolute coordinates)
        domain: Domain boundaries dict with 'x', 'y', 'z' keys
        label: Label for legend
        zorder: Z-order for rendering
        guide_size: Radius of guide circles in world units (default 0.25); typically larger than agents.
        guide_radius: If set, draw a dashed circle around each guide with this radius (influence zone).
    """
    if guide_agents:
        guide_agents = np.array(guide_agents)
        if guide_agents.size > 0 and np.all(guide_agents[:, :2] <= 1.0):
            xy = guide_agents[:, :2] * np.array([domain['x'], domain['y']])
        else:
            xy = guide_agents[:, :2]
        # Draw influence circle (dashed) behind the guide dot
        if guide_radius is not None and guide_radius > 0:
            for i in range(len(xy)):
                ring = Circle((xy[i, 0], xy[i, 1]), guide_radius,
                              facecolor='none', edgecolor='orange', linestyle='--',
                              linewidth=1.2, alpha=0.8, zorder=zorder - 1)
                ax.add_patch(ring)
        # Draw guide dot
        r = guide_size * VIS_RADIUS_SCALE
        for i in range(len(xy)):
            circle = Circle((xy[i, 0], xy[i, 1]), r,
                            facecolor='gold', edgecolor='orange', linewidth=2,
                            zorder=zorder, label=label if i == 0 else '')
            ax.add_patch(circle)


def visualize_policy(model, device, offset=np.array([0.5, 0.5])):
    """
    Visualize the learned evacuation policy
    
    Args:
        model: Trained DQN model
        device: PyTorch device (CPU or CUDA)
        offset: Position offset for normalization
    """
    # Create grid for visualization
    x, y = np.meshgrid(np.linspace(0, 1, 100) - offset[0], 
                       np.linspace(0, 1, 100) - offset[1])
    x_arrow, y_arrow = np.meshgrid(np.linspace(0.05, 0.95, 15) - offset[0], 
                                     np.linspace(0.05, 0.95, 15) - offset[1])
    xy = np.vstack([x.ravel(), y.ravel()]).T
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T
    
    # Create velocity components (zero for static visualization)
    vxy = np.zeros_like(xy)
    vxy_arrow = np.zeros_like(xy_arrow)
    
    xtest = np.hstack([xy, vxy])
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])
    
    # Predict actions
    import torch
    with torch.no_grad():
        xtest_tensor = torch.FloatTensor(xtest).to(device)
        x_arrow_test_tensor = torch.FloatTensor(x_arrow_test).to(device)
        
        ypred = model(xtest_tensor).cpu().numpy()
        ypred_arrow = model(x_arrow_test_tensor).cpu().numpy()
    
    action_pred = np.argmax(ypred, axis=1)
    action_arrow_pred = np.argmax(ypred_arrow, axis=1)
    
    action_grid = action_pred.reshape(x.shape)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), 
                           subplot_kw={'xlim': (-0.5, 0.5), 'ylim': (-0.5, 0.5)})
    
    # Contour plot of actions
    contour = ax.contourf(x, y, action_grid + 0.1, 
                          cmap=plt.cm.get_cmap('rainbow'), alpha=0.8)
    
    # Arrow visualization
    arrow_len = 0.07
    angle = np.sqrt(2) / 2
    arrow_map = {
        0: [0, arrow_len],                              # Up
        1: [-angle * arrow_len, angle * arrow_len],     # Up-Left
        2: [-arrow_len, 0],                             # Left
        3: [-angle * arrow_len, -angle * arrow_len],    # Down-Left
        4: [0, -arrow_len],                             # Down
        5: [angle * arrow_len, -angle * arrow_len],     # Down-Right
        6: [arrow_len, 0],                              # Right
        7: [angle * arrow_len, angle * arrow_len]       # Up-Right
    }
    
    for idx, p in enumerate(xy_arrow):
        ax.annotate('', xy=p, xytext=np.array(arrow_map[action_arrow_pred[idx]]) + p,
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))
    
    ax.set_xlabel('X Position (normalized)', fontsize=12)
    ax.set_ylabel('Y Position (normalized)', fontsize=12)
    ax.set_title('Learned Evacuation Policy', fontsize=14)
    ax.tick_params(labelsize='large')
    
    plt.colorbar(contour, ax=ax, label='Action ID')
    plt.tight_layout()
    
    return fig, ax


def plot_training_stats(episodes, losses, steps, save_path=None):
    """
    Plot training statistics
    
    Args:
        episodes: List of episode numbers
        losses: List of loss values
        steps: List of steps per episode
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot loss
    ax1.plot(episodes, losses)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Episodes')
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(episodes, steps)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Exit')
    ax2.set_title('Steps per Episode')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def create_animation_from_configs(config_dir, output_file='evacuation.gif', fps=10, domain=None, obstacle_configs=None, agent_size=None, guide_size=None, guide_radius=None):
    """
    Create animation from saved configuration files.
    Circle sizes should be passed from config (real simulation values), not hardcoded.

    Args:
        config_dir: Directory containing configuration files
        output_file: Output animation file path
        fps: Frames per second
        domain: Optional domain dict with 'x', 'y', 'z' dimensions
        obstacle_configs: List of obstacle configs with type info (for proper visualization)
        agent_size: Radius of agent circles in world units (from config exit_parameters.agent_size).
        guide_size: Radius of guide agent circles in world units (from config guide_parameters.guide_size).
        guide_radius: Radius of dashed influence circle around each guide (from config guide_parameters.guide_radius).
    """
    # Use passed config values; fallback only when caller does not pass (e.g. legacy call)
    if agent_size is None:
        agent_size = 0.18
    if guide_size is None:
        guide_size = 0.25
    import glob
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Find all config files
    config_files = sorted(glob.glob(os.path.join(config_dir, 's.*')), 
                         key=lambda x: int(os.path.basename(x).split('.')[1]) if '.' in os.path.basename(x) and os.path.basename(x).split('.')[1].isdigit() else 0)
    
    if not config_files:
        print(f"No configuration files found in {config_dir}")
        return None
    
    print(f"Found {len(config_files)} frames to animate")
    
    # Parse all frames
    all_frames = []
    for config_file in config_files:
        exits, guides, obstacles, agents, guide_agents, frame_domain = parse_config_file(config_file)
        if domain is None:
            domain = frame_domain
        all_frames.append((exits, guides, obstacles, agents, guide_agents))
    
    if not domain:
        domain = {'x': 10.0, 'y': 10.0, 'z': 2.0}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update_frame(frame_idx):
        ax.clear()
        exits, guides, obstacles, agents, guide_agents = all_frames[frame_idx]
        
        # Set domain limits
        ax.set_xlim(0, domain['x'])
        ax.set_ylim(0, domain['y'])
        ax.set_aspect('equal')
        
        # Draw all elements using dedicated functions
        draw_exits(ax, exits, domain)
        draw_guides(ax, guides, domain)
        draw_obstacles(ax, obstacle_configs=obstacle_configs, obstacles=obstacles, domain=domain)
        draw_guide_agents(ax, guide_agents, domain, guide_size=guide_size, guide_radius=guide_radius)
        draw_agents(ax, agents, domain, agent_size=agent_size)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Labels and title
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(f'Guided Evacuation - Frame {frame_idx}/{len(all_frames)-1}', fontsize=14)
        
        # Legend (show on every frame)
        ax.legend(loc='upper right', fontsize=10)
        
        # Info text
        remaining_agents = len(agents) if agents else 0
        remaining_guides = len(guide_agents) if guide_agents else 0
        info_text = f"Frame: {frame_idx}\nGuide Agents: {remaining_guides}\nAgents: {remaining_agents}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create animation
    anim = FuncAnimation(fig, update_frame, frames=len(all_frames), 
                        interval=1000/fps, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to: {output_file}")
    
    plt.close(fig)
    return anim


def parse_config_file(filepath):
    """
    Parse configuration file and extract particle information
    
    Returns:
        exits: List of exit positions
        guides: List of guide positions (static red points)
        obstacles: List of obstacle positions
        agents: List of agent positions (normal blue agents)
        guide_agents: List of guide agent positions (moving yellow guides)
        domain: Domain boundaries
    """
    exits = []
    guides = []
    obstacles = []
    agents = []
    guide_agents = []
    domain = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header to get domain size
    for i, line in enumerate(lines):
        if 'H0(1,1)' in line:
            hx = float(line.split('=')[1].strip().split()[0])
        elif 'H0(2,2)' in line:
            hy = float(line.split('=')[1].strip().split()[0])
        elif 'H0(3,3)' in line:
            hz = float(line.split('=')[1].strip().split()[0])
            domain = {'x': hx, 'y': hy, 'z': hz}
            break
    
    # Parse particle data
    current_type = None
    for line in lines:
        line = line.strip()
        
        # Detect particle types
        if line == 'At':
            current_type = 'exit'
        elif line == 'Fe':
            current_type = 'guide'
        elif line == 'C':
            current_type = 'obstacle_c'
        elif line == 'Si':
            current_type = 'obstacle_si'
        elif line == 'S':
            current_type = 'guide_agent'
        elif line == 'Br':
            current_type = 'agent'
        elif current_type and len(line.split()) >= 7:
            # Parse position line - only if we have a valid current type
            parts = line.split()
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                
                # Only add to appropriate list
                if current_type == 'exit':
                    exits.append([x, y, z])
                elif current_type == 'guide':
                    guides.append([x, y, z])
                elif current_type in ['obstacle_c', 'obstacle_si']:
                    obstacles.append([x, y, z])
                elif current_type == 'guide_agent':
                    guide_agents.append([x, y, z])
                elif current_type == 'agent':
                    agents.append([x, y, z])
            except (ValueError, IndexError):
                # Reset type if parsing fails (likely end of this particle type)
                current_type = None
                continue
    
    return exits, guides, obstacles, agents, guide_agents, domain


def draw_training_frame(ax, env, domain, obstacle_configs, agent_size=0.18, guide_size=0.25, guide_radius=None,
                       episode=None, total_episodes=None, step=None, ep_reward=0.0, fig=None):
    """
    Draw a single frame for real-time training visualization from env state.
    env must have get_all_positions_for_vis(), Exit, L.
    Optionally show episode, step, remaining agents/guides, and episode reward in a text box.
    If fig is provided, use fig.canvas to refresh without raising the window (avoids steal-focus).
    """
    agents_xy, guide_agents_xy = env.get_all_positions_for_vis()
    n_agents = len(agents_xy)
    n_guides = len(guide_agents_xy)
    exits = [e.tolist() if hasattr(e, 'tolist') else list(e) for e in env.Exit]
    ax.clear()
    ax.set_xlim(0, domain['x'])
    ax.set_ylim(0, domain['y'])
    ax.set_aspect('equal')
    draw_exits(ax, exits, domain)
    draw_obstacles(ax, obstacle_configs=obstacle_configs, domain=domain)
    draw_guide_agents(ax, guide_agents_xy, domain, guide_size=guide_size, guide_radius=guide_radius)
    draw_agents(ax, agents_xy, domain, agent_size=agent_size)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    # Info box: episode, step, remaining agents/guides, episode reward
    info_lines = []
    if episode is not None and total_episodes is not None:
        info_lines.append(f"Episode: {episode}/{total_episodes}")
    if step is not None:
        info_lines.append(f"Step: {step}")
    info_lines.append(f"Agents (remaining): {n_agents}")
    info_lines.append(f"Guides: {n_guides}")
    info_lines.append(f"Episode reward: {ep_reward:.2f}")
    info_text = "\n".join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(loc='upper right', fontsize=8)
    if fig is not None:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    else:
        plt.draw()
        plt.pause(0.001)


def draw_reward_curve(ax, total_reward_per_episode, max_episodes=None, fig=None):
    """Update the reward curve axes with episode rewards (real-time). Initializes title/labels/axis range from the start.
    If fig is provided, use fig.canvas to refresh without raising the window (avoids steal-focus)."""
    ax.clear()
    if max_episodes is None and total_reward_per_episode:
        max_episodes = len(total_reward_per_episode)
    if max_episodes is None:
        max_episodes = 50
    ax.set_xlim(0, max(1, max_episodes))
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Total reward', fontsize=10)
    ax.set_title('Episode reward', fontsize=12)
    ax.grid(True, alpha=0.3)
    if not total_reward_per_episode:
        ax.set_ylim(-50, 0)  # typical reward scale
        if fig is not None:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        else:
            plt.draw()
            plt.pause(0.001)
        return
    episodes = list(range(1, len(total_reward_per_episode) + 1))
    ax.plot(episodes, total_reward_per_episode, 'b-', linewidth=1.5, label='Reward')
    if total_reward_per_episode:
        lo, hi = min(total_reward_per_episode), max(total_reward_per_episode)
        margin = max(1, (hi - lo) * 0.1) if hi > lo else 1
        ax.set_ylim(lo - margin, hi + margin)
    ax.legend(loc='upper right', fontsize=8)
    if fig is not None:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    else:
        plt.draw()
        plt.pause(0.001)


def visualize_evacuation(exits, guides, obstacle_configs, agents, domain, save_path=None, agent_size=0.18):
    """
    Visualize the evacuation scenario
    
    Args:
        exits: List of exit positions (normalized [0,1])
        guides: List of guide positions (normalized [0,1])
        obstacle_configs: List of ORIGINAL obstacle configs (absolute coordinates)
        agents: List of agent positions (normalized [0,1])
        domain: Domain boundaries
        save_path: Optional path to save the figure
        agent_size: Radius of agent circles in world units (default 0.18).
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set domain limits
    ax.set_xlim(0, domain['x'])
    ax.set_ylim(0, domain['y'])
    ax.set_aspect('equal')
    
    # Draw all elements using dedicated functions
    draw_exits(ax, exits, domain)
    draw_guides(ax, guides, domain)
    draw_obstacles(ax, obstacle_configs=obstacle_configs, domain=domain)
    draw_agents(ax, agents, domain, agent_size=agent_size)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('X Position', fontsize=14)
    ax.set_ylabel('Y Position', fontsize=14)
    ax.set_title('Guided Evacuation Scenario\nRed points are guide locations', fontsize=16, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Add info text
    info_text = f"Exits: {len(exits)}\nGuides: {len(guides)}\nAgents: {len(agents)}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, ax


def plot_evacuation_trajectory(positions, exits, obstacles=None, obstacle_configs=None, domain=None, save_path=None):
    """
    Plot evacuation trajectories
    
    Args:
        positions: List of position arrays for each time step
        exits: Array of exit positions
        obstacles: Optional list of obstacle positions (legacy support)
        obstacle_configs: Optional list of obstacle configs with type info (circle/rectangle)
        domain: Optional domain dict with 'x', 'y', 'z' dimensions
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Determine domain size
    if domain is None:
        domain = {'x': 10.0, 'y': 10.0, 'z': 2.0}
    
    # Plot trajectories
    positions = np.array(positions)
    for i in range(positions.shape[1]):
        ax.plot(positions[:, i, 0], positions[:, i, 1], 'b-', alpha=0.5, linewidth=0.5)
        ax.plot(positions[0, i, 0], positions[0, i, 1], 'go', markersize=5, label='Start' if i == 0 else '')
        ax.plot(positions[-1, i, 0], positions[-1, i, 1], 'ro', markersize=5, label='End' if i == 0 else '')
    
    # Draw exits and obstacles using dedicated functions
    # Note: exits in trajectory plot are already absolute coordinates
    if exits:
        exits_list = [[e[0], e[1], e[2]] if len(e) > 2 else [e[0], e[1], 0] for e in exits]
        draw_exits(ax, exits_list, domain)
    
    # Draw obstacles using dedicated function
    draw_obstacles(ax, obstacle_configs=obstacle_configs, obstacles=obstacles, domain=domain)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Evacuation Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax
