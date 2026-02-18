"""
Visualization utilities for evacuation simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import os


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


def create_animation_from_configs(config_dir, output_file='evacuation.gif', fps=10, domain=None, obstacle_configs=None):
    """
    Create animation from saved configuration files
    
    Args:
        config_dir: Directory containing configuration files
        output_file: Output animation file path
        fps: Frames per second
        domain: Optional domain dict with 'x', 'y', 'z' dimensions
        obstacle_configs: List of obstacle configs with type info (for proper visualization)
    """
    import glob
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle
    
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
        
        # Plot exits (yellow stars)
        if exits:
            exits_arr = np.array(exits)
            ax.scatter(exits_arr[:, 0] * domain['x'], exits_arr[:, 1] * domain['y'], 
                      c='yellow', marker='*', s=500, edgecolors='black', 
                      linewidths=2, label='Exits', zorder=5)
        
        # Plot guides (red circles)
        if guides:
            guides_arr = np.array(guides)
            ax.scatter(guides_arr[:, 0] * domain['x'], guides_arr[:, 1] * domain['y'], 
                      c='red', marker='o', s=300, edgecolors='darkred', 
                      linewidths=2, label='Guides', zorder=4)
        
        # Plot obstacles with proper type support
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
                                   facecolor='gray', alpha=0.4, zorder=3)
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
                                    facecolor='gray', alpha=0.4, zorder=3)
                    ax.add_patch(rect)
        elif obstacles:
            # Fallback to plotting raw obstacle points if no configs provided
            obstacles_arr = np.array(obstacles)
            ax.scatter(obstacles_arr[:, 0] * domain['x'], obstacles_arr[:, 1] * domain['y'], 
                      c='black', marker='s', s=200, label='Obstacles', zorder=3)
        
        # Plot guide agents (yellow circles - moving guides)
        if guide_agents:
            guide_agents_arr = np.array(guide_agents)
            ax.scatter(guide_agents_arr[:, 0] * domain['x'], guide_agents_arr[:, 1] * domain['y'], 
                      c='gold', marker='o', s=100, edgecolors='orange', 
                      linewidths=2, label='Guide Agents', zorder=4)
        
        # Plot agents (blue circles)
        if agents:
            agents_arr = np.array(agents)
            ax.scatter(agents_arr[:, 0] * domain['x'], agents_arr[:, 1] * domain['y'], 
                      c='blue', marker='o', s=50, alpha=0.6, label='Agents', zorder=2)
        
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
    from matplotlib.patches import Rectangle, Circle
    
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
    
    # Plot exits
    for e in exits:
        ax.plot(e[0], e[1], 'y*', markersize=20, markeredgecolor='k', markeredgewidth=1.5)
    
    # Plot obstacles with type support
    if obstacle_configs:
        for obs in obstacle_configs:
            obs_type = obs.get('type', 'circle')
            
            if obs_type == 'circle':
                # Draw circle obstacle
                # Coordinates are already absolute, no scaling needed
                center_x = obs['x']
                center_y = obs['y']
                radius = obs.get('size', 0.5)  # Size is in absolute units
                
                circle = Circle((center_x, center_y), radius, 
                               linewidth=2, edgecolor='black', 
                               facecolor='gray', alpha=0.5, label='Obstacle' if obs == obstacle_configs[0] else '')
                ax.add_patch(circle)
            
            elif obs_type == 'rectangle':
                # Draw rectangle obstacle
                center_x = obs['x']
                center_y = obs['y']
                width = obs.get('width', 0.4)
                height = obs.get('height', 0.3)
                
                rect_x = center_x - width / 2
                rect_y = center_y - height / 2
                
                rect = Rectangle((rect_x, rect_y), width, height, 
                                linewidth=2, edgecolor='black', 
                                facecolor='gray', alpha=0.5, label='Obstacle' if obs == obstacle_configs[0] else '')
                ax.add_patch(rect)
    elif obstacles:
        # Fallback to plotting raw obstacle points
        for ob in obstacles:
            ax.plot(ob[0], ob[1], 'ks', markersize=15)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Evacuation Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax
