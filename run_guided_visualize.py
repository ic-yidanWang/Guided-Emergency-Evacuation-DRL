"""
Visualize Guided Evacuation with Guide Points

This script creates a guided evacuation scenario with:
- 4 randomly positioned guide points (shown in red)
- Multiple exits
- Agents
- Animated GIF generation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import json
import argparse

# Import the guided environment
from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace
from evacuation_rl.environments import cellspace
from evacuation_rl.utils.visualization import create_animation_from_configs, parse_config_file


def load_config(config_path="config/simulation_config.json"):
    """
    Load simulation configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from: {config_path}")
    if 'description' in config:
        print(f"  - {config['description']}")
    
    return config


def run_simulation(env, num_steps=200, save_interval=5, output_dir="output/guided/frames"):
    """
    Run evacuation simulation and save configuration files
    
    Args:
        env: GuidedCellSpace environment
        num_steps: Number of simulation steps
        save_interval: Save configuration every N steps
        output_dir: Directory to save configuration files (frames will be saved here)
    
    Returns:
        List of saved configuration file paths
    """
    print(f"\nRunning simulation for {num_steps} steps...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old frame files from previous runs
    import glob
    old_files = glob.glob(os.path.join(output_dir, 's.*'))
    for old_file in old_files:
        try:
            os.remove(old_file)
        except:
            pass
    print(f"Cleaned up {len(old_files)} old frame files")
    
    saved_files = []
    step = 0
    
    while step < num_steps and env.Number > 0:
        # Save configuration at intervals
        if step % save_interval == 0:
            config_file = os.path.join(output_dir, f"s.{step}")
            env.save_output(config_file)
            saved_files.append(config_file)
            print(f"  Step {step:4d}: {env.Number:3d} agents remaining", end='\r')
        
        # Use the environment's step_guided method for proper physics, KNN behavior, and guide logic
        done = env.step_guided()
        
        step += 1
        
        if done:
            break
    
    # Save final state
    if (step - 1) % save_interval != 0 and env.Number >= 0:
        config_file = os.path.join(output_dir, f"s.{step-1}")
        env.save_output(config_file)
        saved_files.append(config_file)
    
    print(f"\n  Simulation complete: {len(saved_files)} frames saved")
    return saved_files


def visualize_evacuation(exits, guides, obstacle_configs, agents, domain, save_path=None):
    """
    Visualize the evacuation scenario
    
    Args:
        exits: List of exit positions (normalized [0,1])
        guides: List of guide positions (normalized [0,1])
        obstacle_configs: List of ORIGINAL obstacle configs (absolute coordinates)
        agents: List of agent positions (normalized [0,1])
        domain: Domain boundaries
        save_path: Optional path to save the figure
    """
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set domain limits
    ax.set_xlim(0, domain['x'])
    ax.set_ylim(0, domain['y'])
    ax.set_aspect('equal')
    
    # Plot exits (yellow stars) - multiply by domain to convert from [0,1] to absolute
    if exits:
        exits = np.array(exits)
        ax.scatter(exits[:, 0] * domain['x'], exits[:, 1] * domain['y'], 
                  c='yellow', marker='*', s=500, edgecolors='black', 
                  linewidths=2, label='Exits', zorder=5)
    
    # Plot guides (red circles) - multiply by domain to convert from [0,1] to absolute
    if guides:
        guides = np.array(guides)
        ax.scatter(guides[:, 0] * domain['x'], guides[:, 1] * domain['y'], 
                  c='red', marker='o', s=300, edgecolors='darkred', 
                  linewidths=2, label='Guides', zorder=4)
    
    # Plot obstacles based on their type (absolute coordinates from config file)
    if obstacle_configs:
        from matplotlib.patches import Circle
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
                # Rectangle obstacle - absolute coordinates from config file
                center_x = obs['x']
                center_y = obs['y']
                width = obs.get('width', 0.4)
                height = obs.get('height', 0.3)
                
                # Calculate bottom-left corner for Rectangle
                rect_x = center_x - width / 2
                rect_y = center_y - height / 2
                
                # Create and add rectangle
                rect = Rectangle((rect_x, rect_y), width, height, 
                                linewidth=2, edgecolor='black', 
                                facecolor='gray', alpha=0.4, zorder=3)
                ax.add_patch(rect)
    
    # Plot agents (blue circles) - multiply by domain to convert from [0,1] to absolute
    if agents:
        agents = np.array(agents)
        ax.scatter(agents[:, 0] * domain['x'], agents[:, 1] * domain['y'], 
                  c='blue', marker='o', s=50, alpha=0.6, label='Agents', zorder=2)
    
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


def main(config_path="config/simulation_config.json"):
    """Main function to create and visualize guided evacuation scenario"""
    print("=" * 80)
    print("Creating Guided Evacuation Scenario with Guide Points")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Apply exit parameters from config to cellspace module
    if 'exit_parameters' in config:
        cellspace.door_size = config['exit_parameters'].get('door_size', 1.0)
        cellspace.agent_size = config['exit_parameters'].get('agent_size', 0.5)
        # BUGFIX: dis_lim must account for agent size! Agents can't get their center to exit center
        # if they're wider than the door. The equilibrium distance is roughly agent_size + door_size/2
        cellspace.dis_lim = cellspace.agent_size + cellspace.door_size
        print(f"\nExit parameters:")
        print(f"  - Agent size: {cellspace.agent_size}")
        print(f"  - Door size: {cellspace.door_size}")
        print(f"  - Evacuation distance threshold (dis_lim): {cellspace.dis_lim}")
        print(f"    NOTE: dis_lim = agent_size + door_size to account for agent body width")
    
    # Configure exits from config file
    # Exists are stored as normalized [0,1] coordinates
    # The Cell_Space will convert them to actual domain coordinates
    cellspace.Exit = [
        np.array([exit_cfg['x'], exit_cfg['y'], exit_cfg['z']]) 
        for exit_cfg in config['exits']
    ]
    
    # Configure obstacles from config file (OPTIMIZED - no point grid generation!)
    cellspace.Ob = []
    cellspace.Ob_size = []
    cellspace.Ob_type = []
    cellspace.Ob_params = []
    
    for obstacle in config.get('obstacles', []):
        obstacle_type = obstacle.get('type', 'circle')
        
        if obstacle_type == 'circle':
            # Circle obstacle - single center point
            cellspace.Ob.append([np.array([obstacle['x'], obstacle['y'], obstacle['z']])])
            # Use actual radius size for collision detection
            actual_radius = obstacle.get('size', 0.5) * 0.8  # 80% of visual size for collision
            cellspace.Ob_size.append(actual_radius)
            cellspace.Ob_type.append('circle')
            cellspace.Ob_params.append({})
        
        elif obstacle_type == 'rectangle':
            # Rectangle obstacle - OPTIMIZED: store parameters directly instead of point grid!
            center_x = obstacle['x']
            center_y = obstacle['y']
            center_z = obstacle['z']
            width = obstacle.get('width', 0.4)
            height = obstacle.get('height', 0.3)
            
            # Store single center point and geometry parameters
            cellspace.Ob.append([np.array([center_x, center_y, center_z])])
            cellspace.Ob_size.append(0.05)  # Collision margin
            cellspace.Ob_type.append('rectangle')
            cellspace.Ob_params.append({
                'center': np.array([center_x, center_y, center_z]),
                'width': width,
                'height': height
            })
    
    # Clear any existing guides
    cellspace.Guide = []
    
    # Create guided environment from config
    print("\nInitializing environment...")
    env = GuidedCellSpace(
        xmin=config['domain']['xmin'],
        xmax=config['domain']['xmax'],
        ymin=config['domain']['ymin'],
        ymax=config['domain']['ymax'],
        zmin=config['domain']['zmin'],
        zmax=config['domain']['zmax'],
        rcut=config['physics']['rcut'],
        dt=config['physics']['dt'],
        Number=config['agents']['number'],
        door_visible_radius=config['guide_parameters']['door_visible_radius'],
        knn_k=config['guide_parameters']['knn_k'],
        n_move_guide=config['agents']['n_move_guide'],
        guide_radius=config['guide_parameters']['guide_radius'],
        use_knn=config['guide_parameters']['use_knn'],
        speed_scale=config['physics']['speed_scale'],
        n_static_guide=config['agents']['n_static_guide'],
        obstacle_configs=config.get('obstacles', []),  # Pass obstacle configs during initialization
        knn_max_distance=config['guide_parameters'].get('knn_max_distance', 3.0),
        knn_filter_obstacles=config['guide_parameters'].get('knn_filter_obstacles', True)
    )
    
    print(f"\nEnvironment created:")
    print(f"  - Domain: {env.L[0, 1] - env.L[0, 0]} x {env.L[1, 1] - env.L[1, 0]}")
    print(f"  - Agents: {env.Number}")
    print(f"  - Exits: {len(env.Exit)}")
    print(f"  - Static Guide Points: {env.n_static_guide}")
    print(f"  - Moving Guide Agents: {env.n_move_guide}")
    print(f"  - Guide Radius: {config['guide_parameters']['guide_radius']}")
    print(f"  - Door Visible Radius: {config['guide_parameters']['door_visible_radius']}")
    print(f"  - KNN Max Distance: {env.knn_max_distance}")
    print(f"  - KNN Filter Obstacles: {env.knn_filter_obstacles}")
    
    # Save initial configuration
    output_dir = config['simulation']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "initial_state.cfg")
    
    print(f"\nSaving initial configuration to: {config_file}")
    env.save_output(config_file)
    
    # Parse initial configuration to get domain info
    print("\nParsing initial configuration...")
    exits, guides, obstacles, agents, guide_agents, domain = parse_config_file(config_file)
    
    # Save the ORIGINAL obstacle configs for visualization
    obstacle_configs = config.get('obstacles', [])
    
    print(f"\nInitial state:")
    print(f"  - Exits: {len(exits)}")
    print(f"  - Guides (red points): {len(guides)}")
    print(f"  - Guide Agents (yellow): {len(guide_agents)}")
    print(f"  - Obstacles: {len(obstacle_configs)} (from config)")
    print(f"  - Agents: {len(agents)}")
    
    # Run simulation and save frames
    print("\n" + "=" * 80)
    print("Running Evacuation Simulation")
    print("=" * 80)
    
    # Create frames subdirectory
    frames_dir = os.path.join(output_dir, 'frames')
    
    saved_files = run_simulation(
        env, 
        num_steps=config['simulation']['num_steps'],
        save_interval=config['simulation']['save_interval'],
        output_dir=frames_dir
    )
    
    # Create animated GIF
    print("\n" + "=" * 80)
    print("Creating Animated GIF")
    print("=" * 80)
    
    gif_path = os.path.join(output_dir, config['visualization']['gif_filename'])
    print(f"\nGenerating animation: {gif_path}")
    
    create_animation_from_configs(
        config_dir=frames_dir,
        output_file=gif_path,
        fps=config['visualization']['fps'],
        domain=domain,
        obstacle_configs=obstacle_configs
    )
    
    # Create trajectory plot
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - Configuration files: {len(saved_files)} frames in {frames_dir}")
    print(f"  - Animated GIF: {gif_path}")
    print("\n" + "=" * 80)
    print("Key Features:")
    print("  - Guide points (static) shown as RED circles")
    print("  - Guide agents (moving) shown as YELLOW/GOLD circles")
    print("  - Exits shown as YELLOW stars")
    print("  - Normal agents shown as BLUE dots")
    print("=" * 80)
    print(f"\n[SUCCESS] Animation ready! Open {gif_path} to view.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run guided evacuation simulation')
    parser.add_argument('--config', '-c', type=str, 
                       default='config/simulation_config.json',
                       help='Path to configuration file (default: config/simulation_config.json)')
    
    args = parser.parse_args()
    main(config_path=args.config)
