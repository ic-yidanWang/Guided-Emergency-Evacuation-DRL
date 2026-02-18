"""
Simulation utilities for evacuation simulation
"""

import os
import glob
import numpy as np
from evacuation_rl.environments import GuidedCellSpace
from evacuation_rl.environments import cellspace


def setup_environment(config):
    """
    Setup and configure the evacuation environment from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GuidedCellSpace: Configured environment instance
    """
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
    # Exits are stored as normalized [0,1] coordinates
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
    
    # Create guided environment from config
    add_guide = config['agents'].get('add_guide_agent', False)
    n_guide_agent = 1 if add_guide else 0
    
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
        Number=config['agents']['n_particle'],
        door_visible_radius=config['guide_parameters']['door_visible_radius'],
        knn_k=config['guide_parameters']['knn_k'],
        guide_radius=config['guide_parameters']['guide_radius'],
        use_knn=config['guide_parameters']['use_knn'],
        speed_scale=config['physics']['speed_scale'],
        obstacle_configs=config.get('obstacles', []),
        knn_max_distance=config['guide_parameters'].get('knn_max_distance', 3.0),
        knn_filter_obstacles=config['guide_parameters'].get('knn_filter_obstacles', True),
        n_guide_agent=n_guide_agent,
        guide_initial_position_mode=config['guide_parameters'].get('guide_initial_position_mode', 'random'),
        guide_initial_position=config['guide_parameters'].get('guide_initial_position'),
        guide_influence_gain=config['guide_parameters'].get('guide_influence_gain', 2.0),
        guide_influence_decay=config['guide_parameters'].get('guide_influence_decay', 0.5),
        follow_guide_distance_threshold=config['guide_parameters'].get('follow_guide_distance_threshold', 3.0)
    )
    
    print(f"\nEnvironment created:")
    print(f"  - Domain: {env.L[0, 1] - env.L[0, 0]} x {env.L[1, 1] - env.L[1, 0]}")
    print(f"  - Particles: {env.Number}")
    print(f"  - Exits: {len(env.Exit)}")
    print(f"  - Guide agent: {'yes (1 guide, evacuees can be influenced)' if add_guide else 'no (baseline)'}")
    if add_guide and getattr(env, 'guide_initial_position_mode', 'random') == 'fixed':
        print(f"  - Guide initial position: fixed {getattr(env, 'guide_initial_position', None)}")
    elif add_guide:
        print(f"  - Guide initial position: random")
    print(f"  - Door Visible Radius: {config['guide_parameters']['door_visible_radius']}")
    print(f"  - KNN Max Distance: {env.knn_max_distance}")
    print(f"  - KNN Filter Obstacles: {env.knn_filter_obstacles}")
    
    return env


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
