#!/usr/bin/env python3
"""
Debug script to understand why agents can't exit with small door_size
"""
import numpy as np
import json
from evacuation_rl.environments import cellspace
from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace

# Load configuration
config_path = 'config/simulation_config.json'
with open(config_path) as f:
    config = json.load(f)

# Set up parameters from config  
door_size = config['exit_parameters'].get('door_size', 1.0)
agent_size = config['exit_parameters'].get('agent_size', 0.5)

print(f"Initial parameters from config:")
print(f"  door_size: {door_size}")
print(f"  agent_size: {agent_size}")
print()

# Apply to cellspace module (mimicking run_guided_visualize.py)
cellspace.door_size = door_size
cellspace.agent_size = agent_size
# BUGFIX: dis_lim must account for agent size! Agents can't get their center to exit center
# if they're wider than the door. The equilibrium distance is roughly agent_size + door_size
cellspace.dis_lim = cellspace.agent_size + cellspace.door_size  # FIX: was just cellspace.door_size
print(f"After assignment to cellspace module:")
print(f"  cellspace.door_size: {cellspace.door_size}")
print(f"  cellspace.agent_size: {cellspace.agent_size}")
print(f"  cellspace.dis_lim: {cellspace.dis_lim} (FIXED: now agent_size + door_size)")
print()

# Set up exits from config (these are in absolute coordinates)
cellspace.Exit = [
    np.array([exit_cfg['x'], exit_cfg['y'], exit_cfg['z']]) 
    for exit_cfg in config['exits']
]
print(f"Exit positions (from config - in absolute coordinates):")
for i, e in enumerate(cellspace.Exit):
    print(f"  Exit {i}: {e}")
print()

# Create the environment
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
    obstacle_configs=config.get('obstacles', []),
    knn_max_distance=config['guide_parameters']['knn_max_distance'],
    knn_filter_obstacles=config['guide_parameters'].get('knn_filter_obstacles', True)
)

print(f"Environment created:")
print(f"  Number of agents: {env.Number}")
print(f"  Number of exits: {len(env.Exit)}")
print(f"  env.door_visible_radius: {env.door_visible_radius}")
print(f"  env.dis_lim at env: {cellspace.dis_lim}")
print()

# Run a few steps and track distances
print("=" * 80)
print("Running simulation and tracking agent distances to exits...")
print("=" * 80)

for step in range(20):
    env.step_guided()
    
    if step % 5 == 0:
        print(f"\nStep {step}:")
        print(f"  Agents remaining: {env.Number}")
        
        if env.Number > 0:
            # Find closest agent to any exit
            min_dist_to_exit = np.inf
            closest_agent = None
            closest_exit = None
            
            for c in env.Cells:
                for p in c.Particles:
                    for e_idx, e in enumerate(env.Exit):
                        dist = np.sqrt(np.sum((p.position - e) ** 2))
                        if dist < min_dist_to_exit:
                            min_dist_to_exit = dist
                            closest_agent = p
                            closest_exit = e_idx
            
            if closest_agent is not None:
                agent_pos = closest_agent.position
                exit_pos = env.Exit[closest_exit]
                print(f"  Closest agent:")
                print(f"    Position: ({agent_pos[0]:.3f}, {agent_pos[1]:.3f}, {agent_pos[2]:.3f})")
                print(f"    Distance to exit {closest_exit} at ({exit_pos[0]:.1f}, {exit_pos[1]:.1f}): {min_dist_to_exit:.4f}")
                print(f"    cellspace.dis_lim: {cellspace.dis_lim}")
                print(f"    Will evacuate? {min_dist_to_exit < cellspace.dis_lim}")
        
        if env.Number == 0:
            print(f"\nAll agents evacuated at step {step}!")
            break

print("\n" + "=" * 80)
print("Debug complete")
print("=" * 80)
