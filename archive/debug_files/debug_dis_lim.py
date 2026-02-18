"""Debug exit detection in step_guided"""
import numpy as np
from evacuation_rl.environments import cellspace
from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace
import json

# Configure
with open("config/simulation_config.json", 'r') as f:
    config = json.load(f)

cellspace.door_size = config['exit_parameters'].get('door_size', 1.0)
cellspace.agent_size = config['exit_parameters'].get('agent_size', 0.2)
# BUGFIX: dis_lim must account for agent size!
cellspace.dis_lim = cellspace.agent_size + cellspace.door_size

print(f"Set cellspace.dis_lim = {cellspace.dis_lim} (agent_size + door_size)")

# Configure exits
cellspace.Exit = [np.array([e['x'], e['y'], e['z']]) for e in config['exits']]

# Configure obstacles
cellspace.Ob = []
cellspace.Ob_size = []
cellspace.Ob_type = []
cellspace.Ob_params = []

for obstacle in config.get('obstacles', []):
    obs_type = obstacle.get('type', 'circle')
    if obs_type == 'circle':
        cellspace.Ob.append([np.array([obstacle['x'], obstacle['y'], obstacle['z']])])
        cellspace.Ob_size.append(obstacle.get('size', 0.5) * 0.8)
        cellspace.Ob_type.append('circle')
        cellspace.Ob_params.append({})
    elif obs_type == 'rectangle':
        center_x = obstacle['x']
        center_y = obstacle['y']
        center_z = obstacle['z']
        width = obstacle.get('width', 0.4)
        height = obstacle.get('height', 0.3)
        cellspace.Ob.append([np.array([center_x, center_y, center_z])])
        cellspace.Ob_size.append(0.05)
        cellspace.Ob_type.append('rectangle')
        cellspace.Ob_params.append({'center': np.array([center_x, center_y, center_z]), 'width': width, 'height': height})

cellspace.Guide = []

# Create environment
env = GuidedCellSpace(
    xmin=config['domain']['xmin'],
    xmax=config['domain']['xmax'],
    ymin=config['domain']['ymin'],
    ymax=config['domain']['ymax'],
    zmin=config['domain']['zmin'],
    zmax=config['domain']['zmax'],
    rcut=config['physics']['rcut'],
    dt=config['physics']['dt'],
    Number=3,
    door_visible_radius=config['guide_parameters']['door_visible_radius'],
    knn_k=config['guide_parameters']['knn_k'],
    n_move_guide=config['agents']['n_move_guide'],
    guide_radius=config['guide_parameters']['guide_radius'],
    use_knn=config['guide_parameters']['use_knn'],
    speed_scale=config['physics']['speed_scale'],
    n_static_guide=config['agents']['n_static_guide'],
    obstacle_configs=config.get('obstacles', []),
    knn_max_distance=config['guide_parameters'].get('knn_max_distance', 3.0),
    knn_filter_obstacles=config['guide_parameters'].get('knn_filter_obstacles', True)
)

print(f"After GuidedCellSpace init, cellspace.dis_lim = {cellspace.dis_lim}")

# Insert a trace monkey-patch to see if step_guided is detecting exits
original_step_guided = env.step_guided

def traced_step_guided():
    print(f"  [TRACE] step_guided called, cellspace.dis_lim = {cellspace.dis_lim}")
    
    # Check agent distances before step
    for c in env.Cells:
        for p in c.Particles:
            for e in env.Exit:
                dis = np.sqrt(np.sum((p.position - e) ** 2))
                if dis < 2.0:
                    print(f"    Agent {p.ID} at distance {dis:.3f} from exit")
    
    done = original_step_guided()
    
    print(f"  [TRACE] After step, agents remaining = {env.Number}")
    return done

env.step_guided = traced_step_guided

# Run a few steps
print("\nRunning simulation...")
for step in range(30):
    done = env.step_guided()
    if step % 5 == 0:
        print(f"Step {step}: {env.Number} agents")
    if env.Number == 0:
        print(f"All agents evacuated at step {step}")
        break
