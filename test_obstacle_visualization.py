"""
Test script to verify obstacle visualization is working correctly
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from evacuation_rl.utils.visualization import parse_config_file

# Load the config
with open('config/simulation_config.json', 'r') as f:
    config = json.load(f)

# Get domain info
domain = {
    'x': config['domain']['xmax'] - config['domain']['xmin'],
    'y': config['domain']['ymax'] - config['domain']['ymin'],
    'z': config['domain']['zmax'] - config['domain']['zmin']
}

# Parse the initial configuration file
exits, guides, obstacles, agents, guide_agents, parsed_domain = parse_config_file(
    'output/guided/initial_state.cfg'
)

# Get obstacle configs from JSON
obstacle_configs = config.get('obstacles', [])

print(f"\nDomain info:")
print(f"  Config domain: {domain}")
print(f"  Parsed domain: {parsed_domain}")
print(f"\nObstacle config from JSON:")
for obs in obstacle_configs:
    print(f"  {obs}")

print(f"\nParsed obstacles from file:")
print(f"  Number of obstacle points: {len(obstacles)}")
if obstacles:
    print(f"  Sample: {obstacles[0]}")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Plot 1: Using parsed data (from file)
ax1.set_xlim(0, domain['x'])
ax1.set_ylim(0, domain['y'])
ax1.set_aspect('equal')
ax1.set_title('Obstacles from Parsed Config File')
ax1.grid(True, alpha=0.3)

# Plot obstacles from parsed file
if obstacles:
    obstacles_arr = np.array(obstacles)
    # Scale from [0,1] to absolute coordinates
    ax1.scatter(obstacles_arr[:, 0] * domain['x'], obstacles_arr[:, 1] * domain['y'], 
               c='black', marker='s', s=100, label=f'Obstacles ({len(obstacles)} points)', zorder=3)

# Plot exits
if exits:
    exits_arr = np.array(exits)
    ax1.scatter(exits_arr[:, 0] * domain['x'], exits_arr[:, 1] * domain['y'], 
               c='yellow', marker='*', s=500, edgecolors='black', 
               linewidths=2, label='Exits', zorder=5)

ax1.legend()

# Plot 2: Using JSON config (as used in visualization)
ax2.set_xlim(0, domain['x'])
ax2.set_ylim(0, domain['y'])
ax2.set_aspect('equal')
ax2.set_title('Obstacles from JSON Config (What should display)')
ax2.grid(True, alpha=0.3)

# Plot obstacles from JSON config
if obstacle_configs:
    for obs in obstacle_configs:
        obs_type = obs.get('type', 'circle')
        
        if obs_type == 'circle':
            center_x = obs['x']
            center_y = obs['y']
            radius = obs.get('size', 0.5)
            
            circle = Circle((center_x, center_y), radius, 
                           linewidth=2, edgecolor='black', 
                           facecolor='gray', alpha=0.4, zorder=3)
            ax2.add_patch(circle)
            print(f"\nCircle drawn at ({center_x}, {center_y}) with radius {radius}")
        
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
            ax2.add_patch(rect)

# Plot exits
if exits:
    exits_arr = np.array(exits)
    ax2.scatter(exits_arr[:, 0] * domain['x'], exits_arr[:, 1] * domain['y'], 
               c='yellow', marker='*', s=500, edgecolors='black', 
               linewidths=2, label='Exits', zorder=5)

ax2.legend()

plt.tight_layout()
plt.savefig('output/guided/obstacle_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to output/guided/obstacle_comparison.png")
plt.show()
