# Emergency Evacuation - Deep Reinforcement Learning

**深度强化学习应急疏散系统 | Deep Reinforcement Learning for Emergency Evacuation**

> This code accompanies "[Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles](https://doi.org/10.1016/j.physa.2021.125845)", published in *Physica A: Statistical Mechanics and its Applications* (2021).

## Table of Contents

- [About](#about)
- [Recent Improvements](#recent-improvements)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Visualization Tools](#visualization-tools)
- [Exit Visibility System](#exit-visibility-system)
- [Configuration](#configuration)
- [Future Work](#future-work)
- [Reference](#reference)
- [License](#license)

---

## About

This project uses a **deep reinforcement learning (DRL)** algorithm in association with a **particle dynamics model** to train agents to find the fastest path to evacuate a room with obstacles.

### Research Background

Efficient emergency evacuation is crucial for survival. However, it is not clear if the application of the self-driven force of the social-force model results in optimal evacuation, especially in complex environments with obstacles. In this work, we developed a deep reinforcement learning algorithm in association with the social force model to train agents to find the fastest evacuation path. 

**Key Approach:**
- Training: Penalize every step of an agent in the room and give zero reward at the exit
- Algorithm: Dyna-Q learning approach with DQN (Deep Q-Network)
- Environment: Particle dynamics simulation using social force model

**Results:**
Our model can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and convex and concave obstacles where it is difficult to obtain an intuitive rule for fast evacuation using just the social force model.

---

## Recent Improvements

### 🆕 Limited Exit Visibility Agent (February 2026)

**Realistic Exit Perception:**
- Agents can only "see" exits when within a certain distance threshold
- Beyond this distance, agents exhibit crowd-following behavior (herd mentality)
- Simulates real-world scenarios where people don't know exit locations
- More realistic than assuming all agents know all exits

**Key Implementation:**
- Distance-based exit visibility: agents detect exits only when close enough
- Crowd-following behavior: agents follow the flow of nearby people when exits are not visible
- Velocity threshold mechanism: agents match crowd movement patterns

### 🧱 Wall and Obstacle Detection

**Enhanced Collision System:**
- Added comprehensive wall detection mechanism
- Agents now properly avoid room boundaries and obstacles
- Improved collision handling with walls and other agents
- Prevents agents from passing through walls or obstacles

### 📐 Optimized Obstacle Visualization and Collision Detection

**Performance Improvements:**
- Refactored obstacle visualization rendering for better clarity
- Optimized collision detection algorithms for better performance
- More accurate spatial partitioning and cell-based detection
- Enhanced visual representation of obstacles in simulation

### 🚶 Guide Agent System (Stationary and Mobile)

**Foundation for Intelligent Evacuation Guidance:**
- **Stationary Guide Agents**: Fixed-position agents that help direct evacuees
- **Mobile Guide Agents**: Agents capable of moving to optimize evacuation flow
- Lays groundwork for future RL-based guide agent training
- Separate agent type with different behavioral rules

**Current Status:**
- Stationary guides are fully implemented and tested
- Mobile guide framework is in place
- Ready for reinforcement learning integration

---

## Key Features

🎯 **Three Agent Types:**
- **Smart Agents**: Assumes all agents know the nearest exit (theoretical benchmark)
- **Guided Agents**: Realistic scenario with limited exit visibility and crowd-following (NEW)
- **Guide Agents**: Stationary and mobile guides to direct evacuation (NEW)

👁️ **Limited Exit Visibility System:**
- Distance-based exit detection (realistic perception)
- Crowd-following behavior when exits not visible
- Velocity threshold for herd mentality activation
- More realistic emergency scenario simulation

🧱 **Enhanced Collision Detection:**
- Comprehensive wall detection and avoidance
- Optimized obstacle collision algorithms
- Improved spatial partitioning
- Prevents unrealistic wall penetration

🔬 **Advanced Physics Simulation:**
- Particle dynamics with social force model
- Cell-based spatial partitioning for efficient computation
- Real-time collision detection and avoidance
- Optimized visualization rendering

🧠 **Deep Q-Network (DQN):**
- 6-layer fully connected network
- Experience replay for stable training
- Target network for improved convergence
- Ready for guide agent training integration

📊 **Comprehensive Visualization Tools:**
- Real-time animation of evacuation process
- Trajectory plotting for all agents
- Statistical analysis of evacuation efficiency
- Enhanced obstacle and wall visualization

---

## Project Structure

```
Emergency-evacuation-Deep-reinforcement-learning/
├── evacuation_rl/                    # Main package
│   ├── __init__.py
│   ├── __main__.py                   # Main entry point
│   ├── environments/                 # Simulation environments
│   │   ├── __init__.py
│   │   └── cellspace.py             # Particle dynamics environment
│   ├── agents/                       # Agent implementations
│   │   ├── __init__.py
│   │   ├── smart_agents/            # Smart agents (all know exits)
│   │   │   ├── __init__.py
│   │   │   ├── dqn_network.py       # DQN architecture
│   │   │   ├── train_3exits_obstacles.py
│   │   │   ├── train_4exits.py
│   │   │   └── test.py
│   │   └── guided_agents/           # Guided agents (realistic)
│   │       ├── __init__.py
│   │       └── environment.py
│   └── utils/                        # Utilities
│       ├── __init__.py
│       └── visualization.py
├── model/                            # Saved models
│   ├── smart_agents_3exits_obstacles/
│   └── smart_agents_4exits/
├── output/                           # Training output
├── Test/                             # Test output
├── README.md                         # This file
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

### Smart Agents vs Guided Agents

#### Smart Agents (聪明智能体)
**Assumptions:**
- Every person knows the nearest exit location
- Each person makes optimal individual decisions
- All agents act independently
- Suitable for theoretical analysis and benchmarking

**Files:**
- `evacuation_rl/agents/smart_agents/train_3exits_obstacles.py`
- `evacuation_rl/agents/smart_agents/train_4exits.py`
- `evacuation_rl/agents/smart_agents/test.py`

#### Guided Agents (引导智能体) - NEW
**More Realistic Assumptions:**
- **Limited Exit Visibility**: Agents only see exits when within detection distance
- **Crowd-Following Behavior**: Agents follow nearby crowds when exits not visible
- **Velocity-Based Herding**: Agents adopt crowd velocity when threshold is met
- **Wall Detection**: Comprehensive obstacle and boundary awareness
- **Guide Agent Support**: Stationary and mobile guides to direct evacuation
- More realistic real-world emergency evacuation scenario

**Recent Enhancements:**
- ✅ Distance-based exit visibility system
- ✅ Wall and obstacle collision detection
- ✅ Optimized visualization and rendering
- ✅ Stationary guide agents
- ✅ Mobile guide agent framework
- 🔜 RL-based guide agent training (in progress)

**Files:**
- `evacuation_rl/agents/guided_agents/environment.py`
- `run_guided_visualize.py` - Visualization tool for guided simulation
- Training scripts (in development)

---

## Installation

### Requirements

- **Python** >= 3.8
- **PyTorch** >= 2.0.0
- **NumPy** >= 1.21.0
- **Matplotlib** >= 3.5.0

### Method 1: Using pip

```bash
pip install -r requirements.txt
```

### Method 2: Using uv (Recommended)

```bash
# Install uv if you haven't
pip install uv

# Create virtual environment
uv venv --python 3.8

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### GPU Support

For faster training with GPU acceleration, install CUDA-enabled PyTorch:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The code automatically detects and uses GPU if available.

---

## Quick Start

### 1. Train a Model

Train smart agents for 4-exit scenario:
```bash
python -m evacuation_rl.agents.smart_agents.train_4exits
```

Train smart agents for 3-exit + obstacles scenario:
```bash
python -m evacuation_rl.agents.smart_agents.train_3exits_obstacles
```

**Training Parameters:**
- Training episodes: 10,000
- Max steps per episode: 10,000
- Learning rate: 0.0001
- Discount factor (gamma): 0.999
- Memory capacity: 1,000
- Batch size: 50

### 2. Test the Model

After training, test and visualize:
```bash
python -m evacuation_rl.agents.smart_agents.test
```

**Training Progress Display:**
```
Episode: 1234, Loss: 0.001234, Steps: 156, Epsilon: 0.1234
```

- **Episode**: Current training episode
- **Loss**: Current training loss
- **Steps**: Steps taken in this episode
- **Epsilon**: Current exploration rate

### 3. View Available Commands

```bash
python -m evacuation_rl help
```

---

## Examples

### Example 1: Setting up an Evacuation Environment

```python
import numpy as np
from evacuation_rl.environments.cellspace import Cell_Space, Exit, delta_t

# Configure exits (in normalized coordinates [0, 1])
Exit.clear()
Exit.append(np.array([0.5, 1.0, 0.5]))  # Top exit
Exit.append(np.array([0.5, 0.0, 0.5]))  # Bottom exit
Exit.append(np.array([0.0, 0.5, 0.5]))  # Left exit
Exit.append(np.array([1.0, 0.5, 0.5]))  # Right exit

# Create environment
env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=10)
print(f"Created environment with {env.Number} agents")

# Reset and get initial state
state = env.reset()
print(f"Initial state of agent 0: {state}")
```

### Example 2: Loading a Trained Model

```python
import torch
from evacuation_rl.agents.smart_agents.dqn_network import DQN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create model
model = DQN(state_size=4, action_size=8, hidden_size=64)
model.to(device)
model.eval()

# Load checkpoint
checkpoint_path = './model/smart_agents_4exits/checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['main_qn_state_dict'])
print(f"Loaded model from episode {checkpoint['episode']}")
```

### Example 3: Testing Agent Action Selection

```python
import numpy as np
import torch
from evacuation_rl.agents.smart_agents.dqn_network import DQN

model, device = # ... (load model as above)

# Create a test state
test_state = np.array([5.0, 5.0, 0.1, 0.1])  # [x, y, vx, vy]

# Normalize (assuming environment range 0-10)
normalized_state = test_state.copy()
normalized_state[:2] = (normalized_state[:2] - 5.0) / 5.0 - 0.5

# Get action from model
with torch.no_grad():
    state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
    q_values = model(state_tensor).cpu().numpy()[0]

action = np.argmax(q_values)

action_names = ['Up', 'Up-Left', 'Left', 'Down-Left', 
                'Down', 'Down-Right', 'Right', 'Up-Right']

print(f"Q-values: {q_values}")
print(f"Selected action: {action} ({action_names[action]})")
```

### Example 4: Exit Visibility System

```python
from evacuation_rl.environments.cellspace import Cell_Space, Exit

# Configure exits
Exit.append([0.2, 0.5, 0.5])  # Left exit
Exit.append([0.8, 0.5, 0.5])  # Right exit

# Initialize environment
env = Cell_Space(xmin=0., xmax=1., ymin=0., ymax=1., zmin=0., zmax=1., 
                 rcut=0.1, dt=0.01, Number=50)

# Configure visibility system
env.visibility_alpha = 0.5  # Sensitivity to congestion

# Update visibility (automatically called each step)
env.update_visibility_system()

# Get visibility for a particle
sample_particle = env.Cells[0].Particles[0]
visibility = env.get_exit_visibility_for_particle(sample_particle, exit_id=0)
print(f"Exit 0 visibility: {visibility:.3f}")
```

### Example 5: Guided Agents Framework

```python
from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace

print("Guided agents framework features:")
print("  - Only agents near exits know the optimal way out")
print("  - Other agents follow crowd behavior")
print("  - Agents move with crowd when velocity threshold is met")
print("  - Guide agents help direct evacuation")
```

---

## Visualization Tools

This project includes several visualization tools for analyzing evacuation simulations.

### Tool 1: Quick Visualization (Auto-play Animation)

Quickly preview evacuation animation with automatic playback:

```bash
python quick_visualize.py
```

**Features:**
- Auto-play animation with customizable frame rate
- Statistical display (remaining agents, evacuated count, rate)
- Color-coded elements: exits (green stars), obstacles (red squares), agents (blue dots)
- Sampling control for large datasets

**Customization:**
```python
from quick_visualize import create_quick_animation

# Create animation with custom parameters
anim = create_quick_animation(
    case_dir='./Test/case_0',  # Data directory
    step=5,                     # Sample every 5 steps
    max_frames=500              # Limit to 500 frames
)
```

### Tool 2: Full Visualization Tool

More detailed visualization with frame-by-frame control:

```bash
python visualize_evacuation.py
```

**Features:**
- Manual frame navigation
- Single frame visualization
- Detailed statistics display
- Customizable appearance

**Usage:**
```python
from visualize_evacuation import create_animation, visualize_single_frame

# Create animation
anim = create_animation(case_dir='./Test/case_0', max_frames=1000, step=1)

# Or view a single frame
visualize_single_frame(case_dir='./Test/case_0', step_num=100)
```

### Tool 3: Guided Agent Visualization

Visualize guided evacuation with trajectory plotting:

```bash
python run_guided_visualize.py
```

**Features:**
- Trajectory plotting for all agents
- Animated GIF generation
- Evacuation status tracking (evacuating vs evacuated)
- Statistical logging

**Outputs:**
- `output/guided/guided_trajectory.png` - Overall trajectory plot
- `output/guided/guided_animation.gif` - Animated evacuation process

---

## Exit Visibility System

The Exit Visibility System is a key feature that dynamically adjusts exit visibility based on path congestion.

### Core Concept

**When many people are on a path to an exit, that exit becomes less "visible" (harder to assume it's a good choice). When few people are on a path, the exit is more visible (easier to recognize as a good choice).**

### Visibility Calculation

For each cell in the grid, the system calculates the visibility of each exit using:

```
visibility = 1.0 / (1.0 + alpha * particle_count)
```

Where:
- `visibility`: ranges from 0 to 1
  - 1.0 = fully visible (no people on path)
  - 0.0 = not visible (infinite people or unreachable)
- `alpha`: sensitivity parameter (default = 0.5)
- `particle_count`: total number of particles on the shortest path to the exit

### Key Methods

#### `update_visibility_system()`
**Called automatically at each simulation step**

Updates visibility metrics for all cells based on current particle positions.

```python
env.update_visibility_system()
```

#### `get_exit_visibility_for_particle(particle, exit_id=None)`
**Get the visibility of an exit from a particle's perspective**

Returns visibility value (0-1) for a specific exit or the nearest exit.

```python
# Get visibility for nearest exit
nearest_exit_visibility = env.get_exit_visibility_for_particle(particle)

# Get visibility for specific exit
exit_visibility = env.get_exit_visibility_for_particle(particle, exit_id=0)
```

#### `bfs_path_to_exit(from_cell_id, exit_id)`
**Find shortest path from a cell to an exit**

Returns list of cell IDs forming the path.

```python
path = env.bfs_path_to_exit(cell_id=5, exit_id=0)
```

#### `calculate_path_visibility(path, exit_id)`
**Calculate visibility for a specific path**

Returns tuple (particle_count, visibility).

```python
particle_count, visibility = env.calculate_path_visibility(path, exit_id=0)
```

### Cell Attributes

Each cell maintains visibility information:

```python
cell = env.Cells[cell_id]

# Get visibility to exit 0
visibility = cell.exit_visibility[0]  # 0.0 to 1.0

# Get particle count on path to exit 0
count = cell.particle_count_to_exit[0]

# Get the path (list of cell IDs)
path = cell.path_to_exit[0]
```

### Configuration Parameters

```python
# Sensitivity to congestion (higher = more sensitive)
env.visibility_alpha = 0.5

# Maximum distance to calculate visibility for
env.max_visibility_distance = 10
```

### Example Usage

See the complete example in the [Examples](#example-4-exit-visibility-system) section above.

---

## Configuration

### Environment Parameters

In `evacuation_rl/environments/cellspace.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `door_size` | float | 1.0 | Door size in absolute units |
| `agent_size` | float | 0.5 | Agent radius in absolute units |
| `reward` | float | -0.1 | Reward per step |
| `end_reward` | float | 0.0 | Reward at exit |
| `dis_lim` | float | 0.05 | Distance threshold for evacuation |
| `action_force` | float | 1.0 | Unit action force |
| `desire_velocity` | float | 2.0 | Desired velocity |
| `relaxation_time` | float | 0.5 | Relaxation time |
| `delta_t` | float | 0.1 | Time step |
| `cfg_save_step` | int | 5 | Interval for saving config files |
| `visibility_alpha` | float | 0.5 | Visibility sensitivity to congestion |
| `max_visibility_distance` | float | 10 | Max distance for visibility calculation |

### Training Parameters

In training scripts (e.g., `evacuation_rl/agents/smart_agents/train_4exits.py`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_episodes` | int | 10000 | Max number of episodes |
| `max_steps` | int | 10000 | Max steps per episode |
| `batch_size` | int | 50 | Training batch size |
| `gamma` | float | 0.999 | Discount factor |
| `memory_size` | int | 1000 | Experience replay buffer size |
| `explore_start` | float | 1.0 | Initial exploration rate |
| `explore_stop` | float | 0.1 | Minimum exploration rate |
| `decay_rate` | float | 0.0001 | Exploration decay rate |
| `learning_rate` | float | 1e-4 | Adam optimizer learning rate |
| `update_target_every` | int | 1 | Target network update frequency |
| `tau` | float | 0.1 | Soft update factor |
| `save_step` | int | 1000 | Model checkpoint save frequency |
| `train_step` | int | 1 | Training frequency per step |
| `Cfg_save_freq` | int | 100 | Config save frequency |

### Testing Parameters

In testing script (`evacuation_rl/agents/smart_agents/test.py`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_episodes` | int | 10 | Number of episodes to test |
| `Number_Agent` | int | 80 | Number of agents to evacuate |
| `max_steps` | int | 10000 | Max steps in an episode |
| `Cfg_save_freq` | int | 1 | Config save frequency |

---

## Future Work

### 🎯 Planned Enhancements

#### 1. RL-Based Guide Agent Training
**Primary Goal:**
- Train mobile guide agents using Deep Reinforcement Learning
- Optimize guide agent positioning and movement strategies
- Develop reward functions for effective evacuation guidance
- Multi-agent coordination between guides and evacuees

**Technical Approach:**
- Extend DQN architecture for guide agent decision-making
- State space: guide position, evacuee distribution, exit congestion
- Action space: guide movement directions and influence radius
- Reward design: minimize total evacuation time, prevent congestion

#### 2. Advanced Crowd Dynamics
- Implement more sophisticated crowd psychology models
- Add panic and stress factors to agent behavior
- Model information propagation through crowd
- Study emergent evacuation patterns

#### 3. Multi-Agent Guide Coordination
- Multiple guide agents working collaboratively
- Communication protocols between guides
- Distributed decision-making strategies
- Load balancing across multiple exits

#### 4. Adaptive Exit Selection
- Dynamic exit recommendation based on real-time congestion
- Predictive modeling of evacuation flow
- Integration with IoT sensors for smart buildings
- Emergency response system integration

#### 5. Extended Scenarios
- Multi-floor building evacuations
- Complex building layouts with corridors
- Dynamic obstacles (collapsed structures, fire spread)
- Heterogeneous agent populations (different mobility levels)

---

## Reference

This project is based on the research published in:

**Zhang, Y., Chai, Z., & Lykotrafitis, G. (2021).** Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles. *Physica A: Statistical Mechanics and its Applications*, 571, 125845.

**DOI:** [https://doi.org/10.1016/j.physa.2021.125845](https://doi.org/10.1016/j.physa.2021.125845)

---

## License

See [LICENSE](LICENSE) file for details.

---

## About This Project

This is a graduate research project on critical decision making, focusing on emergency evacuation using deep reinforcement learning.

**Contributors:** Research team at Decision Making course, Postgraduate Studies

**Latest Update:** February 18, 2026

### Recent Development Focus

**Phase 1 (Completed):**
- ✅ Limited exit visibility implementation
- ✅ Wall and obstacle detection system
- ✅ Optimized collision detection algorithms
- ✅ Enhanced visualization tools
- ✅ Stationary and mobile guide agent framework

**Phase 2 (In Progress):**
- 🔜 Deep RL training for guide agents
- 🔜 Multi-agent guide coordination
- 🔜 Advanced crowd dynamics modeling
