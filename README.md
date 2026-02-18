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

🎯 **Two Agent Types:**
- **Evacuee Agents**: Realistic agents with limited exit visibility and crowd-following behavior
- **Guide Agents**: Stationary and mobile guides to direct evacuation (framework ready for RL training)

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
│   │   └── guided_agents/           # Guided agents (realistic)
│   │       ├── __init__.py
│   │       └── environment.py
│   └── utils/                        # Utilities
│       ├── __init__.py
│       └── visualization.py
├── config/                           # Configuration files
│   ├── simulation_config.json
│   ├── single_exit.json
│   ├── with_obstacles.json
│   └── large_scale.json
├── model/                            # Saved models (for future guide agents)
├── output/                           # Training output
│   └── guided/                      # Guided simulation outputs
├── archive/                          # Deprecated code (smart agents)
├── run_guided_visualize.py          # Guided simulation visualizer
├── README.md                         # This file
├── pyproject.toml
└── LICENSE
```

### Agent Types

#### Evacuee Agents with Limited Visibility
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
- `evacuation_rl/agents/guided_agents/environment.py` - Main environment implementation
- `run_guided_visualize.py` - Visualization tool for guided simulation
- Training scripts (in development)

#### Guide Agents
**Purpose:**
- Direct evacuees toward optimal exits
- Help prevent congestion and improve evacuation efficiency
- Can be stationary or mobile

**Current Status:**
- ✅ Framework implemented
- ✅ Stationary guides working
- ✅ Mobile guide structure in place
- 🔜 RL-based training (planned)

**Files:**
- Integrated in `evacuation_rl/agents/guided_agents/environment.py`

---

### Archived Code

The previous **Smart Agents** implementation (which assumed all agents know all exits) has been moved to the [`archive/`](archive/) folder. It represented a different research direction (vector field approach) and is preserved for historical reference only.

See [`archive/README.md`](archive/README.md) for details.

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

### 1. Run Guided Evacuation Simulation

Run the guided agent simulation with visualization:
```bash
uv run python run_guided_visualize.py
```

Or using standard Python:
```bash
python run_guided_visualize.py
```

**Features:**
- Distance-based exit visibility
- Crowd-following behavior
- Wall and obstacle detection
- Trajectory visualization
- Statistical output

**Outputs:**
- `output/guided/guided_trajectory.png` - Trajectory plot
- `output/guided/guided_animation.gif` - Animation (if enabled)
- Console statistics

### 2. Configure Simulation

Edit configuration files in the `config/` folder:
```bash
config/simulation_config.json     # Main configuration
config/single_exit.json          # Single exit scenario
config/with_obstacles.json       # With obstacles
config/large_scale.json          # Large-scale simulation
```

### 3. Train Guide Agents (Coming Soon)

Guide agent training using RL will be implemented in future updates.

### 4. View Available Commands

```bash
python -m evacuation_rl --help
```

> **Note:** For archived smart agents code (theoretical baseline), see [`archive/`](archive/) folder.

---

## Quick Examples

### Running Guided Evacuation Simulation

```bash
# Run guided evacuation with visualization
uv run python run_guided_visualize.py
```

This will:
- Create a guided evacuation simulation
- Apply distance-based exit visibility
- Enable crowd-following behavior
- Generate trajectory plots and statistics
- Save outputs to `output/guided/`

### Basic Environment Setup

```python
import numpy as np
from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace
import json

# Load configuration
with open("config/simulation_config.json", 'r') as f:
    config = json.load(f)

# Create guided environment
env = GuidedCellSpace(
    config=config,
    num_agents=50,
    exit_visibility_distance=3.0  # Agents can see exits within 3 units
)

# Run simulation
for step in range(1000):
    done = env.step_guided()
    if done:
        print(f"All evacuated in {step} steps!")
        break
```

### Configuration Files

Edit `config/simulation_config.json` to customize:

```json
{
  "exits": [[0.5, 1.0, 0.5], [0.5, 0.0, 0.5]],
  "obstacles": [...],
  "exit_visibility_distance": 3.0,
  "crowd_following_threshold": 0.5,
  "num_guide_agents": 2
}
```

---

## Visualization

### Guided Evacuation Visualization

The main visualization tool for guided evacuation:

```bash
python run_guided_visualize.py
```

**Features:**
- Real-time evacuation simulation
- Distance-based exit visibility
- Crowd-following behavior visualization
- Wall and obstacle detection
- Guide agent tracking
- Trajectory plotting

**Outputs:**
- `output/guided/guided_trajectory.png` - Overall trajectory plot
- `output/guided/guided_animation.gif` - Animated process (optional)
- Console statistics and progress

---

## Key Concepts

### Distance-Based Exit Visibility

Agents can only "see" exits when within a certain distance:

```python
if distance_to_exit < exit_visibility_distance:
    # Agent knows about this exit
    move_towards_exit()
else:
    # Agent follows crowd behavior
    follow_nearby_agents()
```

This is more realistic than assuming all agents know all exit locations.

### Crowd-Following Behavior

When exits are not visible, agents follow nearby crowds:

```python
if avg_neighbor_velocity > crowd_threshold:
    # Join the crowd movement
    velocity = match_crowd_velocity(neighbors)
```

### Guide Agents

Guide agents (stationary or mobile) help direct evacuees:
- Positioned strategically near exits or in corridors  
- Influence evacuee movement decisions
- Future: Will be trained using RL to optimize positioning

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

### Environment Parameters

In `evacuation_rl/environments/cellspace.py`:

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
