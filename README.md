# Emergency-evacuation-Deep-reinforcement-learning

This code accompanies "[Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles](https://doi.org/10.1016/j.physa.2021.125845)", which appeared on *Physica A: Statistical Mechanics and its Applications* in 2021.

## About

This project uses a deep reinforcement learning algorithm in association with a particle dynamics model to train agents to find the fastest path to evacuate a room with obstacles.

Efficient emergency evacuation is crucial for survival. However, it is not clear if the application of the self-driven force of the social-force model results in optimal evacuation, especially in complex environments with obstacles. In this work, we developed a deep reinforcement learning algorithm in association with the social force model to train agents to find the fastest evacuation path. During training, we penalized every step of an agent in the room and gave zero reward at the exit. We adopted the Dyna-Q learning approach. We showed that our model can efficiently handle modeling of emergency evacuation in complex environments with multiple room exits and convex and concave obstacles where it is difficult to obtain an intuitive rule for fast evacuation using just the social force model.

## Version History

- **v0.3.0**: Major refactoring with modular structure, smart agents and guided agents separation. See [REFACTORING.md](REFACTORING.md) for details.
- **v0.2.0**: Upgraded from TensorFlow 1.15 to PyTorch 2.0+. See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) and [TF_VS_PYTORCH.md](TF_VS_PYTORCH.md) for details.
- **v0.1.0**: Original TensorFlow implementation (published in Physica A, 2021).

## Project Structure

```
evacuation_rl/                    # Main package
├── environments/                 # Simulation environments
│   └── cellspace.py             # Particle dynamics environment
├── agents/                       # Agent implementations
│   ├── smart_agents/            # All agents know optimal exits
│   │   ├── dqn_network.py       # DQN architecture
│   │   ├── train_3exits_obstacles.py
│   │   ├── train_4exits.py
│   │   └── test.py
│   └── guided_agents/           # Realistic guided evacuation (NEW)
│       ├── environment.py       # Extended environment with guidance
│       └── ... (coming soon)
└── utils/                        # Utilities
    └── visualization.py         # Visualization tools
```

### Smart Agents vs Guided Agents

**Smart Agents (Original Implementation):**
- Assumes all agents know the nearest exit
- Each agent makes optimal individual decisions
For detailed structure explanation and migration guide, see [REFACTORING.md](REFACTORING.md).
**Requirements:**
- Python 3.8 or higher
- PyTorch 2.0+
- NumPy
- Matplotlib

To install this project's package dependencies, please run:

```bash
pip install -r requirements.txt
```

Or using `uv` (recommended):

```bash
uv venv --python 3.8
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Quick Start

### New Modular Interface (Recommended)

```bash
# View available commands
python -m evacuation_rl help

# Train smart agents (3 exits with obstacles)
python -m evacuation_rl.agents.smart_agents.train_3exits_obstacles

# Train smart agents (4 exits)
python -m evacuation_rl.agents.smart_agents.train_4exits

# Test trained models
python -m evacuation_rl.agents.smart_agents.test

# Run examples
python examples.py
```

### Detailed Usage

**Training Smart Agents:**

This project provides framework to train smart agents for emergency evacuation in two scenarios:

1. **An empty room with four exits:**
```bash
python -m evacuation_rl.agents.smart_agents.train_4exits
```

2. **A room with three exits and two obstacles:**
```bash
python -m evacuation_rl.agents.smart_agents.train_3exits_obstacles
```

**Testing:**

After training, you can assess generalization capabilities and visualize the optimal policy learned:

```bash
python -m evacuation_rl.agents.smart_agents.test
```

**Legacy Commands (Deprecated):**

The original training scripts have been removed in v0.3.0. Please use the new modular commands above.

### Configure

This code was developed with many customizable parameters to facilitate its application to different evacuation environments. You can modify the following parameters in the source code to appropriately configure your training:

- In file `evacuation_rl/environments/cellspace.py`:

    | Argument                 | Type     | Default    | Description                                                               |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------------------- |
    | door_size                | float    | 1.0        | Size of door                                                              |
    | agent_size               | float    | 0.5        | Size of agent (particle)                                                  |
    | reward                   | float    | -0.1       | Reward per step                                                           |
    | end_reward               | float    | 0          | Reward at exit                                                            |
    | dis_lim                  | float    | 0.75       | Distance threshold to exit                                                |
    | action_force             | float    | 1.0        | Unit action force                                                         |
    | desire_velocity          | float    | 2.0        | Desired velocity                                                          |
    | relaxation_time          | float    | 0.5        | Relaxation time                                                           |
    | delta_t                  | float    | 0.01       | Time step                                                                 |
    | cfg_save_step            | int      | 5          | Time interval for saving Cfg file                                         |

- In training scripts (e.g., `evacuation_rl/agents/smart_agents/train_4exits.py`):

    | Argument                 | Type     | Default    | Description                                                  |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
    | train_episodes           | int      | 10000      | Max number of episodes to learn from                         |
                    |
    | explore_stop             | float    | 0.1        | Minimum exploration probability                              |
    | learning_rate            | float    | 1e-4       | Adam optimizer learning rate                                 |
    | update_target_every      | int      | 1          | Target network update frequency (episodes)                   |
    | tau                      | float    | 0.1        | Soft update factor for target network                        |
    | save_step                | int      | 1000       | Model checkpoint save frequency (episodes)                   |
    | train_step               | int      | 1          | Training frequency per step                                  |
    | Cfg_save_freq            | int      | 100        | Configuration save frequency (episodes)                      |

- In testing script (e.g., `evacuation_rl/agents/smart_agents/test.py`):

    | Argument                 | Type     | Default    | Description                                                  |
    | ------------------------ | -------- | ---------- | ------------------------------------------------------------ |
    | test_episodes            | int      | 10         | Number of episodes to test                                   |
    | Number_Agent             | int      | 80         | Number of agents to evacuate during test                     |
    | max_steps                | int      | 10000      | Max steps in an episode                                      |
    | Cfg_save_freq            | int      | 1          | Configuration save frequency (episodes)                      |

### Quick Start

For a quick start guide, see [QUICKSTART.md](QUICKSTART.md).

For detailed migration information from TensorFlow, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

## About This Project

This is a graduate research project on critical decision making, focusing on emergency evacuation using deep reinforcement learning.

### Original Reference

This project is based on and extends the following work:
```
@article{ZHANG2021125845,
title = {Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles},
journal = {Physica A: Statistical Mechanics and its Applications},
volume = {571},
pages = {125845},
year = {2021},
issn = {0378-4371},
doi = {https://doi.org/10.1016/j.physa.2021.125845},
url = {https://www.sciencedirect.com/science/article/pii/S0378437121001175},
author = {Yihao Zhang and Zhaojie Chai and George Lykotrafitis},
keywords = {Dyna-Q learning, Particle dynamics simulation, Social-force model, Pedestrian dynamics},
}
```

## Files Overview

### Main Package Structure
- `evacuation_rl/` - Main package containing all core functionality
  - `environments/cellspace.py` - Particle dynamics environment and simulation
  - `agents/smart_agents/` - Smart agent implementations
    - `dqn_network.py` - DQN architecture and training utilities
    - `train_4exits.py` - Train agent in 4-exit scenario
    - `train_3exits_obstacles.py` - Train agent in 3-exit + obstacles scenario
    - `test.py` - Test trained model and visualize policy
  - `agents/guided_agents/` - Guided agent framework (in development)
    - `environment.py` - Extended environment with guidance mechanics
  - `utils/visualization.py` - Visualization utilities

### Utility Scripts
- `examples.py` - Usage examples and demonstrations
- `visualize_evacuation.py` - Evacuation visualization tool
- `quick_visualize.py` - Quick visualization tool

### Documentation
- `README.md` - This file
- `REFACTORING.md` - Detailed refactoring guide and file migration information
- `MIGRATION_GUIDE.md` - PyTorch migration guide from TensorFlow
- `QUICKSTART.md` - Quick start guide
- `TF_VS_PYTORCH.md` - Code comparison between TensorFlow and PyTorch

### Configuration
- `pyproject.toml` - Project metadata and dependencies
- `requirements.txt` - Python dependencies (PyTorch 2.0+)

## License

See LICENSE file for details.
