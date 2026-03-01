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
- [Guide Agent Training (RL)](#guide-agent-training-rl)
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

### ⚡ Vectorized Simulation (NumPy)

**Performance (GuidedCellSpace):**

- **region_confine**: Wall, obstacle, and friction forces computed in bulk (NumPy arrays) instead of per-particle Python loops.
- **loop_cells / loop_neighbors**: Pairwise collision forces within and between cells use matrix operations (distance/force matrices) instead of double loops.
- **Integration**: Leapfrog integration over all particles in one vectorized step.
- For further speed: consider **Numba** (`@numba.jit` on hot loops) or **Cython** (compile pairwise force / integration into a `.pyx` module).

### 🚶 Guide Agent System (Stationary and Mobile)

**Foundation for Intelligent Evacuation Guidance:**

- **Stationary Guide Agents**: Fixed-position agents that help direct evacuees
- **Mobile Guide Agents**: RL-trained agents that move to optimize evacuation flow
- **Actor-Critic (continuous action)** training via `train_guide.py`
- Separate agent type with different behavioral rules

**Current Status:**

- Stationary guides are fully implemented and tested
- Mobile guide RL training is implemented (state, reward, boundary penalty with observable position)
- Guide state includes normalized position so the agent can learn where walls are; reward uses A* direction (obstacle-aware) and avoids noisy terms (no global evacuation progress, no “toward crowd” when the guide has no crowd state)

### 单 Actor + Q(s,a) Critic + 两种 Conformal（2026）

- **Critic**：**Q(s, a)**（状态-动作价值），TD 目标为 `r + γ·Q(s', π(s'))`；对外 `get_value(s)=Q(s, π(s))` 用于 Conformal 与画图。
- **单 Actor**：**ActorMove** 输出连续动作 (vx, vy)（guide 的移动方向/速度），用 **advantage = td_target − Q(s,a)** 做 policy gradient。
- **Value Conformal**：对“价值/回报”做区间预测，校准 (s, G)，得到每步的 [下界, 上界]，图中对比 Critic 预测与真实回报 G_t。
- **Value Conformal**：仅对 Critic 的 return 做 Conformal 区间预测，由 `train.do_value_conformal` 开关。详见 [Guide Agent Training (RL)](#guide-agent-training-rl) 中的架构与 Conformal 小节。

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
- ✅ Mobile guide agent framework and RL training (Actor-Critic, see Guide Agent Training section)

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
- ✅ Mobile guide RL training (Actor-Critic) in `train_guide.py`
- ✅ 13D state (dir_to_avg_pos_xy, avg_vel_dir_xy, astar_dir_xy, x_norm, y_norm, n_remaining_norm, n_escaped_norm, n_first_guided_norm, memory_sum_norm, control_mode) and reward design documented in README

**Files:**

- `train_guide.py` — Guide RL training (Actor-Critic)
- `evacuation_rl/agents/guided_agents/environment.py` — Guided simulation
- `evacuation_rl/environments/cellspace.py` — `get_guide_state()`, `get_guide_dense_reward()`, `get_guide_boundary_penalty()`

---

### Archived Code

The previous **Smart Agents** implementation (which assumed all agents know all exits) has been moved to the `[archive/](archive/)` folder. It represented a different research direction (vector field approach) and is preserved for historical reference only.

See `[archive/README.md](archive/README.md)` for details.

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

### 3. Train Guide Agents (RL)

Train the mobile guide with Actor-Critic (continuous action):

```bash
uv run python train_guide.py
```

Or with standard Python:

```bash
python train_guide.py
```

Reward terms and the 13-dimensional guide state (perception range, A* to exit, normalized position, n_remaining/n_escaped/n_first_guided/memory_sum, and control_mode for critic) are described in [Guide Agent Training (RL)](#guide-agent-training-rl). Config is in `config/simulation_config.json` under `train`.

### 4. View Available Commands

```bash
python -m evacuation_rl --help
```

> **Note:** For archived smart agents code (theoretical baseline), see `[archive/](archive/)` folder.

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
from evacuation_rl.environments import GuidedCellSpace
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
- Mobile guides are trained with Actor-Critic RL; state and reward are designed so the agent can learn wall positions and obstacle-aware directions

---

## Guide Agent Training (RL)

The mobile guide is trained with **Actor-Critic** (continuous action). The following design keeps the reward and state consistent and learnable.

### Guide State (13 dimensions)

The guide observes a **13-dimensional state**: (1) from a **circular perception range** (`perception_radius`), the **direction to the crowd centroid** and the **unit direction of the crowd’s average velocity**; (2) the **A\*** direction to the nearest exit (to compare with crowd movement); (3) the **guide’s normalized position** in the room (`x_norm`, `y_norm` in [0, 1]); (4) **scalars for critic** (all normalized by initial particle count): **remaining evacuee count**, **number escaped this step**, **number first-guided this step**, and **sum of evacuees’ memory strength** (global progress and guide-impact signals); (5) **control_mode**: 1.0 when using RL policy, 0.0 when using scripted visit-based pathfinding (see below).


| Index | Name                    | Range / Type          | Reason                                                                                                                           |
| ----- | ----------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 0     | `dir_to_avg_pos_x`      | unit vector component | Direction from guide to the average position of evacuees in perception range (x).                                             |
| 1     | `dir_to_avg_pos_y`      | unit vector component | Same for y.                                                                                                                      |
| 2     | `avg_vel_dir_x`         | unit vector component | **Unit direction of crowd average velocity** (x). Lets the guide know if the crowd is evacuating or moving randomly.           |
| 3     | `avg_vel_dir_y`         | unit vector component | Same for y.                                                                                                                      |
| 4     | `astar_dir_x`          | unit vector component | A* direction to nearest exit (x). Used to compare with `avg_vel_dir` to see if the crowd is moving toward the exit.             |
| 5     | `astar_dir_y`          | unit vector component | Same for y.                                                                                                                      |
| 6     | `x_norm`                | [0, 1]                | Guide’s x-position in the room, normalized by domain bounds. So the guide knows its rough location (e.g. near left/right walls). |
| 7     | `y_norm`                | [0, 1]                | Same for y. Together with `x_norm`, gives the guide’s global position in the room.                                             |
| 8     | `n_remaining_norm`      | [0, 1]                | Remaining evacuee count / initial count. Tells the critic how many people are still in the room.                                |
| 9     | `n_escaped_norm`        | [0, 1]                | Number of evacuees who exited this step / initial count. One-step escape progress.                                              |
| 10    | `n_first_guided_norm`  | [0, 1]                | Number of evacuees first guided this step / initial count. Guide-impact signal.                                                 |
| 11    | `memory_sum_norm`       | [0, ∞) normalized     | Sum of evacuees’ memory strength / initial count. How much “learned route” the crowd carries.                                    |
| 12    | `control_mode`         | 0.0 or 1.0            | 1.0 = RL policy in control; 0.0 = scripted visit-based pathfinding (when alone and `use_visit_pathfinding_when_alone` is true).   |


When no evacuees are in the perception range, the first four components are 0. When there are no exits, components 4–5 are 0. The perception radius is configured in `config/simulation_config.json` under `guide_parameters.perception_radius`.

**Critic input:** The critic is **Q(s, a)**. Its input is the **state s** (the 13 dimensions in the table above) and the **action a** = (vx, vy) from the actor. So the critic network input dimension is **state_dim + action_dim** (e.g. 13 + 2 = 15). The interface `get_value(s)` used for Conformal and plots returns **Q(s, π(s))** (the value of the current policy at state s).

State is computed in `evacuation_rl/environments/cellspace.py` via `get_guide_state()` and `_get_evacuees_perception_state()`.

### Guide Reward (per step)

Total reward is the sum of the following（已移除基于“圈内人数密度”的 dense reward 和 toward crowd 项）:


| Term                            | Formula / Behavior                                                       | Reason                                                                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Boundary penalty**            | `-get_guide_boundary_penalty(margin, penalty_scale, corner_extra_scale)` | Penalty when the guide is close to walls (distance to nearest wall < margin); extra penalty near corners. **Learnable** from boundary distance and penalties. |
| **Time penalty**                | `+get_time_penalty_reward(time_penalty_scale=...)`                       | 对每个尚未出门的 evacuee 施加持续的时间惩罚：每步为 `-time_penalty_scale × N_remaining`（config 中填正数如 0.01），鼓励 guide 尽快让所有人离开。                                                      |
| **Memory reward (continuous)**  | `+get_guide_memory_reward(step_scale=...)`                               | 当 evacuee 的 `memory_strength > 0` 时，每步按照 memory_strength 给 guide 少量正奖励，鼓励在给出明确路线后，人群沿着记忆路线前进。                                                                 |
| **Memory-first reward (bonus)** | `+get_guide_memory_reward(first_scale=...)`                              | evacuee 第一次在 guide 附近获得明确 A* 路线时，按 memory_strength 给一次性较大正奖励，鼓励 guide 主动“教路”。                                                                                 |
| **Memory-exit reward**          | `+get_guide_memory_reward(exit_scale=...)`                               | evacuee 成功从出口离开时，按照其离开前的 memory_strength 给 guide 一点正奖励，体现“成功带出一批记得路的人”。                                                                                       |


Config for training (e.g. scales, margin, threshold) is under `config/simulation_config.json` → `train`.

### Visit-based pathfinding when alone

When **no evacuees** are within the guide’s `perception_radius`, the guide can optionally use a **scripted visit-based pathfinding** instead of the RL policy (config: `guide_parameters.use_visit_pathfinding_when_alone`). A coarse **visit grid** (normalized in [0, 1]) is used: the guide records how often each grid cell has been visited and moves via A\* toward the **globally least-visited** cell.

- **Reachable cells only:** At **initialization**, which cells are valid targets is determined by **A\* reachability**: from the domain center and from every exit, a BFS is run on the A\* obstacle grid; any visit cell whose center maps to an A\* cell **not** reached by this BFS is marked as blocked and excluded from the “least-visited” search. There is **no** obstacle-area-ratio parameter; unreachable regions are purely from connectivity.
- **Unreachable goal from current position:** If the chosen least-visited cell is unreachable from the guide’s **current** position (e.g. disconnected region), `get_visit_pathfinding_direction()` returns **(0, 0)** so the guide does not move toward an invalid target.
- **Config** (`config/simulation_config.json` → `guide_parameters`):
  - `visit_grid_norm_x`, `visit_grid_norm_y`: normalized cell size in [0, 1] (e.g. 0.1 → 10×10 grid).
  - `use_visit_pathfinding_when_alone`: if true, when there are no evacuees in perception, the guide uses this pathfinding and does **not** call the RL model or train (no `agent.update`). The state still includes `control_mode` (0.0 in this mode, 1.0 when using RL).

### 架构：单 Actor + Q(s,a) Critic

- **Critic**：**Q(s, a)**（状态-动作价值），输入 (s, a)，输出标量。TD 目标为 `r + γ·Q(s', a')`，其中 `a' = π(s')`（当前策略在下一状态的确定性动作）。对外接口 `get_value(s)` 定义为 **Q(s, π(s))**，用于 Conformal 画图和展示。
- **Actor（单一）**：输出 **(vx, vy)**（移动方向/速度），高斯策略。每步都用 **advantage = td_target − Q(s, a)** 做 policy gradient，增大“比预期好”的动作概率；不再区分“是否触发 go_find”。

### Conformal 预测（仅 Value / Critic return）

训练中可选用 **Value Conformal** 对 Critic 回报做不确定性量化，由配置 `train.do_value_conformal` 开关：


| 类型                  | 预测对象   | 校准数据              | 输出                                        | 用途                                       |
| ------------------- | ------ | ----------------- | ----------------------------------------- | ---------------------------------------- |
| **Value Conformal** | 价值（回报） | 多段轨迹的 (s, 真实回报 G) | 每步的 **区间 [下界, 上界]**，盖住真实回报 G_t 的 (1−α) 置信 | 看 Critic 的 Q(s, π(s)) 预测是否可靠，真实回报是否落在区间内 |


- **Value Conformal**：非共形分数为 G − Q(s, π(s))，校准时取分位数得到区间半径；图中纵轴为“价值/回报”，展示 **Q(s, π(s)) (Critic 预测)**、**G_t (真实回报)** 与 Conformal 区间。

配置见 `config/simulation_config.json` 的 `value_conformal`（alpha、calibration_episodes、every_n_episodes、output_dir、figure_name、**fix_seed**、**seed** 等）。**fix_seed**：为 true 时在 conformal 的校准与评估阶段固定随机种子（numpy + torch），保证同一 episode 下每步的 state/轨迹可复现，避免因随机性导致各时间步预测不可比；**seed**：固定种子时使用的整数值（默认 42）。结束后会恢复原 RNG 状态，不影响后续训练。

---

## Configuration

### Environment Parameters

In `evacuation_rl/environments/cellspace.py`:


| Parameter                 | Type  | Default | Description                             |
| ------------------------- | ----- | ------- | --------------------------------------- |
| `door_size`               | float | 1.0     | Door size in absolute units             |
| `agent_size`              | float | 0.5     | Agent radius in absolute units          |
| `reward`                  | float | -0.1    | Reward per step                         |
| `end_reward`              | float | 0.0     | Reward at exit                          |
| `dis_lim`                 | float | 0.05    | Distance threshold for evacuation       |
| `action_force`            | float | 1.0     | Unit action force                       |
| `desire_velocity`         | float | 2.0     | Desired velocity                        |
| `relaxation_time`         | float | 0.5     | Relaxation time                         |
| `delta_t`                 | float | 0.1     | Time step                               |
| `cfg_save_step`           | int   | 5       | Interval for saving config files        |
| `visibility_alpha`        | float | 0.5     | Visibility sensitivity to congestion    |
| `max_visibility_distance` | float | 10      | Max distance for visibility calculation |


### Environment Parameters

In `evacuation_rl/environments/cellspace.py`:

### 🎯 Planned Enhancements

#### 1. RL-Based Guide Agent Training (Implemented)

**Implemented:**

- Mobile guide trained with **Actor-Critic** (continuous action) in `train_guide.py` 
- Conformal prediction for critic performance.
- See [Guide Agent Training (RL)](#guide-agent-training-rl) for full reward and state documentation

**Possible Extensions:**

- Multi-agent coordination between multiple guides
- Tuning reward scales and exploration for different room layouts

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

- ✅ Deep RL training for guide agents (Actor-Critic; reward and state documented in README)
- 🔜 Multi-agent guide coordination
- 🔜 Advanced crowd dynamics modeling

