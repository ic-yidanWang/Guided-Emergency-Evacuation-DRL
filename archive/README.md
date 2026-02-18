# Archive - Deprecated Code

This folder contains deprecated code that is no longer part of the main project but is preserved for historical reference.

## Smart Agents (Deprecated)

**Date Archived:** February 18, 2026

### Why Deprecated?

The smart agents implementation represents a different research direction from the current project goals:

**Smart Agents Approach:**
- Assumes all agents know all exit locations
- Uses Deep Q-Learning to create a vector field/potential field
- State space: Only position and velocity (x, y, vx, vy)
- Creates an optimal direction map for evacuation
- Similar to traditional potential field methods

**Current Project Direction (Guided Agents):**
- **Limited exit visibility** based on distance
- **Crowd-following behavior** when exits are not visible
- **Guide agents** to direct evacuation
- More realistic emergency evacuation scenarios
- Focus on training guide agents using RL

### What's Included

```
archive/
├── smart_agents/              # Smart agent code
│   ├── dqn_network.py        # DQN architecture
│   ├── train_4exits.py       # Training for 4-exit scenario
│   ├── train_3exits_obstacles.py  # Training with obstacles
│   └── test.py               # Testing script
├── models/                    # Pre-trained models
│   ├── Continuum_4Exits_DQN_Fully/
│   └── Continuum_3Exits_Ob_DQN_Fully/
├── Test/                      # Test output data from smart agents
│   └── case_0/ ... case_14/  # Various test scenarios
└── debug_files/               # Debug and development files
    ├── debug_dis_lim.py      # Distance limit debugging
    ├── debug_exit_issue.py   # Exit detection debugging
    ├── test_*.py             # Various test scripts
    ├── EXIT_BUG_FIX_EXPLANATION.md
    └── OBSTACLE_VISUALIZATION_FIX.md
```

### Usage (If Needed)

If you need to reference or use the smart agents code:

1. **View the code:**
   ```bash
   cd archive/smart_agents
   ```

2. **Restore to main project** (not recommended):
   ```bash
   cp -r archive/smart_agents evacuation_rl/agents/
   ```

3. **Use as baseline comparison:**
   The smart agents can serve as a theoretical baseline to compare against the more realistic guided agent approach.

### Key Differences Summary

| Aspect | Smart Agents (Archived) | Guided Agents (Current) |
|--------|------------------------|-------------------------|
| **Exit Knowledge** | All agents know all exits | Distance-based visibility |
| **Behavior** | Optimal pathfinding | Crowd-following + guidance |
| **State Space** | 4D (x, y, vx, vy) | Complex with visibility |
| **Purpose** | Theoretical benchmark | Realistic evacuation |
| **Agent Types** | Single type | Evacuees + Guide agents |
| **Training Goal** | Learn vector field | Train guide agents |

### References

The smart agents implementation was based on:
- Zhang, Y., Chai, Z., & Lykotrafitis, G. (2021). Deep reinforcement learning with a particle dynamics environment applied to emergency evacuation of a room with obstacles. *Physica A*, 571, 125845.

---

**Note:** This code is preserved for reference only. All active development focuses on the guided agent system.
