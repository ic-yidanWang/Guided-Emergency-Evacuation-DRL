# 项目重构说明 (Project Restructuring Guide)

## 概述 (Overview)

本项目已经重构为更清晰、模块化的结构。新结构将"聪明智能体"（假设所有人都知道最近出口）和"引导智能体"（更真实的场景）分开。

The project has been refactored into a clearer, modular structure. The new structure separates "smart agents" (assuming everyone knows the nearest exit) from "guided agents" (more realistic scenarios).

## What's New in v0.3.0 (v0.3.0 新功能)

🎉 **Project has been completely refactored** with a new modular architecture:

- 🏗️ **Modular Structure**: Code organized into `evacuation_rl` package
- 🧠 **Smart Agents**: Original implementation where all agents know optimal exits
- 🎯 **Guided Agents** (NEW): More realistic scenario where only nearby agents know exits, others follow crowd behavior
- 📦 **Better Organization**: Clear separation of environments, agents, and utilities
- 📚 **Improved Documentation**: Comprehensive guides and examples

## 新项目结构 (New Project Structure)

```
Emergency-evacuation-Deep-reinforcement-learning/
├── evacuation_rl/                    # 主要代码库 Main package
│   ├── __init__.py
│   ├── __main__.py                   # 主入口 Main entry point
│   ├── environments/                 # 环境模拟 Simulation environments
│   │   ├── __init__.py
│   │   └── cellspace.py             # 重命名自 Continuum_Cellspace.py
│   ├── agents/                       # 智能体模块 Agent modules
│   │   ├── __init__.py
│   │   ├── smart_agents/            # 聪明智能体 Smart agents
│   │   │   ├── __init__.py
│   │   │   ├── dqn_network.py       # DQN网络定义
│   │   │   ├── train_3exits_obstacles.py  # 3出口+障碍训练
│   │   │   ├── train_4exits.py      # 4出口训练
│   │   │   └── test.py              # 测试脚本
│   │   └── guided_agents/           # 引导智能体 Guided agents (新功能)
│   │       ├── __init__.py
│   │       └── environment.py       # 引导疏散环境
│   └── utils/                        # 工具模块 Utilities
│       ├── __init__.py
│       └── visualization.py         # 可视化工具
├── model/                            # 保存的模型 Saved models
│   ├── smart_agents_3exits_obstacles/
│   └── smart_agents_4exits/
├── output/                           # 训练输出 Training output
├── Test/                             # 测试输出 Test output
├── README.md
├── REFACTORING.md                    # 本文件 This file
└── pyproject.toml
```

## 主要变化 (Key Changes)

### 1. 环境模块 (Environment Module)

**原文件 (Old):** `Continuum_Cellspace.py`  
**新路径 (New):** `evacuation_rl/environments/cellspace.py`

- 更清晰的文档说明
- 保持所有原有功能
- 添加了类型提示和注释

### 2. 智能体分类 (Agent Classification)

#### Smart Agents (聪明智能体)
**假设 (Assumptions):**
- 每个人都知道最近的出口位置
- 每个人都能做出最优决策
- 所有智能体独立行动
- Assumes all agents know the nearest exit
- Each agent makes optimal individual decisions
- Suitable for theoretical analysis

**文件 (Files):**
- `evacuation_rl/agents/smart_agents/train_3exits_obstacles.py`
- `evacuation_rl/agents/smart_agents/train_4exits.py`
- `evacuation_rl/agents/smart_agents/test.py`

**原文件映射 (Original file mapping):**
- `Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py` → `train_3exits_obstacles.py`
- `Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py` → `train_4exits.py`
- `Evacuation_Continuum_DQN_Fully_test_pytorch.py` → `test.py`

#### Guided Agents (引导智能体) - 新功能
**更真实的假设 (More Realistic Assumptions):**
- 只有靠近出口的人知道如何出门
- 其他人会跟随人群行为
- 当周围很多人朝同一方向移动且速度达到阈值时，该人也会跟随
- Guide agent（引导智能体）帮助指导疏散
- Only agents near exits know the optimal way out
- Other agents follow crowd behavior
- Agents move with the crowd when velocity threshold is met
- Guide agents help direct evacuation
- More realistic real-world scenario

**文件 (Files):**
- `evacuation_rl/agents/guided_agents/environment.py` - 引导疏散环境
- 待实现：训练和测试脚本

### 3. 使用方法 (Usage)

#### 旧方法 (Old Way)
```bash
python Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py
python Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py
python Evacuation_Continuum_DQN_Fully_test_pytorch.py
```

#### 新方法 (New Way)
```bash
# 查看帮助
python -m evacuation_rl help

# 训练聪明智能体 (3出口+障碍)
python -m evacuation_rl.agents.smart_agents.train_3exits_obstacles

# 训练聪明智能体 (4出口)
python -m evacuation_rl.agents.smart_agents.train_4exits

# 测试聪明智能体
pyt文件迁移详情 (File Migration Details)

All original files have been migrated to the new modular structure and **old files have been removed**:

所有原始文件已迁移到新的模块化结构，**旧文件已删除**：

**已删除文件（功能已迁移） Deleted Files (functionality migrated):**
- ❌ `Continuum_Cellspace.py` → ✅ `evacuation_rl/environments/cellspace.py`
- ❌ `Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py` → ✅ `evacuation_rl/agents/smart_agents/train_3exits_obstacles.py`
- ❌ `Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py` → ✅ `evacuation_rl/agents/smart_agents/train_4exits.py`
- ❌ `Evacuation_Continuum_DQN_Fully_test_pytorch.py` → ✅ `evacuation_rl/agents/smart_agents/test.py`
- ❌ `main.py` → ✅ `evacuation_rl/__main__.py`

## 向后兼容 (Backward Compatibility)

**注意：** 原有的训练脚本已在 v0.3.0 中删除。请使用新的模块化命令。

**Note:** The original training scripts have been removed in v0.3.0. Please use the new modular commands
python -m evacuation_rl test
```

## 向后兼容 (Backward Compatibility)

原有的训练脚本仍然保留在根目录，可以继续使用。但建议使用新的模块化结构。

The original training scripts are still kept in the root directory and can continue to be used. However, the new modular structure is recommended.

## 模型兼容性 (Model Compatibility)

新代码完全兼容旧模型文件。模型保存路径已更新：

New code is fully compatible with old model files. Model save paths have been updated:

- 旧 (Old): `./model/Continuum_3Exits_Ob_DQN_Fully_pytorch/`
- 新 (New): `./model/smart_agents_3exits_obstacles/`

- 旧 (Old): `./model/Continuum_4Exits_DQN_Fully_pytorch/`
- 新 (New): `./model/smart_agents_4exits/`

## 下一步工作 (Next Steps)

### Guided Agents 实现计划 (Implementation Plan)

1. **环境扩展 (Environment Extension)**
   - ✅ 创建 `GuidedParticle` 类
   - ✅ 创建 `GuidedCellSpace` 类
   - ⏳ 实现出口知识更新机制
   - ⏳ 实现人群跟随行为

2. **Guide Agent (引导智能体)**
   - ⏳ 设计 guide agent 的状态空间
   - ⏳ 设计 guide agent 的动作空间
   - ⏳ 实现 guide agent DQN 网络
   - ⏳ 实现训练循环

3. **实验和评估 (Experiments and Evaluation)**
   - ⏳ 对比有/无 guide agent 的疏散效率
   - ⏳ 分析不同出口知识分布的影响
   - ⏳ 可视化人群跟随行为

## 贡献指南 (Contributing)

如果你想为 guided agents 功能做贡献：

1. 查看 `evacuation_rl/agents/guided_agents/environment.py` 中的 TODO
2. 实现相应的功能
3. 添加测试
4. 更新文档

If you want to contribute to the guided agents feature:

1. Check TODOs in `evacuation_rl/agents/guided_agents/environment.py`
2. Implement the corresponding features
3. Add tests
4. Update documentation

## 问题反馈 (Issues)

如有任何问题或建议，请创建 issue 或直接联系项目维护者。

For any questions or suggestions, please create an issue or contact the project maintainer.
