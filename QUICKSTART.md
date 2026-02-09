# 快速开始 Quick Start Guide

## PyTorch 版本快速开始 / PyTorch Version Quick Start

### 1. 安装依赖 / Install Dependencies

#### 方法 A: 使用 pip
```bash
pip install torch numpy matplotlib
```

或者

```bash
pip install -r requirements.txt
```

#### 方法 B: 使用 uv (推荐)
```bash
# 安装 uv (如果还没安装)
pip install uv

# 创建虚拟环境
uv venv --python 3.8

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 2. 训练模型 / Train Model

#### 场景 1: 4个出口场景
```bash
python Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py
```

训练参数说明：
- 训练回合: 10,000 episodes
- 每回合最大步数: 10,000 steps
- 学习率: 0.0001
- 折扣因子 gamma: 0.999
- 记忆容量: 1,000
- 批次大小: 50

#### 场景 2: 3个出口 + 障碍物场景
```bash
python Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py
```

### 3. 测试模型 / Test Model

训练完成后，运行测试脚本：

```bash
python Evacuation_Continuum_DQN_Fully_test_pytorch.py
```

**注意**: 在测试脚本中，需要根据使用的场景选择对应的模型：
- 4出口场景: 使用 `mainQN_4Exits`
- 3出口+障碍物场景: 使用 `mainQN_3Exits_Ob`

### 4. 查看结果 / View Results

训练过程中的输出保存在：
- 模型文件: `./model/Continuum_*_pytorch/`
- 配置输出: `./output/Continuum_*_pytorch/`

测试结果保存在：
- `./Test/case_*/`

## 训练监控 / Training Monitoring

训练过程中会显示：
```
Episode: 1234, Loss: 0.001234, Steps: 156, Epsilon: 0.1234
```

- **Episode**: 当前训练回合数
- **Loss**: 当前损失值
- **Steps**: 该回合的步数
- **Epsilon**: 当前探索率

## GPU 加速 / GPU Acceleration

代码会自动检测并使用 GPU (如果可用)：
```
Using device: cuda    # 使用 GPU
Using device: cpu     # 使用 CPU
```

### 安装 CUDA 版本的 PyTorch

如果你有 NVIDIA GPU，可以安装 CUDA 版本以获得更快的训练速度：

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 常见问题排查 / Troubleshooting

### 问题 1: ImportError: No module named 'torch'
```bash
pip install torch
```

### 问题 2: Python 版本过低
需要 Python 3.8 或更高版本：
```bash
python --version
# 应该显示 3.8.x 或更高
```

### 问题 3: CUDA out of memory
如果 GPU 内存不足，可以：
1. 减小批次大小 (batch_size)
2. 减小记忆容量 (memory_size)
3. 使用 CPU 训练

### 问题 4: 训练速度慢
- 检查是否正在使用 GPU
- 尝试增加批次大小
- 确保没有其他程序占用 GPU

## 超参数调整 / Hyperparameter Tuning

可以在训练脚本中调整以下参数：

```python
# 训练参数
train_episodes = 10000        # 训练回合数
max_steps = 10000             # 每回合最大步数
learning_rate = 1e-4          # 学习率
gamma = 0.999                 # 折扣因子

# 探索参数
explore_start = 1.0           # 起始探索率
explore_stop = 0.1            # 最小探索率
decay_rate = 8.0              # 探索衰减率

# 记忆参数
memory_size = 1000            # 经验回放容量
batch_size = 50               # 批次大小

# 网络更新参数
update_target_every = 1       # 目标网络更新频率
tau = 0.1                     # 软更新因子
```

## 性能优化建议 / Performance Optimization

1. **使用 GPU**: 如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch
2. **调整批次大小**: 增加 batch_size 可以提高 GPU 利用率
3. **并行化**: 考虑使用多进程进行数据收集
4. **混合精度训练**: 使用 `torch.cuda.amp` 加速训练

## 下一步 / Next Steps

1. 阅读 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) 了解从 TensorFlow 迁移的详细信息
2. 查看代码注释了解每个组件的功能
3. 尝试调整超参数优化性能
4. 根据需求修改网络结构

## 项目结构 / Project Structure

```
.
├── Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py  # 训练脚本 (3出口+障碍物)
├── Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py     # 训练脚本 (4出口)
├── Evacuation_Continuum_DQN_Fully_test_pytorch.py       # 测试脚本
├── Continuum_Cellspace.py                               # 环境定义
├── requirements.txt                                      # Python 依赖
├── pyproject.toml                                        # 项目配置
└── MIGRATION_GUIDE.md                                    # 迁移指南
```

## 帮助与支持 / Help & Support

- 查看代码注释获取详细说明
- 阅读 PyTorch 官方文档: https://pytorch.org/docs/
- 查看 PyTorch 教程: https://pytorch.org/tutorials/

---

**祝你训练顺利！ / Happy Training!** 🚀
