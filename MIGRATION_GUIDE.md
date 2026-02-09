# PyTorch 迁移指南 / PyTorch Migration Guide

## 概述 / Overview

本项目已从 **TensorFlow 1.15** 升级到 **PyTorch 2.0+**，同时 Python 版本要求从 3.5 升级到 3.8+。

The project has been upgraded from **TensorFlow 1.15** to **PyTorch 2.0+**, and Python version requirement has been upgraded from 3.5 to 3.8+.

---

## 主要变化 / Key Changes

### 1. 依赖更新 / Dependencies Update

**旧版本 / Old:**
```
Python 3.5
tensorflow==1.15
tensorflow-probability==0.7
trfl==1.1
matplotlib
```

**新版本 / New:**
```
Python >= 3.8
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### 2. 文件对照 / File Mapping

| 旧文件 (TensorFlow) | 新文件 (PyTorch) |
|-------------------|------------------|
| `Evacuation_Continuum_3Exits_Ob_DQN_Fully.py` | `Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py` |
| `Evacuation_Continuum_4Exits_DQN_Fully.py` | `Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py` |
| `Evacuation_Continuum_DQN_Fully_test.py` | `Evacuation_Continuum_DQN_Fully_test_pytorch.py` |

---

## 安装 / Installation

### 方法 1: 使用 pip

```bash
pip install -r requirements.txt
```

### 方法 2: 使用 uv (推荐)

```bash
# 创建虚拟环境
uv venv --python 3.8

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

---

## 使用方法 / Usage

### 训练模型 / Training

#### 3 出口 + 障碍物场景 / 3 Exits with Obstacles
```bash
python Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py
```

#### 4 出口场景 / 4 Exits
```bash
python Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py
```

### 测试模型 / Testing

```bash
python Evacuation_Continuum_DQN_Fully_test_pytorch.py
```

---

## 代码架构变化 / Architecture Changes

### DQN 网络定义 / DQN Network Definition

**旧版本 (TensorFlow 1.x):**
```python
class DQN:
    def __init__(self, name, learning_rate=0.0001, gamma=0.99, ...):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')
            self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
            # ...
```

**新版本 (PyTorch):**
```python
class DQN(nn.Module):
    def __init__(self, state_size=4, action_size=8, hidden_size=64):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # ...
        )
```

### 训练循环 / Training Loop

**旧版本 (TensorFlow 1.x):**
```python
with tf.Session(config=config) as sess:
    sess.run(init)
    # 训练代码
    loss, _ = sess.run([mainQN.loss, mainQN.opt], feed_dict={...})
```

**新版本 (PyTorch):**
```python
main_qn = DQN().to(device)
optimizer = optim.Adam(main_qn.parameters(), lr=learning_rate)

# 训练代码
loss = train_dqn(main_qn, target_qn, optimizer, batch, gamma, device)
```

### 模型保存和加载 / Model Save/Load

**旧版本 (TensorFlow 1.x):**
```python
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")
saver.restore(sess, "model.ckpt")
```

**新版本 (PyTorch):**
```python
# 保存
checkpoint = {
    'episode': ep,
    'main_qn_state_dict': main_qn.state_dict(),
    'target_qn_state_dict': target_qn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth', map_location=device)
main_qn.load_state_dict(checkpoint['main_qn_state_dict'])
```

---

## 模型文件格式 / Model File Format

### TensorFlow 模型文件
- `checkpoint`
- `*.ckpt-xxxx.data-00000-of-00001`
- `*.ckpt-xxxx.index`
- `*.ckpt-xxxx.meta`

### PyTorch 模型文件
- `checkpoint.pth` (最新检查点)
- `checkpoint_ep_xxxx.pth` (每隔固定 episode 保存)

---

## GPU 支持 / GPU Support

### 自动检测设备 / Automatic Device Detection

PyTorch 版本会自动检测并使用可用的 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

如果有 CUDA GPU，模型会自动在 GPU 上训练，否则使用 CPU。

---

## 性能对比 / Performance Comparison

### 优势 / Advantages

1. **简洁性**: PyTorch 代码更简洁直观
2. **调试性**: 动态图更易于调试
3. **现代性**: 支持最新的 Python 和依赖库
4. **兼容性**: 更好的跨平台支持
5. **社区支持**: 活跃的社区和丰富的资源

### 兼容性 / Compatibility

- ✅ Windows 10/11
- ✅ Linux
- ✅ macOS
- ✅ CUDA GPU 加速
- ✅ CPU 训练

---

## 常见问题 / FAQ

### Q1: 旧的 TensorFlow 模型能否加载？
**A:** 不能直接加载。需要重新训练模型或手动转换权重。

### Q2: 训练性能有变化吗？
**A:** PyTorch 和 TensorFlow 的训练性能相近，实际性能取决于硬件配置。

### Q3: 如何选择使用哪个版本？
**A:** 
- 新项目：推荐使用 PyTorch 版本
- 已有训练好的 TF 模型：可以继续使用 TF 版本或重新训练

### Q4: Python 3.5 环境无法运行新代码？
**A:** 需要升级到 Python 3.8 或更高版本。推荐使用 Python 3.8、3.9 或 3.10。

---

## 目录结构 / Directory Structure

```
Emergency-evacuation-Deep-reinforcement-learning/
├── Continuum_Cellspace.py                          # 环境定义 (已更新支持 PyTorch)
├── Evacuation_Continuum_3Exits_Ob_DQN_Fully.py     # TF 训练脚本 (3出口+障碍物)
├── Evacuation_Continuum_3Exits_Ob_DQN_Fully_pytorch.py  # PyTorch 训练脚本 (3出口+障碍物)
├── Evacuation_Continuum_4Exits_DQN_Fully.py        # TF 训练脚本 (4出口)
├── Evacuation_Continuum_4Exits_DQN_Fully_pytorch.py     # PyTorch 训练脚本 (4出口)
├── Evacuation_Continuum_DQN_Fully_test.py          # TF 测试脚本
├── Evacuation_Continuum_DQN_Fully_test_pytorch.py  # PyTorch 测试脚本
├── requirements.txt                                 # PyTorch 依赖
├── pyproject.toml                                   # 项目配置
├── MIGRATION_GUIDE.md                               # 本文件
└── model/                                           # 模型保存目录
    ├── Continuum_3Exits_Ob_DQN_Fully/              # TF 模型
    ├── Continuum_3Exits_Ob_DQN_Fully_pytorch/      # PyTorch 模型
    ├── Continuum_4Exits_DQN_Fully/                 # TF 模型
    └── Continuum_4Exits_DQN_Fully_pytorch/         # PyTorch 模型
```

---

## 技术支持 / Support

如有问题，请参考：
- PyTorch 官方文档: https://pytorch.org/docs/
- PyTorch 教程: https://pytorch.org/tutorials/

---

## 更新日志 / Changelog

### v0.2.0 (PyTorch 版本)
- ✅ 从 TensorFlow 1.15 迁移到 PyTorch 2.0+
- ✅ Python 版本要求从 3.5 升级到 3.8+
- ✅ 重构 DQN 网络实现
- ✅ 更新训练和测试脚本
- ✅ 添加 `step_all_pytorch` 方法到 `Continuum_Cellspace.py`
- ✅ 改进模型保存/加载机制
- ✅ 自动 GPU/CPU 设备检测

### v0.1.0 (TensorFlow 版本)
- 原始 TensorFlow 1.15 实现
