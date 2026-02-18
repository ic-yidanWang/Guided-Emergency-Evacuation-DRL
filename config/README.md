# 仿真配置文件说明

## 概述
所有仿真参数现在都集中在 JSON 配置文件中，方便修改和管理。

## 使用方法

### 基本使用
```bash
# 使用默认配置文件
uv run .\run_guided_visualize.py

# 使用指定配置文件
uv run .\run_guided_visualize.py --config config/custom_config.json
```

## 配置文件结构

### 1. domain (空间域参数)
定义仿真空间的大小
```json
"domain": {
  "xmin": 0.0,    // X轴最小值
  "xmax": 10.0,   // X轴最大值 - 控制空间宽度
  "ymin": 0.0,    // Y轴最小值
  "ymax": 10.0,   // Y轴最大值 - 控制空间高度
  "zmin": 0.0,    // Z轴最小值
  "zmax": 2.0     // Z轴最大值
}
```

### 2. physics (物理参数)
控制仿真的物理行为
```json
"physics": {
  "rcut": 0.5,        // 邻近半径 - 影响粒子间相互作用距离
  "dt": 0.1,          // 时间步长 - 越小仿真越精确但越慢
  "speed_scale": 1.0  // 速度缩放因子
}
```

### 3. agents (智能体参数)
控制智能体数量
```json
"agents": {
  "number": 80,          // 普通 agent 总数
  "n_move_guide": 4,     // 移动 guide agent 数量（黄色）
  "n_static_guide": 4    // 静态 guide 点数量（红色）
}
```

### 4. guide_parameters (引导参数)
控制 guide 的行为和知识传播
```json
"guide_parameters": {
  "guide_radius": 1.0,           // guide 影响半径 - 在此范围内的普通 agent 会被引导向出口
  "exit_knowledge_radius": 2.0,  // 出口知识范围 - agent 能在这个距离内感知到出口
  "exit_radius": 1.0,            // 出口冲刺范围 - agent 在这个距离内会全力冲向出口
  "use_knn": true,               // 是否使用 k-近邻算法（当看不见出口时跟随人流）
  "knn_k": 5                     // k-近邻的 k 值
}
```

**四段式行为**：
- **Stage A**：是 guide 或被 guide 引导 → 直接冲向出口
- **Stage B**：距离 ≤ `exit_radius` → 全力冲向出口
- **Stage C**：`exit_radius` < 距离 ≤ `exit_knowledge_radius` → 70% 速度向出口移动
- **Stage D**：距离 > `exit_knowledge_radius` → 跟随周围 k 个 agent（KNN）

### 5. exits (出口配置)
定义出口位置（使用归一化坐标 0.0-1.0）
```json
"exits": [
  {"x": 0.5, "y": 1.0, "z": 0.5, "description": "Top center"},
  {"x": 0.5, "y": 0.0, "z": 0.5, "description": "Bottom center"},
  {"x": 0.0, "y": 0.5, "z": 0.5, "description": "Left center"},
  {"x": 1.0, "y": 0.5, "z": 0.5, "description": "Right center"}
]
```

### 6. obstacles (障碍物配置)
定义障碍物位置和大小
```json
"obstacles": [
  {"x": 0.5, "y": 0.5, "z": 0.5, "size": 0.3, "description": "Center obstacle"}
]
```

### 7. simulation (仿真参数)
控制仿真过程
```json
"simulation": {
  "num_steps": 200,           // 最大仿真步数
  "save_interval": 5,         // 每隔几步保存一次
  "output_dir": "output/guided"  // 输出目录
}
```

### 8. visualization (可视化参数)
控制 GIF 生成
```json
"visualization": {
  "fps": 10,                        // 帧率（每秒帧数）
  "gif_filename": "guided_animation.gif"  // GIF 文件名
}
```

## 常见修改场景

### 增加智能体数量
修改 `agents.number`，例如改为 100：
```json
"agents": {
  "number": 100,
  ...
}
```

### 调整空间大小
修改 `domain` 参数，例如改为 20x20：
```json
"domain": {
  "xmax": 20.0,
  "ymax": 20.0,
  ...
}
```

### 调整 guide 的知识传播范围
修改 `guide_parameters.guide_radius`，例如增加 guide 影响范围：
```json
"guide_parameters": {
  "guide_radius": 2.0,
  ...
}
```

### 调整出口感知和冲刺距离
这两个参数配合控制 agent 的流动平缓程度：
```json
"guide_parameters": {
  "exit_knowledge_radius": 3.0,  // 更远处能看到出口
  "exit_radius": 1.0              // 但只有 1.0 范围内才全力冲刺
  // 中间 2.0 范围是 70% 速度的平缓移动区间
}
```

### 添加障碍物
在 `obstacles` 数组中添加：
```json
"obstacles": [
  {"x": 0.3, "y": 0.3, "z": 0.5, "size": 0.2, "description": "Wall 1"},
  {"x": 0.7, "y": 0.7, "z": 0.5, "size": 0.2, "description": "Wall 2"}
]
```

### 修改出口数量
添加或删除 `exits` 数组中的元素：
```json
"exits": [
  {"x": 0.5, "y": 1.0, "z": 0.5, "description": "Single exit"}
]
```

## 提供的配置文件

- `simulation_config.json` - 默认配置
  - 4 出口，80 agents
  - `exit_knowledge_radius=2.0`：agent 能在距出口 2.0 单位时感知到
  - `exit_radius=1.0`：只有距出口 ≤ 1.0 时才全力冲刺
  - 1.0-2.0 之间是平缓移动区

- `large_scale.json` - 大规模仿真
  - 200 agents，6 出口
  - 更大的范围应对拥堵
  - `exit_knowledge_radius=3.0`, `exit_radius=1.5`
  - 更强的 guide 影响（`guide_radius=1.5`）

- `single_exit.json` - 单出口瓶颈场景
  - 60 agents，1 出口
  - 更多的 guide 来组织流量（`n_move_guide=6`）
  - 较小的 `exit_radius=0.8` 防止门前堆积

- `with_obstacles.json` - 带障碍物场景
  - 60 agents，2 出口，3 个障碍物
  - 测试 guide 在复杂环境中的引导效果
  - 平衡的参数设置

## 注意事项

1. **归一化坐标**：出口和障碍物使用 0.0-1.0 的归一化坐标
2. **四段式决策系统**：
   - `guide_radius`：guide 对周围 agent 的直接影响范围
   - `exit_radius`：agent 与出口距离，触发全力冲刺
   - `exit_knowledge_radius`：agent 能感知到出口的范围（需要 > exit_radius）
   - 三层范围的递进关系：`exit_radius < exit_knowledge_radius`
3. **参数平衡建议**：
   - `guide_radius` → 0.5-2.0 之间
   - `exit_radius` → 0.5-1.5 之间
   - `exit_knowledge_radius` → `exit_radius` 的 1.5-3 倍
   - 比例例如：exit_radius=1.0, exit_knowledge_radius=2.0（2倍关系）
4. **时间步长**：`dt` 越小仿真越精确，但运行时间越长
5. **保存间隔**：`save_interval` 太小会产生大量文件，太大会导致动画不流畅
