# TensorFlow vs PyTorch 代码对比

本文档详细对比了 TensorFlow 1.x 和 PyTorch 实现的主要代码差异。

## 目录
1. [网络定义](#1-网络定义)
2. [前向传播](#2-前向传播)
3. [损失计算](#3-损失计算)
4. [优化器](#4-优化器)
5. [训练循环](#5-训练循环)
6. [模型保存与加载](#6-模型保存与加载)
7. [推理/测试](#7-推理测试)

---

## 1. 网络定义

### TensorFlow 1.x (旧版本)

```python
class DQN:
    def __init__(self, name, learning_rate=0.0001, gamma=0.99,
                 action_size=8, batch_size=20):
        
        self.name = name
        
        # state inputs to the Q-network
        with tf.variable_scope(name):
            
            self.inputs_ = tf.placeholder(tf.float32, [None, 4], name='inputs')  
            self.actions_ = tf.placeholder(tf.int32, [batch_size], name='actions')
            
            with tf.contrib.framework.arg_scope(
                    [tf.contrib.layers.fully_connected],
                    activation_fn=tf.nn.relu,                    
                    weights_initializer=tf.initializers.he_normal()
                    ):
                self.f1 = tf.contrib.layers.fully_connected(self.inputs_, 64)
                self.f2 = tf.contrib.layers.fully_connected(self.f1, 64)
                self.f3 = tf.contrib.layers.fully_connected(self.f2, 64)
                self.f4 = tf.contrib.layers.fully_connected(self.f3, 64)
                self.f5 = tf.contrib.layers.fully_connected(self.f4, 64)
                self.f6 = tf.contrib.layers.fully_connected(self.f5, 64)

            self.output = tf.contrib.layers.fully_connected(self.f6, action_size, 
                                                           activation_fn=None)
```

### PyTorch (新版本)

```python
class DQN(nn.Module):
    """Deep Q-Network implemented in PyTorch"""
    
    def __init__(self, state_size=4, action_size=8, hidden_size=64):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Initialize weights using He initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)
```

**主要区别：**
- ✅ PyTorch 使用面向对象的继承方式 (`nn.Module`)
- ✅ 不需要 `tf.variable_scope` 管理变量
- ✅ 不需要显式定义 `placeholder`
- ✅ 使用 `nn.Sequential` 更简洁
- ✅ 权重初始化更直观

---

## 2. 前向传播

### TensorFlow 1.x

```python
# 需要通过 Session 运行
feed = {mainQN.inputs_: state[np.newaxis, :]}
Qs = sess.run(mainQN.output, feed_dict=feed)[0]
```

### PyTorch

```python
# 直接调用模型
with torch.no_grad():
    state_tensor = torch.FloatTensor(feed_state).unsqueeze(0).to(device)
    q_values = main_qn(state_tensor).cpu().numpy()[0]
```

**主要区别：**
- ✅ PyTorch 不需要 Session
- ✅ 使用 `torch.no_grad()` 禁用梯度计算（推理时）
- ✅ 更直观的 numpy-tensor 转换
- ✅ 自动处理设备迁移 (CPU/GPU)

---

## 3. 损失计算

### TensorFlow 1.x (使用 TRFL)

```python
import trfl

# 在 DQN 类中定义
self.targetQs_ = tf.placeholder(tf.float32, [batch_size, action_size], name='target')
self.reward = tf.placeholder(tf.float32, [batch_size], name="reward")
self.discount = tf.constant(gamma, shape=[batch_size], dtype=tf.float32, name="discount")

# TRFL Q-learning
qloss, q_learning = trfl.qlearning(self.output, self.actions_, self.reward, 
                                   self.discount, self.targetQs_)
self.loss = tf.reduce_mean(qloss)
self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
```

### PyTorch

```python
def train_dqn(main_qn, target_qn, optimizer, batch, gamma, device):
    """Train the DQN using a batch of experiences"""
    
    states = torch.FloatTensor([each[0] for each in batch]).to(device)
    actions = torch.LongTensor([each[1] for each in batch]).to(device)
    rewards = torch.FloatTensor([each[2] for each in batch]).to(device)
    next_states = torch.FloatTensor([each[3] for each in batch]).to(device)
    dones = torch.FloatTensor([each[4] for each in batch]).to(device)
    
    # Get current Q values
    current_q_values = main_qn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Get next Q values from target network
    with torch.no_grad():
        next_q_values = target_qn(next_states).max(1)[0]
        next_q_values[dones == 1] = 0.0
        target_q_values = rewards + gamma * next_q_values
    
    # Compute loss
    loss = nn.MSELoss()(current_q_values, target_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

**主要区别：**
- ✅ PyTorch 不需要第三方库 (trfl)
- ✅ 损失计算更透明、易理解
- ✅ 优化步骤更清晰 (`zero_grad` → `backward` → `step`)
- ✅ 可以更灵活地自定义损失函数

---

## 4. 优化器

### TensorFlow 1.x

```python
# 在定义网络时创建优化器
self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

# 在训练时运行
sess.run([mainQN.loss, mainQN.opt], feed_dict={...})
```

### PyTorch

```python
# 创建独立的优化器
optimizer = optim.Adam(main_qn.parameters(), lr=learning_rate)

# 训练步骤
optimizer.zero_grad()  # 清除梯度
loss.backward()        # 反向传播
optimizer.step()       # 更新参数
```

**主要区别：**
- ✅ PyTorch 优化器是独立对象
- ✅ 训练步骤更明确
- ✅ 更容易实现自定义优化策略

---

## 5. 训练循环

### TensorFlow 1.x

```python
tf.reset_default_graph()

mainQN = DQN(name=name_mainQN, learning_rate=learning_rate, batch_size=batch_size)
targetQN = DQN(name=name_targetQN, learning_rate=learning_rate, batch_size=batch_size)

target_network_update_ops = trfl.update_target_variables(
    targetQN.get_qnetwork_variables(),
    mainQN.get_qnetwork_variables(),
    tau=tau
)

init = tf.global_variables_initializer()
saver = tf.train.Saver() 

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    sess.run(init)
    
    for ep in range(1, train_episodes+1):
        # 训练代码
        loss, _ = sess.run([mainQN.loss, mainQN.opt], feed_dict={...})
        
        # 更新目标网络
        if ep % update_target_every == 0:
            sess.run(target_network_update_ops)
```

### PyTorch

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_qn = DQN(state_size=state_size, action_size=action_size).to(device)
target_qn = DQN(state_size=state_size, action_size=action_size).to(device)

# Copy main network to target network
target_qn.load_state_dict(main_qn.state_dict())

optimizer = optim.Adam(main_qn.parameters(), lr=learning_rate)

for ep in range(start_episode, train_episodes + 1):
    # 训练代码
    if len(memory) >= memory_size and t % train_step == 0:
        batch = memory.sample(batch_size)
        loss = train_dqn(main_qn, target_qn, optimizer, batch, gamma, device)
    
    # 更新目标网络
    if ep % update_target_every == 0:
        update_target_network(target_qn, main_qn, tau=tau)
```

**主要区别：**
- ✅ PyTorch 不需要 Session
- ✅ 自动 GPU 检测和使用
- ✅ 代码更简洁
- ✅ 不需要显式初始化变量

---

## 6. 模型保存与加载

### TensorFlow 1.x

```python
# 保存
saver = tf.train.Saver()
saver.save(sess, os.path.join(model_saved_path, "model.ckpt"), global_step=ep)

# 加载
checkpoint = tf.train.get_checkpoint_state(model_saved_path)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
```

生成的文件：
```
checkpoint
model.ckpt-1000.data-00000-of-00001
model.ckpt-1000.index
model.ckpt-1000.meta
```

### PyTorch

```python
# 保存
checkpoint = {
    'episode': ep,
    'main_qn_state_dict': main_qn.state_dict(),
    'target_qn_state_dict': target_qn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, os.path.join(model_saved_path, f'checkpoint_ep_{ep}.pth'))

# 加载
checkpoint = torch.load(checkpoint_path, map_location=device)
main_qn.load_state_dict(checkpoint['main_qn_state_dict'])
target_qn.load_state_dict(checkpoint['target_qn_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_episode = checkpoint['episode'] + 1
```

生成的文件：
```
checkpoint.pth
checkpoint_ep_1000.pth
checkpoint_ep_2000.pth
```

**主要区别：**
- ✅ PyTorch 使用单个 .pth 文件
- ✅ 可以保存任意 Python 对象
- ✅ 更容易迁移到不同设备
- ✅ 文件结构更简单

---

## 7. 推理/测试

### TensorFlow 1.x

```python
mainQN = DQN_4exit(name='main_qn_4exits', ...)
saver = tf.train.Saver(mainQN.get_qnetwork_variables())

with tf.Session(config=config) as sess:   
    sess.run(init)
    
    checkpoint = tf.train.get_checkpoint_state(model_saved_path)
    if checkpoint:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    
    # 推理
    feed = {mainQN.inputs_: xtest}
    ypred = sess.run(mainQN.output, feed_dict=feed)
```

### PyTorch

```python
mainQN = DQN_4exit().to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
mainQN.load_state_dict(checkpoint['main_qn_state_dict'])

# 设置为评估模式
mainQN.eval()

# 推理
with torch.no_grad():
    xtest_tensor = torch.FloatTensor(xtest).to(device)
    ypred = mainQN(xtest_tensor).cpu().numpy()
```

**主要区别：**
- ✅ PyTorch 有明确的 `train()`/`eval()` 模式
- ✅ 不需要 Session
- ✅ 使用 `torch.no_grad()` 提高推理效率
- ✅ 代码更简洁

---

## 目标网络更新

### TensorFlow 1.x (使用 TRFL)

```python
import trfl

target_network_update_ops = trfl.update_target_variables(
    targetQN.get_qnetwork_variables(),
    mainQN.get_qnetwork_variables(),
    tau=tau
)

# 执行更新
sess.run(target_network_update_ops)
```

### PyTorch

```python
def update_target_network(target_net, main_net, tau=1.0):
    """Soft update of target network parameters
    θ_target = τ*θ_main + (1 - τ)*θ_target
    """
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

# 执行更新
update_target_network(target_qn, main_qn, tau=tau)
```

**主要区别：**
- ✅ PyTorch 实现更透明
- ✅ 不需要第三方库
- ✅ 更容易自定义更新策略

---

## 经验回放缓冲区

### TensorFlow 1.x & PyTorch (相似)

两者的经验回放实现基本相同：

```python
class Memory:
    def __init__(self, max_size=500):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return self.buffer
        
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]
```

---

## 性能对比总结

| 特性 | TensorFlow 1.x | PyTorch |
|------|---------------|---------|
| **代码简洁性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **易于调试** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **灵活性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **学习曲线** | 陡峭 | 平缓 |
| **社区支持** | 逐渐减少 | 非常活跃 |
| **文档质量** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **模型部署** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **训练速度** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 推荐

**对于新项目，强烈推荐使用 PyTorch 版本**，原因：

1. ✅ **代码更简洁**: 减少约 30-40% 的代码量
2. ✅ **更易调试**: 动态图，可以像普通 Python 代码一样调试
3. ✅ **更好的生态**: PyTorch 在研究社区更流行
4. ✅ **现代化**: 支持最新的 Python 特性和库
5. ✅ **更好的文档**: PyTorch 文档更清晰、示例更丰富

**如果有以下情况，可以继续使用 TensorFlow 版本**：

- 已经有训练好的模型，且不想重新训练
- 团队已经熟悉 TensorFlow 1.x
- 需要使用特定的 TensorFlow 工具或库

---

## 迁移建议

从 TensorFlow 迁移到 PyTorch 的步骤：

1. ✅ 更新 Python 到 3.8+
2. ✅ 安装 PyTorch: `pip install torch`
3. ✅ 使用新的 PyTorch 脚本
4. ✅ 重新训练模型（TF 模型无法直接转换）
5. ✅ 验证结果一致性

转换工作量估计：
- 小型项目 (< 1000 行): 1-2 天
- 中型项目 (1000-5000 行): 3-5 天
- 大型项目 (> 5000 行): 1-2 周

---

**希望这个对比对你有帮助！** 🚀
