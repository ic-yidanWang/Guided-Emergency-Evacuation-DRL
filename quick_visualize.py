"""
快速可视化疏散模拟 - 自动播放动画
Quick evacuation visualization - Auto-play animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
import re


def parse_config_file(filename):
    """解析配置文件，提取粒子位置和类型信息"""
    exits = []
    obstacles = []
    agents = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    current_type = None
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 识别粒子类型
        if line == 'At':  # 出口
            current_type = 'exit'
            i += 1
            continue
        elif line == 'C' or line == 'Si':  # 障碍物
            current_type = 'obstacle'
            i += 1
            continue
        elif line == 'Br':  # 人员
            current_type = 'agent'
            i += 1
            continue
        
        # 解析坐标数据
        parts = line.split()
        if len(parts) >= 7:
            try:
                x = float(parts[0])
                y = float(parts[1])
                
                if current_type == 'exit':
                    exits.append([x, y])
                elif current_type == 'obstacle':
                    obstacles.append([x, y])
                elif current_type == 'agent':
                    agents.append([x, y])
            except ValueError:
                pass
        
        i += 1
    
    return np.array(exits) if exits else None, \
           np.array(obstacles) if obstacles else None, \
           np.array(agents) if agents else None


def create_quick_animation(case_dir='./Test/case_0', step=5, max_frames=500):
    """创建快速疏散动画预览"""
    
    print(f"Loading data from: {case_dir}")
    
    # 获取所有配置文件并排序
    files = glob.glob(os.path.join(case_dir, 's.*'))
    files.sort(key=lambda x: int(re.search(r's\.(\d+)', x).group(1)))
    
    # 限制帧数和步长
    files = files[:max_frames:step]
    
    print(f"Found {len(files)} data files (sampling every {step} steps)")
    
    if not files:
        print("Error: No data files found!")
        return
    
    # 解析第一个文件获取静态元素
    exits, obstacles, _ = parse_config_file(files[0])
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (Normalized)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (Normalized)', fontsize=13, fontweight='bold')
    ax.set_title('Emergency Evacuation Simulation - Deep Reinforcement Learning', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 绘制背景区域
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, 
                               facecolor='lightgray', alpha=0.2, zorder=0))
    
    # 绘制出口（绿色星形，较大）
    if exits is not None and len(exits) > 0:
        for i, exit_pos in enumerate(exits):
            ax.scatter(exit_pos[0], exit_pos[1], 
                      c='green', marker='*', s=800, 
                      label='Exit' if i == 0 else '', 
                      zorder=3, edgecolors='darkgreen', linewidths=3)
            ax.text(exit_pos[0], exit_pos[1] - 0.05, f'Exit {i+1}', 
                   ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    # 绘制障碍物（红色方块）
    if obstacles is not None and len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], 
                  c='red', marker='s', s=300, 
                  label='Obstacle', zorder=2, alpha=0.8, edgecolors='darkred', linewidths=2)
    
    # 初始化人员散点图
    agents_scatter = ax.scatter([], [], c='blue', s=120, 
                               label='Agent', zorder=1, alpha=0.8, edgecolors='navy')
    
    # 添加统计信息文本框
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', 
                                 alpha=0.9, edgecolor='orange', linewidth=2),
                        fontsize=11, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 初始人数
    _, _, initial_agents = parse_config_file(files[0])
    total_agents = len(initial_agents) if initial_agents is not None else 0
    
    def update(frame_idx):
        """更新动画帧"""
        # 解析当前帧数据
        _, _, agents = parse_config_file(files[frame_idx])
        
        # 更新人员位置
        if agents is not None and len(agents) > 0:
            agents_scatter.set_offsets(agents)
            remaining = len(agents)
        else:
            agents_scatter.set_offsets(np.empty((0, 2)))
            remaining = 0
        
        # 提取时间步
        step_num = int(re.search(r's\.(\d+)', files[frame_idx]).group(1))
        evacuated = total_agents - remaining
        evacuation_rate = (evacuated / total_agents * 100) if total_agents > 0 else 0
        
        # 更新统计信息
        stats_text.set_text(
            f'Time Step: {step_num:5d}\n'
            f'Remaining: {remaining:3d}\n'
            f'Evacuated: {evacuated:3d}\n'
            f'Rate: {evacuation_rate:.1f}%'
        )
        
        return agents_scatter, stats_text
    
    print("Creating animation...")
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(files), 
                        interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    print("Animation window opened. Close window to exit.")
    plt.show()
    
    return anim


if __name__ == '__main__':
    print("=" * 70)
    print(" " * 10 + "Emergency Evacuation Visualization Tool")
    print(" " * 15 + "Deep Reinforcement Learning")
    print("=" * 70)
    print()
    
    # 自动播放动画，每5步采样，最多500帧
    anim = create_quick_animation(case_dir='./Test/case_0', step=5, max_frames=500)
