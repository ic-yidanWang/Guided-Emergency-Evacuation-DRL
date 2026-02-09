"""
可视化疏散模拟动画
Visualize evacuation simulation animation
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


def create_animation(case_dir, max_frames=None, step=1):
    """创建疏散动画"""
    
    # 获取所有配置文件并排序
    files = glob.glob(os.path.join(case_dir, 's.*'))
    files.sort(key=lambda x: int(re.search(r's\.(\d+)', x).group(1)))
    
    if max_frames:
        files = files[:max_frames]
    
    # 只显示指定步长的帧
    files = files[::step]
    
    print(f"找到 {len(files)} 个数据文件")
    
    if not files:
        print("没有找到数据文件！")
        return
    
    # 解析第一个文件获取静态元素（出口和障碍物）
    exits, obstacles, _ = parse_config_file(files[0])
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X 位置', fontsize=12)
    ax.set_ylabel('Y 位置', fontsize=12)
    ax.set_title('应急疏散模拟', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 绘制出口（绿色星形，较大）
    if exits is not None and len(exits) > 0:
        ax.scatter(exits[:, 0], exits[:, 1], 
                  c='green', marker='*', s=500, 
                  label='出口', zorder=3, edgecolors='darkgreen', linewidths=2)
    
    # 绘制障碍物（红色方块）
    if obstacles is not None and len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], 
                  c='red', marker='s', s=200, 
                  label='障碍物', zorder=2, alpha=0.7)
    
    # 初始化人员散点图
    agents_scatter = ax.scatter([], [], c='blue', s=100, 
                               label='人员', zorder=1, alpha=0.8)
    
    # 添加统计信息文本
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=10)
    
    ax.legend(loc='upper right', fontsize=10)
    
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
        
        # 更新统计信息
        stats_text.set_text(f'时间步: {step_num}\n剩余人数: {remaining}')
        
        return agents_scatter, stats_text
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(files), 
                        interval=50, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def visualize_single_frame(case_dir, step_num=0):
    """可视化单个时间步的状态"""
    
    filename = os.path.join(case_dir, f's.{step_num}')
    
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return
    
    exits, obstacles, agents = parse_config_file(filename)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X 位置', fontsize=12)
    ax.set_ylabel('Y 位置', fontsize=12)
    ax.set_title(f'应急疏散模拟 - 时间步 {step_num}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 绘制出口
    if exits is not None and len(exits) > 0:
        ax.scatter(exits[:, 0], exits[:, 1], 
                  c='green', marker='*', s=500, 
                  label=f'出口 ({len(exits)})', zorder=3, 
                  edgecolors='darkgreen', linewidths=2)
    
    # 绘制障碍物
    if obstacles is not None and len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], 
                  c='red', marker='s', s=200, 
                  label=f'障碍物 ({len(obstacles)})', zorder=2, alpha=0.7)
    
    # 绘制人员
    if agents is not None and len(agents) > 0:
        ax.scatter(agents[:, 0], agents[:, 1], 
                  c='blue', s=100, 
                  label=f'人员 ({len(agents)})', zorder=1, alpha=0.8)
    
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    
    # 默认使用 Test/case_0
    case_dir = './Test/case_0'
    
    if len(sys.argv) > 1:
        case_dir = sys.argv[1]
    
    if not os.path.exists(case_dir):
        print(f"目录不存在: {case_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("应急疏散可视化工具")
    print("=" * 60)
    print(f"数据目录: {case_dir}")
    print()
    print("选择模式:")
    print("1. 播放动画（所有帧）")
    print("2. 播放动画（每10帧）- 快速预览")
    print("3. 查看单个时间步")
    print()
    
    choice = input("请选择 (1/2/3) [默认=1]: ").strip() or "1"
    
    if choice == "1":
        print("\n正在加载动画...")
        anim = create_animation(case_dir, step=1)
    elif choice == "2":
        print("\n正在加载快速预览...")
        anim = create_animation(case_dir, step=10)
    elif choice == "3":
        step_num = input("请输入时间步编号 [默认=0]: ").strip() or "0"
        visualize_single_frame(case_dir, int(step_num))
    else:
        print("无效选择！")
        sys.exit(1)
