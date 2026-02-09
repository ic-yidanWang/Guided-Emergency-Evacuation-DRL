"""
Visualization utilities for evacuation simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_policy(model, device, offset=np.array([0.5, 0.5])):
    """
    Visualize the learned evacuation policy
    
    Args:
        model: Trained DQN model
        device: PyTorch device (CPU or CUDA)
        offset: Position offset for normalization
    """
    # Create grid for visualization
    x, y = np.meshgrid(np.linspace(0, 1, 100) - offset[0], 
                       np.linspace(0, 1, 100) - offset[1])
    x_arrow, y_arrow = np.meshgrid(np.linspace(0.05, 0.95, 15) - offset[0], 
                                     np.linspace(0.05, 0.95, 15) - offset[1])
    xy = np.vstack([x.ravel(), y.ravel()]).T
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T
    
    # Create velocity components (zero for static visualization)
    vxy = np.zeros_like(xy)
    vxy_arrow = np.zeros_like(xy_arrow)
    
    xtest = np.hstack([xy, vxy])
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])
    
    # Predict actions
    import torch
    with torch.no_grad():
        xtest_tensor = torch.FloatTensor(xtest).to(device)
        x_arrow_test_tensor = torch.FloatTensor(x_arrow_test).to(device)
        
        ypred = model(xtest_tensor).cpu().numpy()
        ypred_arrow = model(x_arrow_test_tensor).cpu().numpy()
    
    action_pred = np.argmax(ypred, axis=1)
    action_arrow_pred = np.argmax(ypred_arrow, axis=1)
    
    action_grid = action_pred.reshape(x.shape)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), 
                           subplot_kw={'xlim': (-0.5, 0.5), 'ylim': (-0.5, 0.5)})
    
    # Contour plot of actions
    contour = ax.contourf(x, y, action_grid + 0.1, 
                          cmap=plt.cm.get_cmap('rainbow'), alpha=0.8)
    
    # Arrow visualization
    arrow_len = 0.07
    angle = np.sqrt(2) / 2
    arrow_map = {
        0: [0, arrow_len],                              # Up
        1: [-angle * arrow_len, angle * arrow_len],     # Up-Left
        2: [-arrow_len, 0],                             # Left
        3: [-angle * arrow_len, -angle * arrow_len],    # Down-Left
        4: [0, -arrow_len],                             # Down
        5: [angle * arrow_len, -angle * arrow_len],     # Down-Right
        6: [arrow_len, 0],                              # Right
        7: [angle * arrow_len, angle * arrow_len]       # Up-Right
    }
    
    for idx, p in enumerate(xy_arrow):
        ax.annotate('', xy=p, xytext=np.array(arrow_map[action_arrow_pred[idx]]) + p,
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))
    
    ax.set_xlabel('X Position (normalized)', fontsize=12)
    ax.set_ylabel('Y Position (normalized)', fontsize=12)
    ax.set_title('Learned Evacuation Policy', fontsize=14)
    ax.tick_params(labelsize='large')
    
    plt.colorbar(contour, ax=ax, label='Action ID')
    plt.tight_layout()
    
    return fig, ax


def plot_training_stats(episodes, losses, steps, save_path=None):
    """
    Plot training statistics
    
    Args:
        episodes: List of episode numbers
        losses: List of loss values
        steps: List of steps per episode
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot loss
    ax1.plot(episodes, losses)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Episodes')
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(episodes, steps)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Exit')
    ax2.set_title('Steps per Episode')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def create_animation_from_configs(config_dir, output_file='evacuation.gif', fps=10):
    """
    Create animation from saved configuration files
    
    Args:
        config_dir: Directory containing configuration files
        output_file: Output animation file path
        fps: Frames per second
    """
    # TODO: Implement animation creation from saved configs
    # This would read the s.* files and create an animated visualization
    pass


def plot_evacuation_trajectory(positions, exits, obstacles=None, save_path=None):
    """
    Plot evacuation trajectories
    
    Args:
        positions: List of position arrays for each time step
        exits: Array of exit positions
        obstacles: Optional list of obstacle positions
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot trajectories
    positions = np.array(positions)
    for i in range(positions.shape[1]):
        ax.plot(positions[:, i, 0], positions[:, i, 1], 'b-', alpha=0.5, linewidth=0.5)
        ax.plot(positions[0, i, 0], positions[0, i, 1], 'go', markersize=5, label='Start' if i == 0 else '')
        ax.plot(positions[-1, i, 0], positions[-1, i, 1], 'ro', markersize=5, label='End' if i == 0 else '')
    
    # Plot exits
    for e in exits:
        ax.plot(e[0], e[1], 'y*', markersize=20, markeredgecolor='k', markeredgewidth=1.5)
    
    # Plot obstacles
    if obstacles:
        for ob in obstacles:
            ax.plot(ob[0], ob[1], 'ks', markersize=15)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Evacuation Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax
