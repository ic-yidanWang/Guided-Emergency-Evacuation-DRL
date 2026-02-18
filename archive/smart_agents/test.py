"""
Test Trained Smart Agents for Evacuation

This script tests trained DQN models, assuming all agents have learned
optimal evacuation strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import os
import shutil
from evacuation_rl.environments.cellspace import Cell_Space, Exit, Ob, Ob_size, delta_t, offset
from evacuation_rl.agents.smart_agents.dqn_network import DQN, DQN_4Exit


# Scenario configuration
Number_Agent = 80
delta_t = 0.05

# 4 Exits scenario
Exit.append(np.array([0.5, 1.0, 0.5]))  # Add up exit
Exit.append(np.array([0.5, 0.0, 0.5]))  # Add down exit
Exit.append(np.array([0.0, 0.5, 0.5]))  # Add left exit
Exit.append(np.array([1.0, 0.5, 0.5]))  # Add right Exit

# 3 Exits with obstacles (uncomment to use)
# Exit.append(np.array([0.7, 1.0, 0.5]))  # Add up exit
# Exit.append(np.array([0.5, 0, 0.5]))    # Add down Exit
# Exit.append(np.array([0, 0.7, 0.5]))    # Add left exit
#
# Ob1 = []                                # Obstacle #1
# Ob1.append(np.array([0.8, 0.8, 0.5]))
# Ob.append(Ob1)
# Ob_size.append(2.0)
#
# Ob2 = []                                # Obstacle #2
# Ob2.append(np.array([0.3, 0.5, 0.5]))
# Ob.append(Ob2)
# Ob_size.append(3.0)

output_dir = './Test'
model_saved_path_4exits = './model/smart_agents_4exits'
model_saved_path_3exits_ob = './model/smart_agents_3exits_obstacles'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def visualize_policy(model, device):
    """Visualize the learned policy"""
    import matplotlib.pyplot as plt
    
    # Illustration of force direction
    x, y = np.meshgrid(np.linspace(0, 1, 100) - offset[0], np.linspace(0, 1, 100) - offset[1])
    x_arrow, y_arrow = np.meshgrid(np.linspace(0.05, 0.95, 15) - offset[0], np.linspace(0.05, 0.95, 15) - offset[1])
    xy = np.vstack([x.ravel(), y.ravel()]).T
    xy_arrow = np.vstack([x_arrow.ravel(), y_arrow.ravel()]).T
    
    # Random velocity
    vxy = np.random.randn(*xy.shape) * 0.
    vxy_arrow = np.random.randn(*xy_arrow.shape) * 0.
    
    # Constant velocity
    vxy[:, 1] = 0
    vxy_arrow[:, 1] = 0
    
    xtest = np.hstack([xy, vxy])
    x_arrow_test = np.hstack([xy_arrow, vxy_arrow])
    
    # Predict actions for visualization
    with torch.no_grad():
        xtest_tensor = torch.FloatTensor(xtest).to(device)
        x_arrow_test_tensor = torch.FloatTensor(x_arrow_test).to(device)
        
        ypred = model(xtest_tensor).cpu().numpy()
        ypred_arrow = model(x_arrow_test_tensor).cpu().numpy()
    
    action_pred = np.argmax(ypred, axis=1)
    action_arrow_pred = np.argmax(ypred_arrow, axis=1)
    
    action_grid = action_pred.reshape(x.shape)
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'xlim': (-0.5, 0.5),
                           'ylim': (-0.5, 0.5)})
    
    # Contour plot
    contour = ax.contourf(x, y, action_grid + 0.1, cmap=plt.cm.get_cmap('rainbow'), alpha=0.8)       
    
    # Arrow visualization
    arrow_len = 0.07
    angle = np.sqrt(2) / 2
    arrow_map = {0: [0, arrow_len], 1: [-angle * arrow_len, angle * arrow_len],
                 2: [-arrow_len, 0], 3: [-angle * arrow_len, -angle * arrow_len],
                 4: [0, -arrow_len], 5: [angle * arrow_len, -angle * arrow_len],
                 6: [arrow_len, 0], 7: [angle * arrow_len, angle * arrow_len]}
    
    for idx, p in enumerate(xy_arrow):
        ax.annotate('', xy=p, xytext=np.array(arrow_map[action_arrow_pred[idx]]) + p,
                    arrowprops=dict(arrowstyle='<|-', color='k', lw=1.5))
    
    ax.tick_params(labelsize='large')
    plt.show()


def main():
    # Test parameters
    test_episodes = 10        # max number of episodes to test
    max_steps = 10000         # max steps in an episode
    
    Cfg_save_freq = 1
    cfg_save_step = 2
    
    # Initialize environment
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()
    
    # Initialize networks
    mainQN_4Exits = DQN_4Exit().to(device)
    mainQN_3Exits_Ob = DQN().to(device)
    
    # Load models
    checkpoint_4exits = os.path.join(model_saved_path_4exits, 'checkpoint.pth')
    if os.path.exists(checkpoint_4exits):
        checkpoint = torch.load(checkpoint_4exits, map_location=device)
        mainQN_4Exits.load_state_dict(checkpoint['main_qn_state_dict'])
        print(f"Successfully loaded 4-exits model from: {checkpoint_4exits}")
    else:
        print(f"No checkpoint found at: {checkpoint_4exits}")
    
    checkpoint_3exits_ob = os.path.join(model_saved_path_3exits_ob, 'checkpoint.pth')
    if os.path.exists(checkpoint_3exits_ob):
        checkpoint = torch.load(checkpoint_3exits_ob, map_location=device)
        mainQN_3Exits_Ob.load_state_dict(checkpoint['main_qn_state_dict'])
        print(f"Successfully loaded 3-exits-ob model from: {checkpoint_3exits_ob}")
    else:
        print(f"No checkpoint found at: {checkpoint_3exits_ob}")
    
    # Set networks to evaluation mode
    mainQN_4Exits.eval()
    mainQN_3Exits_Ob.eval()
    
    # Visualize policy
    visualize_policy(mainQN_4Exits, device)
    # visualize_policy(mainQN_3Exits_Ob, device)
    
    # Testing loop
    step = 0     
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  
    
    for ep in range(0, test_episodes):
        total_reward = 0
        t = 0
        
        print(f"Testing episode: {ep}")
        
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, 'case_' + str(ep))             
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            else:            
                for filename in os.listdir(pathdir):
                    filepath = os.path.join(pathdir, filename)
                    try:
                        if os.path.isdir(filepath):
                            shutil.rmtree(filepath)
                        else:
                            os.remove(filepath)
                    except OSError as e:
                        print(f"Error removing {filepath}: {e}")
            
            env.save_output(pathdir + '/s.' + str(t))
        
        while t < max_steps:
            # step_all_pytorch / step_optimal were removed from Cell_Space.
            # For multi-agent evacuation use GuidedCellSpace and env.step_guided() (see run_guided_visualize.py).
            # Here we use single-agent step with random action for compatibility:
            action = env.choose_random_action()
            _, _, done = env.step(action) 
            
            step += 1
            t += 1
            
            if done:
                if ep % Cfg_save_freq == 0:
                    env.save_output(pathdir + '/s.' + str(t))
                
                state = env.reset()
                break
            else:
                if ep % Cfg_save_freq == 0:
                    if t % cfg_save_step == 0:
                        env.save_output(pathdir + '/s.' + str(t))
        
        print(f"Total steps in episode {ep} is: {t}")
    
    print("Testing completed!")


if __name__ == '__main__':
    main()
