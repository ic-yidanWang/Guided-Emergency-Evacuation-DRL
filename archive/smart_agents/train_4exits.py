"""
Training Smart Agents for Evacuation at 4 Exits

This script trains DQN agents assuming all agents are smart and can make
optimal decisions based on their observations.
"""

import numpy as np
import torch
import torch.optim as optim
import os
from evacuation_rl.environments.cellspace import Cell_Space, Exit, delta_t, cfg_save_step
from evacuation_rl.agents.smart_agents.dqn_network import DQN, Memory, update_target_network, train_dqn


# Scenario configuration
Number_Agent = 1
Exit.append(np.array([0.5, 1.0, 0.5]))   # Add Up exit
Exit.append(np.array([0.5, 0, 0.5]))     # Add Down Exit
Exit.append(np.array([0, 0.5, 0.5]))     # Add Left exit
Exit.append(np.array([1.0, 0.5, 0.5]))   # Add Right Exit

output_dir = './output'
model_saved_path = './model'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
            
if not os.path.isdir(model_saved_path):
    os.mkdir(model_saved_path)
    
output_dir = output_dir + '/smart_agents_4exits'
model_saved_path = model_saved_path + '/smart_agents_4exits'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # Training hyperparameters
    train_episodes = 10000        # max number of episodes to learn from
    max_steps = 10000             # max steps in an episode
    gamma = 0.999                 # future reward discount

    explore_start = 1.0           # exploration probability at start
    explore_stop = 0.1            # minimum exploration probability 
    decay_percentage = 0.5          
    decay_rate = 4 / decay_percentage  # exploration decay rate
            
    # Network parameters
    learning_rate = 1e-4          # Q-network learning rate 
    
    # Memory parameters
    memory_size = 1000            # memory capacity
    batch_size = 50               # experience mini-batch size
    
    # Target QN
    update_target_every = 1       # target update frequency
    tau = 0.1                     # target update factor
    save_step = 1000              # steps to save the model
    train_step = 1                # steps to train the model
    
    Cfg_save_freq = 100           # Cfg save frequency (episode)
    
    # Initialize environment
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=Number_Agent)
    state = env.reset()
    
    # Initialize memory
    memory = Memory(max_size=memory_size)
    
    # Initialize networks
    action_size = 8
    state_size = 4
    
    main_qn = DQN(state_size=state_size, action_size=action_size).to(device)
    target_qn = DQN(state_size=state_size, action_size=action_size).to(device)
    
    # Copy main network to target network
    target_qn.load_state_dict(main_qn.state_dict())
    
    # Initialize optimizer
    optimizer = optim.Adam(main_qn.parameters(), lr=learning_rate)
    
    # Create model directory
    if not os.path.isdir(model_saved_path):
        os.mkdir(model_saved_path)
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(model_saved_path, 'checkpoint.pth')
    start_episode = 1
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        main_qn.load_state_dict(checkpoint['main_qn_state_dict'])
        target_qn.load_state_dict(checkpoint['target_qn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print(f"Successfully loaded checkpoint from episode {checkpoint['episode']}")
        
        # Clean up old checkpoints
        print("Removing old checkpoint files")
        for filename in os.listdir(model_saved_path):
            if filename.startswith('checkpoint_ep_'):
                filepath = os.path.join(model_saved_path, filename)
                try:
                    os.remove(filepath)
                except OSError as e:
                    print(f"Error removing {filepath}: {e}")
        print("Done")
    else:
        print("No checkpoint found. Starting from initialization")
    
    # Training loop
    step = 0     
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    for ep in range(start_episode, train_episodes + 1):
        total_reward = 0
        t = 0
        loss = 0.0
        
        if ep % Cfg_save_freq == 0:
            pathdir = os.path.join(output_dir, 'case_' + str(ep))             
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            env.save_output(pathdir + '/s.' + str(t))
        
        while t < max_steps:
            # Epsilon-greedy action selection
            epsilon = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * ep / train_episodes) 
            feed_state = np.array(state)
            feed_state[:2] = env.Normalization_XY(feed_state[:2])
            
            if np.random.rand() < epsilon:   
                # Get random action
                action = env.choose_random_action()                   
            else:
                # Get action from Q-network
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(feed_state).unsqueeze(0).to(device)
                    q_values = main_qn(state_tensor).cpu().numpy()[0]
                    
                    action_list = [idx for idx, val in enumerate(q_values) if val == np.max(q_values)]                    
                    action = np.random.choice(action_list)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            step += 1
            t += 1
            
            feed_next_state = np.array(next_state)
            feed_next_state[:2] = env.Normalization_XY(feed_next_state[:2])               
            
            memory.add((feed_state, action, reward, feed_next_state, done))
            
            if done:
                # Start new episode
                if ep % Cfg_save_freq == 0:
                    env.save_output(pathdir + '/s.' + str(t))
                state = env.reset()
                break
            else:
                state = next_state
                
                if ep % Cfg_save_freq == 0:
                    if t % cfg_save_step == 0:
                        env.save_output(pathdir + '/s.' + str(t))
        
            # Train the network
            if len(memory) >= memory_size and t % train_step == 0:
                batch = memory.sample(batch_size)
                loss = train_dqn(main_qn, target_qn, optimizer, batch, gamma, device)
        
        # Print progress
        if len(memory) >= memory_size:
            print(f"Episode: {ep}, Loss: {loss:.6f}, Steps: {t}, Epsilon: {epsilon:.4f}")
        
        # Save model checkpoint
        if ep % save_step == 0:
            checkpoint = {
                'episode': ep,
                'main_qn_state_dict': main_qn.state_dict(),
                'target_qn_state_dict': target_qn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_saved_path, f'checkpoint_ep_{ep}.pth'))
            torch.save(checkpoint, checkpoint_path)  # Also save as latest checkpoint
        
        # Update target network
        if ep % update_target_every == 0:
            update_target_network(target_qn, main_qn, tau=tau)
    
    # Save final model
    final_checkpoint = {
        'episode': train_episodes,
        'main_qn_state_dict': main_qn.state_dict(),
        'target_qn_state_dict': target_qn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(model_saved_path, f'checkpoint_ep_{train_episodes}.pth'))
    print("Training completed!")


if __name__ == '__main__':
    main()
