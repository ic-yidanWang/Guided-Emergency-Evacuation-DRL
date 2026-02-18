"""
Deep Q-Network (DQN) Implementation

This module provides the neural network architecture and training utilities
for the smart agent evacuation system.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


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


class DQN_4Exit(nn.Module):
    """Deep Q-Network for 4 exits scenario"""
    
    def __init__(self, state_size=4, action_size=8, layer_sizes=[64, 128, 128, 64]):
        super(DQN_4Exit, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ELU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using He initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class Memory:
    """Experience Replay Buffer"""
    
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
    
    def __len__(self):
        return len(self.buffer)


def update_target_network(target_net, main_net, tau=1.0):
    """Soft update of target network parameters
    θ_target = τ*θ_main + (1 - τ)*θ_target
    """
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


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
