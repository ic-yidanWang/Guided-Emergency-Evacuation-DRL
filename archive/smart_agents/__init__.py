"""
Smart Agents Module

This module contains implementations where all agents are assumed to be smart:
- They know the nearest exit
- They can make optimal decisions
- Each agent acts independently with full knowledge
"""

from .dqn_network import DQN, Memory, update_target_network, train_dqn

__all__ = ["DQN", "Memory", "update_target_network", "train_dqn"]
