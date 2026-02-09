"""
Emergency Evacuation Deep Reinforcement Learning Package

This package contains modules for simulating emergency evacuation using deep reinforcement learning.
"""

__version__ = "0.3.0"

from .environments.cellspace import Cell_Space, Particle, Cell

__all__ = ["Cell_Space", "Particle", "Cell"]
