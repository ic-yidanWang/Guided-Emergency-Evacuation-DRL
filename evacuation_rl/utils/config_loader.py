"""
Configuration loading utilities for evacuation simulation
"""

import os
import json


def load_config(config_path="config/simulation_config.json"):
    """
    Load simulation configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file does not exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from: {config_path}")
    if 'description' in config:
        print(f"  - {config['description']}")
    
    return config
