"""
Example: How to use the refactored evacuation_rl package

This script demonstrates the basic usage of the new modular structure.
"""

import numpy as np
import torch

# Import from new modular structure
from evacuation_rl.environments.cellspace import Cell_Space, Exit, Ob, Ob_size, delta_t
from evacuation_rl.agents.smart_agents.dqn_network import DQN
from evacuation_rl.utils.visualization import visualize_policy


def example_setup_environment():
    """Example: Setting up an evacuation environment"""
    print("=" * 60)
    print("Example 1: Setting up an evacuation environment")
    print("=" * 60)
    
    # Configure exits
    Exit.clear()
    Exit.append(np.array([0.5, 1.0, 0.5]))  # Top exit
    Exit.append(np.array([0.5, 0.0, 0.5]))  # Bottom exit
    Exit.append(np.array([0.0, 0.5, 0.5]))  # Left exit
    Exit.append(np.array([1.0, 0.5, 0.5]))  # Right exit
    
    # Create environment
    env = Cell_Space(0, 10, 0, 10, 0, 2, rcut=1.5, dt=delta_t, Number=10)
    print(f"Created environment with {env.Number} agents")
    print(f"Environment size: {env.L}")
    print(f"Number of exits: {len(env.Exit)}")
    
    # Reset and get initial state
    state = env.reset()
    print(f"Initial state of agent 0: {state}")
    
    return env


def example_load_trained_model():
    """Example: Loading a trained smart agent model"""
    print("\n" + "=" * 60)
    print("Example 2: Loading a trained smart agent model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = DQN(state_size=4, action_size=8, hidden_size=64)
    model.to(device)
    model.eval()
    
    print("Model architecture:")
    print(model)
    
    # Try to load checkpoint (if exists)
    import os
    checkpoint_path = './model/smart_agents_4exits/checkpoint.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['main_qn_state_dict'])
        print(f"\nSuccessfully loaded model from: {checkpoint_path}")
        print(f"Trained for {checkpoint['episode']} episodes")
    else:
        print(f"\nNo checkpoint found at: {checkpoint_path}")
        print("You need to train a model first!")
    
    return model, device


def example_test_agent_action():
    """Example: Testing agent action selection"""
    print("\n" + "=" * 60)
    print("Example 3: Testing agent action selection")
    print("=" * 60)
    
    model, device = example_load_trained_model()
    
    # Create a test state
    test_state = np.array([5.0, 5.0, 0.1, 0.1])  # [x, y, vx, vy]
    print(f"Test state (unnormalized): {test_state}")
    
    # Normalize (assuming environment range 0-10)
    normalized_state = test_state.copy()
    normalized_state[:2] = (normalized_state[:2] - 5.0) / 5.0 - 0.5
    print(f"Test state (normalized): {normalized_state}")
    
    # Get action from model
    with torch.no_grad():
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        q_values = model(state_tensor).cpu().numpy()[0]
    
    action = np.argmax(q_values)
    
    action_names = ['Up', 'Up-Left', 'Left', 'Down-Left', 
                    'Down', 'Down-Right', 'Right', 'Up-Right']
    
    print(f"\nQ-values: {q_values}")
    print(f"Selected action: {action} ({action_names[action]})")


def example_guided_agents():
    """Example: Using the guided agents framework (coming soon)"""
    print("\n" + "=" * 60)
    print("Example 4: Guided Agents Framework (Preview)")
    print("=" * 60)
    
    from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace, GuidedParticle
    
    print("Guided agents framework features:")
    print("  - Only agents near exits know the optimal way out")
    print("  - Other agents follow crowd behavior")
    print("  - Agents move with crowd when velocity threshold is met")
    print("  - Guide agents help direct evacuation")
    print("\nStatus: Framework created, implementation in progress")
    print("See: evacuation_rl/agents/guided_agents/environment.py")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("EVACUATION_RL PACKAGE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    env = example_setup_environment()
    # model, device = example_load_trained_model()
    # example_test_agent_action()
    example_guided_agents()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train a model: python -m evacuation_rl.agents.smart_agents.train_4exits")
    print("  2. Test the model: python -m evacuation_rl.agents.smart_agents.test")
    print("  3. Implement guided agents: see evacuation_rl/agents/guided_agents/")
    print()


if __name__ == "__main__":
    main()
