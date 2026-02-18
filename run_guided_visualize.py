"""
Visualize Guided Evacuation with Guide Points

This script creates a guided evacuation scenario with:
- 4 randomly positioned guide points (shown in red)
- Multiple exits
- Agents
- Animated GIF generation
"""

import os
import argparse

# Import utility functions
from evacuation_rl.utils.config_loader import load_config
from evacuation_rl.utils.simulation import setup_environment, run_simulation
from evacuation_rl.utils.visualization import (
    create_animation_from_configs, 
    parse_config_file,
    visualize_evacuation
)


def main(config_path="config/simulation_config.json"):
    """Main function to create and visualize guided evacuation scenario"""
    print("=" * 80)
    print("Creating Guided Evacuation Scenario with Guide Points")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup environment from config
    env = setup_environment(config)
    
    # Save initial configuration
    output_dir = config['simulation']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "initial_state.cfg")
    
    print(f"\nSaving initial configuration to: {config_file}")
    env.save_output(config_file)
    
    # Parse initial configuration to get domain info
    print("\nParsing initial configuration...")
    exits, guides, obstacles, agents, guide_agents, domain = parse_config_file(config_file)
    
    # Save the ORIGINAL obstacle configs for visualization
    obstacle_configs = config.get('obstacles', [])
    
    print(f"\nInitial state:")
    print(f"  - Exits: {len(exits)}")
    print(f"  - Obstacles: {len(obstacle_configs)} (from config)")
    print(f"  - Particles: {len(agents)}")
    
    # Run simulation and save frames
    print("\n" + "=" * 80)
    print("Running Evacuation Simulation")
    print("=" * 80)
    
    # Create frames subdirectory
    frames_dir = os.path.join(output_dir, 'frames')
    
    saved_files = run_simulation(
        env, 
        num_steps=config['simulation']['num_steps'],
        save_interval=config['simulation']['save_interval'],
        output_dir=frames_dir
    )
    
    # Create animated GIF
    print("\n" + "=" * 80)
    print("Creating Animated GIF")
    print("=" * 80)
    
    gif_path = os.path.join(output_dir, config['visualization']['gif_filename'])
    print(f"\nGenerating animation: {gif_path}")
    
    # All circle sizes from config (real simulation values)
    agent_size = config.get('exit_parameters', {}).get('agent_size', 0.18)
    guide_size = config.get('guide_parameters', {}).get('guide_size', 0.25)
    guide_radius = config.get('guide_parameters', {}).get('guide_radius', 1.5)
    create_animation_from_configs(
        config_dir=frames_dir,
        output_file=gif_path,
        fps=config['visualization']['fps'],
        domain=domain,
        obstacle_configs=obstacle_configs,
        agent_size=agent_size,
        guide_size=guide_size,
        guide_radius=guide_radius
    )
    
    # Create trajectory plot
    print("\n" + "=" * 80)
    print("Final Summary")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - Configuration files: {len(saved_files)} frames in {frames_dir}")
    print(f"  - Animated GIF: {gif_path}")
    print("\n" + "=" * 80)
    print("Key Features:")
    print("  - Exits shown as YELLOW stars")
    print("  - Particles (evacuees) shown as BLUE dots")
    print("=" * 80)
    print(f"\n[SUCCESS] Animation ready! Open {gif_path} to view.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run guided evacuation simulation')
    parser.add_argument('--config', '-c', type=str, 
                       default='config/simulation_config.json',
                       help='Path to configuration file (default: config/simulation_config.json)')
    
    args = parser.parse_args()
    main(config_path=args.config)
