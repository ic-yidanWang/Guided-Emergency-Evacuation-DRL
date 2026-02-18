"""
Emergency Evacuation Deep Reinforcement Learning - Main Entry Point

This is the main entry point for the project.

Project Structure:
    evacuation_rl/
        ├── environments/       - Simulation environments (cellspace)
        ├── agents/
        │   └── guided_agents/  - Realistic agents with limited visibility
        └── utils/              - Visualization and utilities

Usage:
    Run guided evacuation simulation:
        python run_guided_visualize.py
    
    For help:
        python -m evacuation_rl --help

Note:
    Smart agents code (theoretical baseline) has been archived.
    See archive/README.md for details.
"""

import sys


def print_usage():
    """Print usage information"""
    print("=" * 80)
    print("Emergency Evacuation Deep Reinforcement Learning")
    print("PyTorch Implementation")
    print("=" * 80)
    print()
    print("Project Structure:")
    print("  evacuation_rl/")
    print("    ├── environments/       - Simulation environments (cellspace)")
    print("    ├── agents/")
    print("    │   └── guided_agents/  - Realistic agents with limited visibility")
    print("    └── utils/              - Visualization and utilities")
    print()
    print("  archive/                  - Deprecated smart agents code")
    print()
    print("=" * 80)
    print("Available Commands:")
    print("=" * 80)
    print()
    print("GUIDED EVACUATION SIMULATION:")
    print("-" * 80)
    print("  Run simulation with visualization:")
    print("    python run_guided_visualize.py")
    print("    uv run python run_guided_visualize.py")
    print()
    print("  Features:")
    print("    - Distance-based exit visibility")
    print("    - Crowd-following behavior")
    print("    - Wall and obstacle detection")
    print("    - Guide agents (stationary and mobile)")
    print()
    print("=" * 80)
    print("CONFIGURATION:")
    print("-" * 80)
    print("  Edit files in config/ folder:")
    print("    config/simulation_config.json")
    print("    config/single_exit.json")
    print("    config/with_obstacles.json")
    print("    config/large_scale.json")
    print()
    print("=" * 80)
    print("FUTURE WORK:")
    print("-" * 80)
    print("  Guide agent training using RL (in development)")
    print("  Multi-agent coordination")
    print("  Complex evacuation scenarios")
    print()
    print("=" * 80)
    print()
    print("For more information:")
    print("  - README.md - Main documentation")
    print("  - archive/README.md - Info on deprecated smart agents")
    print()


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command in ["help", "--help", "-h"]:
            print_usage()
        
        else:
            print(f"Unknown command: {command}")
            print()
            print_usage()
    
    else:
        print_usage()


if __name__ == "__main__":
    main()
