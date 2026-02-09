"""
Emergency Evacuation Deep Reinforcement Learning - Main Entry Point

This is the main entry point for the refactored project.

Project Structure:
    evacuation_rl/
        ├── environments/       - Simulation environments (cellspace)
        ├── agents/
        │   ├── smart_agents/   - Agents assuming everyone is smart
        │   └── guided_agents/  - Agents with guide assistance (realistic)
        └── utils/              - Visualization and utilities

Usage:
    Training smart agents (3 exits with obstacles):
        python -m evacuation_rl.agents.smart_agents.train_3exits_obstacles
    
    Training smart agents (4 exits):
        python -m evacuation_rl.agents.smart_agents.train_4exits
    
    Testing smart agents:
        python -m evacuation_rl.agents.smart_agents.test

    For guided agents (future work):
        python -m evacuation_rl.agents.guided_agents.train
        python -m evacuation_rl.agents.guided_agents.test
"""

import sys


def print_usage():
    """Print usage information"""
    print("=" * 80)
    print("Emergency Evacuation Deep Reinforcement Learning")
    print("Refactored PyTorch Implementation")
    print("=" * 80)
    print()
    print("Project Structure:")
    print("  evacuation_rl/")
    print("    ├── environments/       - Simulation environments (cellspace)")
    print("    ├── agents/")
    print("    │   ├── smart_agents/   - All agents are smart (know exits)")
    print("    │   └── guided_agents/  - Realistic agents need guidance")
    print("    └── utils/              - Visualization and utilities")
    print()
    print("=" * 80)
    print("Available Commands:")
    print("=" * 80)
    print()
    print("SMART AGENTS (assumes everyone knows optimal exit):")
    print("-" * 80)
    print("  Train 3-exits with obstacles:")
    print("    python -m evacuation_rl.agents.smart_agents.train_3exits_obstacles")
    print()
    print("  Train 4-exits:")
    print("    python -m evacuation_rl.agents.smart_agents.train_4exits")
    print()
    print("  Test trained models:")
    print("    python -m evacuation_rl.agents.smart_agents.test")
    print()
    print("=" * 80)
    print("GUIDED AGENTS (realistic scenario - coming soon):")
    print("-" * 80)
    print("  Only agents near exits know the way out")
    print("  Other agents follow crowd behavior")
    print("  Guide agents help direct evacuation")
    print()
    print("  Implementation in progress...")
    print()
    print("=" * 80)
    print()
    print("For more information, see README.md and documentation files")
    print()


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            scenario = sys.argv[2] if len(sys.argv) > 2 else "4exits"
            if scenario == "3exits":
                from evacuation_rl.agents.smart_agents import train_3exits_obstacles
                train_3exits_obstacles.main()
            elif scenario == "4exits":
                from evacuation_rl.agents.smart_agents import train_4exits
                train_4exits.main()
            else:
                print(f"Unknown scenario: {scenario}")
                print_usage()
        
        elif command == "test":
            from evacuation_rl.agents.smart_agents import test
            test.main()
        
        elif command == "help":
            print_usage()
        
        else:
            print(f"Unknown command: {command}")
            print_usage()
    
    else:
        print_usage()


if __name__ == "__main__":
    main()
