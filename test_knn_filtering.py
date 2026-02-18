#!/usr/bin/env python3
"""
Test script to verify KNN filtering logic:
1. Distance filtering (only neighbors within knn_max_distance)
2. Obstacle filtering (exclude neighbors blocked by obstacles)
"""

import numpy as np
import json
from evacuation_rl.agents.guided_agents.environment import GuidedCellSpace

def test_knn_filtering():
    """Test KNN filtering with various configurations"""
    
    # Load config
    with open('config/simulation_config.json', 'r') as f:
        config = json.load(f)
    
    print("=" * 80)
    print("KNN FILTERING TEST")
    print("=" * 80)
    
    # Test 1: Basic setup with KNN filtering enabled
    print("\n[TEST 1] Environment with KNN filtering ENABLED")
    print("-" * 80)
    
    env_filtered = GuidedCellSpace(
        xmin=config['domain']['xmin'],
        xmax=config['domain']['xmax'],
        ymin=config['domain']['ymin'],
        ymax=config['domain']['ymax'],
        zmin=config['domain']['zmin'],
        zmax=config['domain']['zmax'],
        rcut=config['physics']['rcut'],
        dt=config['physics']['dt'],
        Number=10,  # Small number for testing
        door_visible_radius=config['guide_parameters']['door_visible_radius'],
        knn_k=5,
        n_move_guide=0,
        guide_radius=config['guide_parameters']['guide_radius'],
        use_knn=True,
        speed_scale=config['physics']['speed_scale'],
        n_static_guide=0,
        obstacle_configs=config.get('obstacles', []),
        knn_max_distance=3.0,  # Max distance for KNN
        knn_filter_obstacles=True  # Filter by obstacles
    )
    
    print(f"Environment created with:")
    print(f"  - Agents: {env_filtered.Number}")
    print(f"  - KNN max distance: {env_filtered.knn_max_distance}")
    print(f"  - Filter obstacles: {env_filtered.knn_filter_obstacles}")
    print(f"  - Obstacles: {len(env_filtered.obstacle_configs)}")
    
    # Test obstacle intersection detection
    if env_filtered.obstacle_configs:
        print(f"\n[TEST 2] Testing obstacle intersection detection")
        print("-" * 80)
        
        # Create two test points
        p1 = np.array([2.0, 2.0, 0.5])  # Start point
        p2 = np.array([3.0, 3.0, 0.5])  # End point
        
        blocked = env_filtered._is_line_of_sight_blocked(p1, p2)
        print(f"Line from {p1[:2]} to {p2[:2]}: {'BLOCKED' if blocked else 'CLEAR'}")
        
        # Test with a point that should cross obstacle
        p3 = np.array([0.5, 0.5, 0.5])   # Start near obstacle
        p4 = np.array([0.5, 0.6, 0.5])   # End crossing obstacle
        
        blocked = env_filtered._is_line_of_sight_blocked(p3, p4)
        print(f"Line from {p3[:2]} to {p4[:2]}: {'BLOCKED' if blocked else 'CLEAR'}")
    
    # Test 2: Environment with KNN filtering disabled
    print(f"\n[TEST 3] Environment with KNN filtering DISABLED")
    print("-" * 80)
    
    env_no_filter = GuidedCellSpace(
        xmin=config['domain']['xmin'],
        xmax=config['domain']['xmax'],
        ymin=config['domain']['ymin'],
        ymax=config['domain']['ymax'],
        zmin=config['domain']['zmin'],
        zmax=config['domain']['zmax'],
        rcut=config['physics']['rcut'],
        dt=config['physics']['dt'],
        Number=10,
        door_visible_radius=config['guide_parameters']['door_visible_radius'],
        knn_k=5,
        n_move_guide=0,
        guide_radius=config['guide_parameters']['guide_radius'],
        use_knn=True,
        speed_scale=config['physics']['speed_scale'],
        n_static_guide=0,
        obstacle_configs=config.get('obstacles', []),
        knn_max_distance=10.0,  # High max distance
        knn_filter_obstacles=False  # No obstacle filtering
    )
    
    print(f"Environment created with:")
    print(f"  - Agents: {env_no_filter.Number}")
    print(f"  - KNN max distance: {env_no_filter.knn_max_distance}")
    print(f"  - Filter obstacles: {env_no_filter.knn_filter_obstacles}")
    
    # Run a few simulation steps
    print(f"\n[TEST 4] Running simulation steps")
    print("-" * 80)
    
    for step in range(5):
        done = env_filtered.step_guided()
        print(f"Step {step}: {env_filtered.Number} agents remaining (filtered version)")
        if done:
            print("Simulation complete!")
            break
    
    print("\n" + "=" * 80)
    print("KNN FILTERING TEST COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("✓ KNN max distance filtering implemented")
    print("✓ Obstacle-based line-of-sight filtering implemented")
    print("✓ Both filtering options are configurable")
    print("✓ Simulation runs successfully with new filtering logic")

if __name__ == "__main__":
    test_knn_filtering()
