"""
Performance test to measure the speed improvement from obstacle optimization.
"""
import time
import numpy as np
from evacuation_rl.environments.cellspace import Cell_Space, agent_size

def test_step_performance(num_steps=50):
    """Test the performance of stepping through the simulation."""
    print("="*80)
    print("Performance Test - Obstacle Calculation Optimization")
    print("="*80)
    
    # Create environment
    print("\nInitializing test environment...")
    env = Cell_Space(
        xmin=0.0, xmax=10.0,
        ymin=0.0, ymax=10.0,
        zmin=0.0, zmax=2.0,
        rcut=0.5,
        dt=0.1,
        Number=80
    )
    
    # Add obstacles manually to test
    # Circle obstacle
    env.Ob.append([np.array([2.0, 2.0, 0.5])])
    env.Ob_size.append(0.3)
    env.Ob_type.append('circle')
    env.Ob_params.append({})
    
    # Rectangle obstacle (optimized - no point grid!)
    env.Ob.append([np.array([5.0, 5.0, 0.5])])
    env.Ob_size.append(0.05)
    env.Ob_type.append('rectangle')
    env.Ob_params.append({
        'center': np.array([5.0, 5.0, 0.5]),
        'width': 2.0,
        'height': 1.0
    })
    
    # Another rectangle
    env.Ob.append([np.array([7.0, 7.0, 0.5])])
    env.Ob_size.append(0.05)
    env.Ob_type.append('rectangle')
    env.Ob_params.append({
        'center': np.array([7.0, 7.0, 0.5]),
        'width': 1.5,
        'height': 1.5
    })
    
    print(f"\nTest configuration:")
    print(f"  - Agents: {env.Number}")
    print(f"  - Obstacles: {len(env.Ob)}")
    print(f"    * 1 circle")
    print(f"    * 2 rectangles (optimized - direct geometry, no point grid)")
    print(f"  - Simulation steps: {num_steps}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        env.Zero_acc()
        env.region_confine()
        env.loop_cells()
        env.loop_neighbors()
        env.Integration(1)
        env.Integration(0)
        env.move_particles()
    
    # Performance test
    print("\nRunning performance test...")
    start_time = time.time()
    
    for step in range(num_steps):
        env.Zero_acc()
        env.region_confine()  # This is where obstacle forces are calculated
        env.loop_cells()
        env.loop_neighbors()
        env.Integration(1)
        env.Integration(0)
        env.move_particles()
        
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"  Step {step+1:3d}: {steps_per_sec:.2f} steps/sec")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print("Performance Results")
    print("="*80)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per step: {total_time/num_steps*1000:.2f} ms")
    print(f"Steps per second: {num_steps/total_time:.2f}")
    print("\nOptimization Summary:")
    print("  ✓ Rectangles use direct geometry calculation (no point grid)")
    print("  ✓ Distance threshold applied to skip far obstacles")
    print("  ✓ Early rejection for particles far from obstacles")
    print("="*80)

if __name__ == "__main__":
    test_step_performance(num_steps=50)
