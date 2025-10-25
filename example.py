#!/usr/bin/env python3
"""
SwarmOpt Example - Demonstrating Particle Swarm Optimization

This example shows how to use the SwarmOpt library to optimize various
benchmark functions using different PSO algorithms.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, griewank, rastrigin

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def run_optimization_example():
    """Run a comprehensive example of PSO optimization"""
    
    print("=== SwarmOpt PSO Optimization Example ===\n")
    
    # Define optimization parameters
    n_particles = 30
    dims = 2
    c1 = 2.0  # cognitive weight
    c2 = 2.0  # social weight
    w = 0.9   # inertia weight
    epochs = 100
    velocity_clamp = (-5.12, 5.12)
    
    # Test functions and their optimal solutions
    test_functions = [
        ("Sphere Function", sphere, [0, 0], "Global minimum at origin"),
        ("Rosenbrock Function", rosenbrock, [1, 1], "Global minimum at (1,1)"),
        ("Ackley Function", ackley, [0, 0], "Global minimum at origin"),
        ("Griewank Function", griewank, [0, 0], "Global minimum at origin"),
        ("Rastrigin Function", rastrigin, [0, 0], "Global minimum at origin")
    ]
    
    algorithms = ['global', 'local', 'unified']
    
    results = {}
    
    for func_name, obj_func, optimal_pos, description in test_functions:
        print(f"\n--- Optimizing {func_name} ---")
        print(f"Description: {description}")
        print(f"Optimal position: {optimal_pos}")
        
        results[func_name] = {}
        
        for algo in algorithms:
            print(f"\n  Algorithm: {algo.upper()}")
            
            # Create and run swarm
            swarm = Swarm(
                n_particles=n_particles,
                dims=dims,
                c1=c1,
                c2=c2,
                w=w,
                epochs=epochs,
                obj_func=obj_func,
                algo=algo,
                velocity_clamp=velocity_clamp
            )
            
            # Run optimization
            swarm.optimize()
            
            # Store results
            results[func_name][algo] = {
                'best_cost': swarm.best_cost,
                'best_pos': swarm.best_pos,
                'runtime': swarm.runtime
            }
            
            print(f"    Best cost: {swarm.best_cost:.6f}")
            print(f"    Best position: [{swarm.best_pos[0]:.6f}, {swarm.best_pos[1]:.6f}]")
            print(f"    Runtime: {swarm.runtime:.4f} seconds")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    for func_name in results:
        print(f"\n{func_name}:")
        for algo in algorithms:
            result = results[func_name][algo]
            print(f"  {algo.upper()}: Cost = {result['best_cost']:.6f}, "
                  f"Position = [{result['best_pos'][0]:.3f}, {result['best_pos'][1]:.3f}]")

def visualize_optimization():
    """Create a simple visualization of the optimization process"""
    print("\n=== Creating Optimization Visualization ===")
    
    # Use sphere function for visualization
    n_particles = 20
    dims = 2
    epochs = 50
    
    swarm = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0,
        c2=2.0,
        w=0.9,
        epochs=epochs,
        obj_func=sphere,
        algo='global',
        velocity_clamp=(-5, 5)
    )
    
    # Store particle positions over time
    positions_history = []
    for epoch in range(epochs):
        epoch_positions = []
        for particle in swarm.swarm:
            epoch_positions.append(particle.pos.copy())
        positions_history.append(epoch_positions)
        
        # Update particles
        for particle in swarm.swarm:
            particle.update()
        swarm.update_local_best_pos()
        swarm.update_global_best_pos()
        swarm.update_global_worst_pos()
    
    # Create visualization
    if HAS_MATPLOTLIB:
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot particle trajectories
        for i in range(min(5, n_particles)):  # Show first 5 particles
            x_coords = [pos[i][0] for pos in positions_history]
            y_coords = [pos[i][1] for pos in positions_history]
            ax.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1)
            ax.scatter(x_coords[0], y_coords[0], c='green', s=50, marker='o', label='Start' if i == 0 else "")
            ax.scatter(x_coords[-1], y_coords[-1], c='red', s=50, marker='x', label='End' if i == 0 else "")
        
        # Plot optimal point
        ax.scatter(0, 0, c='gold', s=100, marker='*', label='Optimal (0,0)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('PSO Optimization - Sphere Function\nParticle Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        
        plt.tight_layout()
        plt.savefig('/Users/siobhan/code/swarmopt/optimization_visualization.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'optimization_visualization.png'")
        
    else:
        print("Matplotlib not available - skipping visualization")
        print("Install matplotlib to see particle trajectories: pip install matplotlib")

def benchmark_performance():
    """Benchmark different algorithms on the same function"""
    print("\n=== Performance Benchmark ===")
    
    # Use a more challenging function
    n_particles = 50
    dims = 3
    epochs = 200
    
    algorithms = ['global', 'local', 'unified']
    algorithm_names = ['Global Best PSO', 'Local Best PSO', 'Unified PSO']
    
    print(f"Testing on Rastrigin function (3D, {epochs} epochs, {n_particles} particles)")
    print("-" * 60)
    
    for algo, name in zip(algorithms, algorithm_names):
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0,
            c2=2.0,
            w=0.9,
            epochs=epochs,
            obj_func=rastrigin,
            algo=algo,
            velocity_clamp=(-5.12, 5.12)
        )
        
        swarm.optimize()
        
        print(f"{name:15} | Best Cost: {swarm.best_cost:8.4f} | "
              f"Position: [{swarm.best_pos[0]:6.3f}, {swarm.best_pos[1]:6.3f}, {swarm.best_pos[2]:6.3f}] | "
              f"Time: {swarm.runtime:.3f}s")

if __name__ == "__main__":
    # Run the main example
    run_optimization_example()
    
    # Create visualization
    visualize_optimization()
    
    # Run performance benchmark
    benchmark_performance()
    
    print("\n" + "="*60)
    print("SwarmOpt Example Complete!")
    print("="*60)
    print("\nThe SwarmOpt library is now functional with the following features:")
    print("✓ Multiple PSO algorithms (Global, Local, Unified)")
    print("✓ Various benchmark functions")
    print("✓ Configurable parameters")
    print("✓ Performance tracking")
    print("✓ Comprehensive test suite")
    print("\nYou can now use this library for your optimization needs!")
