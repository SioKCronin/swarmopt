#!/usr/bin/env python3
"""
Test script to demonstrate and compare different inertia weight variations
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def test_inertia_variations():
    """Test and compare different inertia weight functions"""
    print("🧪 Testing Inertia Weight Variations")
    print("=" * 60)
    
    # Test parameters
    n_particles = 30
    dims = 2
    c1 = 2.0
    c2 = 2.0
    epochs = 100
    velocity_clamp = (-5, 5)
    
    # Inertia weight functions to test
    inertia_functions = [
        ('constant', 'Constant Inertia'),
        ('linear', 'Linear Decreasing'),
        ('chaotic', 'Chaotic Inertia'),
        ('random', 'Random Inertia'),
        ('adaptive', 'Adaptive Inertia'),
        ('chaotic_random', 'Chaotic-Random'),
        ('exponential', 'Exponential Decreasing'),
        ('sigmoid', 'Sigmoid Decreasing')
    ]
    
    # Test functions
    test_functions = [
        (sphere, "Sphere Function"),
        (rosenbrock, "Rosenbrock Function"),
        (ackley, "Ackley Function")
    ]
    
    results = {}
    
    for func, func_name in test_functions:
        print(f"\n--- Testing {func_name} ---")
        results[func_name] = {}
        
        for inertia_type, inertia_name in inertia_functions:
            print(f"  {inertia_name}: ", end="")
            
            # Create swarm with specific inertia function
            swarm = Swarm(
                n_particles=n_particles,
                dims=dims,
                c1=c1,
                c2=c2,
                w=0.9,  # Base inertia weight
                epochs=epochs,
                obj_func=func,
                algo='global',
                inertia_func=inertia_type,
                velocity_clamp=velocity_clamp,
                w_start=0.9,
                w_end=0.4,
                z=0.5
            )
            
            # Run optimization
            swarm.optimize()
            
            # Store results
            results[func_name][inertia_name] = {
                'best_cost': swarm.best_cost,
                'best_pos': swarm.best_pos,
                'runtime': swarm.runtime
            }
            
            print(f"Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s")
    
    # Print summary
    print("\n" + "=" * 60)
    print("INERTIA WEIGHT COMPARISON SUMMARY")
    print("=" * 60)
    
    for func_name in results:
        print(f"\n{func_name}:")
        sorted_results = sorted(results[func_name].items(), key=lambda x: x[1]['best_cost'])
        for i, (inertia_name, result) in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"  {rank} {inertia_name:20} | Cost: {result['best_cost']:8.6f} | Time: {result['runtime']:6.3f}s")

def visualize_inertia_weights():
    """Visualize how different inertia weights change over time"""
    print("\n📊 Creating Inertia Weight Visualization...")
    
    if HAS_MATPLOTLIB:
        
        epochs = 100
        w_start = 0.9
        w_end = 0.4
        z = 0.5
        
        # Calculate inertia weights over time
        iterations = np.arange(epochs)
        
        # Linear
        linear_w = [w_start - (w_start - w_end) * (i / epochs) for i in iterations]
        
        # Exponential
        exp_w = [w_end + (w_start - w_end) * np.exp(-3 * i / epochs) for i in iterations]
        
        # Sigmoid
        sigmoid_w = [w_end + (w_start - w_end) / (1 + np.exp(2 * i / epochs - 1)) for i in iterations]
        
        # Chaotic (simplified for visualization)
        chaotic_w = []
        z_val = z
        for i in iterations:
            z_val = 4 * z_val * (1 - z_val)
            chaotic_w.append(0.9 * z_val + 0.1)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(iterations, linear_w, 'b-', linewidth=2, label='Linear')
        plt.plot(iterations, exp_w, 'r-', linewidth=2, label='Exponential')
        plt.plot(iterations, sigmoid_w, 'g-', linewidth=2, label='Sigmoid')
        plt.title('Decreasing Inertia Weights')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(iterations, chaotic_w, 'm-', linewidth=2, label='Chaotic')
        plt.title('Chaotic Inertia Weight')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show constant and random for comparison
        plt.subplot(2, 2, 3)
        constant_w = [0.9] * epochs
        random_w = [0.5 + np.random.random() / 2 for _ in range(epochs)]
        plt.plot(iterations, constant_w, 'k-', linewidth=2, label='Constant')
        plt.plot(iterations, random_w, 'orange', linewidth=2, label='Random')
        plt.title('Constant vs Random Inertia')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # All together
        plt.subplot(2, 2, 4)
        plt.plot(iterations, linear_w, 'b-', linewidth=2, label='Linear')
        plt.plot(iterations, exp_w, 'r-', linewidth=2, label='Exponential')
        plt.plot(iterations, sigmoid_w, 'g-', linewidth=2, label='Sigmoid')
        plt.plot(iterations, chaotic_w, 'm-', linewidth=2, label='Chaotic')
        plt.plot(iterations, constant_w, 'k--', linewidth=2, label='Constant')
        plt.title('All Inertia Weight Types')
        plt.xlabel('Iteration')
        plt.ylabel('Inertia Weight')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/siobhan/code/swarmopt/inertia_weights_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as 'inertia_weights_comparison.png'")
        
    else:
        print("❌ Matplotlib not available - skipping visualization")
        print("Install matplotlib to see inertia weight plots: pip install matplotlib")

def benchmark_inertia_performance():
    """Benchmark performance of different inertia weights"""
    print("\n⚡ Inertia Weight Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    n_particles = 50
    dims = 3
    epochs = 200
    
    inertia_functions = [
        ('constant', 'Constant'),
        ('linear', 'Linear'),
        ('exponential', 'Exponential'),
        ('adaptive', 'Adaptive'),
        ('chaotic', 'Chaotic')
    ]
    
    print(f"Testing on Rastrigin function ({dims}D, {epochs} epochs, {n_particles} particles)")
    print("-" * 50)
    
    for inertia_type, inertia_name in inertia_functions:
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0,
            c2=2.0,
            w=0.9,
            epochs=epochs,
            obj_func=rosenbrock,  # Use rosenbrock for harder optimization
            algo='global',
            inertia_func=inertia_type,
            velocity_clamp=(-5.12, 5.12),
            w_start=0.9,
            w_end=0.4
        )
        
        swarm.optimize()
        
        print(f"{inertia_name:12} | Best Cost: {swarm.best_cost:8.4f} | "
              f"Position: [{swarm.best_pos[0]:6.3f}, {swarm.best_pos[1]:6.3f}, {swarm.best_pos[2]:6.3f}] | "
              f"Time: {swarm.runtime:.3f}s")

if __name__ == "__main__":
    # Run comprehensive inertia weight testing
    test_inertia_variations()
    
    # Create visualization
    visualize_inertia_weights()
    
    # Run performance benchmark
    benchmark_inertia_performance()
    
    print("\n" + "=" * 60)
    print("🎉 Inertia Weight Variations Testing Complete!")
    print("=" * 60)
    print("\nNew inertia weight functions available:")
    print("✅ constant    - Constant inertia weight")
    print("✅ linear      - Linear decreasing (default)")
    print("✅ chaotic     - Chaotic inertia weight")
    print("✅ random      - Random inertia weight")
    print("✅ adaptive    - Adaptive based on convergence")
    print("✅ chaotic_random - Combination of chaotic and random")
    print("✅ exponential - Exponential decreasing")
    print("✅ sigmoid     - Sigmoid decreasing")
    print("\nUsage: Set inertia_func parameter in Swarm constructor")
