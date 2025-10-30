#!/usr/bin/env python3
"""
Test script to demonstrate and compare different velocity clamping variations
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin

def test_velocity_clamping_variations():
    """Test and compare different velocity clamping functions"""
    print("🧪 Testing Velocity Clamping Variations")
    print("=" * 60)
    
    # Test parameters
    n_particles = 30
    dims = 2
    c1 = 2.0
    c2 = 2.0
    w = 0.9
    epochs = 100
    velocity_clamp = (-5.12, 5.12)
    
    # Velocity clamping functions to test
    clamping_functions = [
        ('none', 'No Clamping'),
        ('basic', 'Basic Clamping'),
        ('adaptive', 'Adaptive Clamping'),
        ('exponential', 'Exponential Clamping'),
        ('sigmoid', 'Sigmoid Clamping'),
        ('random', 'Random Clamping'),
        ('chaotic', 'Chaotic Clamping'),
        ('soft', 'Soft Clamping'),
        ('hybrid', 'Hybrid Clamping'),
        ('convergence_based', 'Convergence-Based Clamping')
    ]
    
    # Test functions
    test_functions = [
        (sphere, "Sphere Function"),
        (rosenbrock, "Rosenbrock Function"),
        (ackley, "Ackley Function"),
        (rastrigin, "Rastrigin Function")
    ]
    
    results = {}
    
    for func, func_name in test_functions:
        print(f"\n--- Testing {func_name} ---")
        results[func_name] = {}
        
        for clamp_type, clamp_name in clamping_functions:
            print(f"  {clamp_name}: ", end="")
            
            # Create swarm with specific velocity clamping
            swarm = Swarm(
                n_particles=n_particles,
                dims=dims,
                c1=c1,
                c2=c2,
                w=w,
                epochs=epochs,
                obj_func=func,
                algo='global',
                inertia_func='linear',
                velocity_clamp=velocity_clamp,
                velocity_clamp_func=clamp_type
            )
            
            # Run optimization
            swarm.optimize()
            
            # Store results
            results[func_name][clamp_name] = {
                'best_cost': swarm.best_cost,
                'best_pos': swarm.best_pos,
                'runtime': swarm.runtime
            }
            
            print(f"Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VELOCITY CLAMPING COMPARISON SUMMARY")
    print("=" * 60)
    
    for func_name in results:
        print(f"\n{func_name}:")
        sorted_results = sorted(results[func_name].items(), key=lambda x: x[1]['best_cost'])
        for i, (clamp_name, result) in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"  {rank} {clamp_name:20} | Cost: {result['best_cost']:8.6f} | Time: {result['runtime']:6.3f}s")

def benchmark_velocity_clamping_performance():
    """Benchmark performance of different velocity clamping methods"""
    print("\n⚡ Velocity Clamping Performance Benchmark")
    print("=" * 50)
    
    # Test parameters
    n_particles = 50
    dims = 3
    epochs = 200
    
    clamping_functions = [
        ('none', 'No Clamping'),
        ('basic', 'Basic'),
        ('adaptive', 'Adaptive'),
        ('exponential', 'Exponential'),
        ('hybrid', 'Hybrid'),
        ('convergence_based', 'Convergence-Based')
    ]
    
    print(f"Testing on Rastrigin function ({dims}D, {epochs} epochs, {n_particles} particles)")
    print("-" * 50)
    
    for clamp_type, clamp_name in clamping_functions:
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0,
            c2=2.0,
            w=0.9,
            epochs=epochs,
            obj_func=rastrigin,
            algo='global',
            inertia_func='linear',
            velocity_clamp=(-5.12, 5.12),
            velocity_clamp_func=clamp_type
        )
        
        swarm.optimize()
        
        print(f"{clamp_name:18} | Best Cost: {swarm.best_cost:8.4f} | "
              f"Position: [{swarm.best_pos[0]:6.3f}, {swarm.best_pos[1]:6.3f}, {swarm.best_pos[2]:6.3f}] | "
              f"Time: {swarm.runtime:.3f}s")

def test_velocity_clamping_with_inertia():
    """Test velocity clamping combined with different inertia weights"""
    print("\n🔄 Testing Velocity Clamping + Inertia Weight Combinations")
    print("=" * 60)
    
    # Test combinations
    inertia_types = ['linear', 'adaptive', 'exponential']
    clamping_types = ['basic', 'adaptive', 'exponential', 'hybrid']
    
    print("Testing on Sphere Function (2D, 50 epochs, 20 particles)")
    print("-" * 60)
    
    for inertia_type in inertia_types:
        for clamp_type in clamping_types:
            swarm = Swarm(
                n_particles=20,
                dims=2,
                c1=2.0,
                c2=2.0,
                w=0.9,
                epochs=50,
                obj_func=sphere,
                algo='global',
                inertia_func=inertia_type,
                velocity_clamp=(-5, 5),
                velocity_clamp_func=clamp_type
            )
            
            swarm.optimize()
            
            print(f"{inertia_type:12} + {clamp_type:12} | Cost: {swarm.best_cost:8.6f} | Time: {swarm.runtime:.3f}s")

def demonstrate_velocity_clamping_effects():
    """Demonstrate the effects of different velocity clamping methods"""
    print("\n📊 Velocity Clamping Effects Demonstration")
    print("=" * 50)
    
    # Test with a challenging function
    n_particles = 20
    epochs = 50
    
    print("Testing on Rosenbrock Function (known to be challenging)")
    print("Comparing no clamping vs basic clamping vs adaptive clamping")
    print("-" * 50)
    
    clamping_methods = [
        ('none', 'No Clamping'),
        ('basic', 'Basic Clamping'),
        ('adaptive', 'Adaptive Clamping')
    ]
    
    for clamp_type, clamp_name in clamping_methods:
        swarm = Swarm(
            n_particles=n_particles,
            dims=2,
            c1=2.0,
            c2=2.0,
            w=0.9,
            epochs=epochs,
            obj_func=rosenbrock,
            algo='global',
            inertia_func='linear',
            velocity_clamp=(-2, 2),
            velocity_clamp_func=clamp_type
        )
        
        swarm.optimize()
        
        print(f"{clamp_name:15} | Best Cost: {swarm.best_cost:8.4f} | "
              f"Position: [{swarm.best_pos[0]:6.3f}, {swarm.best_pos[1]:6.3f}] | "
              f"Time: {swarm.runtime:.3f}s")

if __name__ == "__main__":
    # Run comprehensive velocity clamping testing
    test_velocity_clamping_variations()
    
    # Run performance benchmark
    benchmark_velocity_clamping_performance()
    
    # Test combinations with inertia weights
    test_velocity_clamping_with_inertia()
    
    # Demonstrate effects
    demonstrate_velocity_clamping_effects()
    
    print("\n" + "=" * 60)
    print("🎉 Velocity Clamping Variations Testing Complete!")
    print("=" * 60)
    print("\nNew velocity clamping functions available:")
    print("✅ none              - No velocity clamping")
    print("✅ basic             - Basic velocity clamping")
    print("✅ adaptive          - Adaptive clamping over time")
    print("✅ exponential       - Exponential clamping decay")
    print("✅ sigmoid           - Sigmoid clamping decay")
    print("✅ random            - Random clamping bounds")
    print("✅ chaotic           - Chaotic clamping")
    print("✅ soft              - Soft clamping with tanh")
    print("✅ hybrid            - Hybrid adaptive/exponential")
    print("✅ convergence_based - Clamping based on convergence")
    print("\nUsage: Set velocity_clamp_func parameter in Swarm constructor")
