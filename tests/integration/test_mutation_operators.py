#!/usr/bin/env python3
"""
Test Mutation Operators for Local Optima Escape

This script tests the new mutation operators designed to help particles
escape local optima and prevent premature convergence.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin

def test_mutation_operators():
    """Test different mutation operators"""
    print("🧪 Testing Mutation Operators for Local Optima Escape")
    print("=" * 60)
    
    # Test parameters
    n_particles = 20
    dims = 2
    c1 = 2.0
    c2 = 2.0
    w = 0.9
    epochs = 50
    
    # Test different mutation strategies
    mutation_strategies = [
        ('gaussian', 'Gaussian Mutation'),
        ('adaptive_strength', 'Adaptive Strength Mutation'),
        ('escape_local_optima', 'Escape Local Optima'),
        ('diversity_preserving', 'Diversity Preserving'),
        ('opposition_based', 'Opposition-Based Mutation'),
        ('hybrid', 'Hybrid Mutation')
    ]
    
    results = {}
    
    for strategy, name in mutation_strategies:
        print(f"\n🔬 Testing {name}...")
        
        # Test on challenging Rosenbrock function
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=c1, c2=c2, w=w,
            epochs=epochs,
            obj_func=rosenbrock,
            algo='global',
            mutation_strategy=strategy,
            mutation_rate=0.1,
            mutation_strength=0.1
        )
        
        swarm.optimize()
        results[name] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime
        }
        
        print(f"   {name}: Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s")
    
    # Find best mutation strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best mutation strategy: {best_strategy} (cost: {results[best_strategy]['cost']:.6f})")
    
    return results

def test_mutation_vs_no_mutation():
    """Compare PSO with and without mutation"""
    print("\n⚔️ Mutation vs No Mutation Comparison")
    print("=" * 50)
    
    # Test parameters
    n_particles = 15
    dims = 3
    epochs = 40
    obj_func = ackley  # Challenging function with many local optima
    
    # Test without mutation
    print("Testing PSO without mutation...")
    swarm_no_mutation = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global'
    )
    swarm_no_mutation.optimize()
    
    # Test with hybrid mutation
    print("Testing PSO with hybrid mutation...")
    swarm_with_mutation = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global',
        mutation_strategy='hybrid',
        mutation_rate=0.15,
        mutation_strength=0.1
    )
    swarm_with_mutation.optimize()
    
    print(f"\n📊 Results:")
    print(f"   No Mutation:     Cost = {swarm_no_mutation.best_cost:.6f}, Time = {swarm_no_mutation.runtime:.3f}s")
    print(f"   With Mutation:   Cost = {swarm_with_mutation.best_cost:.6f}, Time = {swarm_with_mutation.runtime:.3f}s")
    
    improvement = ((swarm_no_mutation.best_cost - swarm_with_mutation.best_cost) / 
                  swarm_no_mutation.best_cost) * 100
    print(f"   Improvement:     {improvement:.1f}% better with mutation")
    
    return {
        'no_mutation': swarm_no_mutation.best_cost,
        'with_mutation': swarm_with_mutation.best_cost,
        'improvement': improvement
    }

def test_mutation_on_multimodal_functions():
    """Test mutation on functions with multiple local optima"""
    print("\n🎯 Testing Mutation on Multimodal Functions")
    print("=" * 50)
    
    functions = [
        (rastrigin, 'Rastrigin Function'),
        (ackley, 'Ackley Function'),
        (rosenbrock, 'Rosenbrock Function')
    ]
    
    results = {}
    
    for func, name in functions:
        print(f"\nTesting {name}...")
        
        # Without mutation
        swarm_no_mut = Swarm(
            n_particles=20, dims=2, c1=2.0, c2=2.0, w=0.9,
            epochs=30, obj_func=func, algo='global'
        )
        swarm_no_mut.optimize()
        
        # With hybrid mutation
        swarm_mut = Swarm(
            n_particles=20, dims=2, c1=2.0, c2=2.0, w=0.9,
            epochs=30, obj_func=func, algo='global',
            mutation_strategy='hybrid', mutation_rate=0.2, mutation_strength=0.1
        )
        swarm_mut.optimize()
        
        improvement = ((swarm_no_mut.best_cost - swarm_mut.best_cost) / 
                      swarm_no_mut.best_cost) * 100
        
        results[name] = {
            'no_mutation': swarm_no_mut.best_cost,
            'with_mutation': swarm_mut.best_cost,
            'improvement': improvement
        }
        
        print(f"   No Mutation:     {swarm_no_mut.best_cost:.6f}")
        print(f"   With Mutation:   {swarm_mut.best_cost:.6f}")
        print(f"   Improvement:     {improvement:.1f}%")
    
    return results

def test_adaptive_mutation_strength():
    """Test adaptive mutation strength over time"""
    print("\n📈 Testing Adaptive Mutation Strength")
    print("=" * 50)
    
    # Test on challenging function
    swarm = Swarm(
        n_particles=15,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=rastrigin,
        algo='global',
        mutation_strategy='adaptive_strength',
        mutation_rate=0.2,
        mutation_strength=0.05
    )
    
    swarm.optimize()
    
    print(f"✅ Adaptive mutation results:")
    print(f"   Final cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Strategy: Adaptive strength increases over time")
    
    return swarm.best_cost

def test_escape_local_optima():
    """Test escape local optima mutation"""
    print("\n🚀 Testing Escape Local Optima Mutation")
    print("=" * 50)
    
    # Test on function known to have local optima
    swarm = Swarm(
        n_particles=20,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=ackley,
        algo='global',
        mutation_strategy='escape_local_optima',
        mutation_rate=0.1,
        mutation_strength=0.2
    )
    
    swarm.optimize()
    
    print(f"✅ Escape local optima results:")
    print(f"   Final cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Strategy: Strong displacement to escape local optima")
    
    return swarm.best_cost

def test_opposition_based_mutation():
    """Test opposition-based mutation"""
    print("\n🔄 Testing Opposition-Based Mutation")
    print("=" * 50)
    
    # Test opposition-based mutation
    swarm = Swarm(
        n_particles=15,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=sphere,
        algo='global',
        mutation_strategy='opposition_based',
        mutation_rate=0.15,
        mutation_strength=0.1
    )
    
    swarm.optimize()
    
    print(f"✅ Opposition-based mutation results:")
    print(f"   Final cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Strategy: Explore opposite regions of search space")
    
    return swarm.best_cost

def main():
    """Run all mutation operator tests"""
    print("🎯 Mutation Operators Test Suite")
    print("=" * 60)
    print("Testing mutation operators designed to help particles")
    print("escape local optima and prevent premature convergence.")
    
    # Run all tests
    test_mutation_operators()
    test_mutation_vs_no_mutation()
    test_mutation_on_multimodal_functions()
    test_adaptive_mutation_strength()
    test_escape_local_optima()
    test_opposition_based_mutation()
    
    print("\n" + "=" * 60)
    print("🎉 Mutation Operators Testing Complete!")
    print("=" * 60)
    print("\n✨ New Mutation Features:")
    print("✅ Gaussian, adaptive, escape local optima mutations")
    print("✅ Diversity preserving and opposition-based mutations")
    print("✅ Hybrid mutation with multiple strategies")
    print("✅ Stagnation detection and adaptive response")
    print("✅ Strong mutations for stuck particles")
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., mutation_strategy='hybrid', mutation_rate=0.1)")

if __name__ == "__main__":
    main()
