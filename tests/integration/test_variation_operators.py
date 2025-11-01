#!/usr/bin/env python3
"""
Test Variation Operators for Local Optima Escape

This script tests the new variation operators designed to help particles
escape local optima and prevent premature convergence.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin

def test_variation_operators():
    """Test different variation operators"""
    print("🧪 Testing Variation Operators for Local Optima Escape")
    print("=" * 60)
    
    # Test parameters
    n_particles = 20
    dims = 2
    c1 = 2.0
    c2 = 2.0
    w = 0.9
    epochs = 50
    
    # Test different variation strategies
    variation_strategies = [
        ('gaussian', 'Gaussian Variation'),
        ('adaptive_strength', 'Adaptive Strength Variation'),
        ('escape_local_optima', 'Escape Local Optima'),
        ('diversity_preserving', 'Diversity Preserving'),
        ('opposition_based', 'Opposition-Based Variation'),
        ('hybrid', 'Hybrid Variation')
    ]
    
    results = {}
    
    for strategy, name in variation_strategies:
        print(f"\n🔬 Testing {name}...")
        
        # Test on challenging Rosenbrock function
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=c1, c2=c2, w=w,
            epochs=epochs,
            obj_func=rosenbrock,
            algo='global',
            variation_strategy=strategy,
            variation_rate=0.1,
            variation_strength=0.1
        )
        
        swarm.optimize()
        results[name] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime
        }
        
        print(f"   {name}: Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s")
    
    # Find best variation strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best variation strategy: {best_strategy} (cost: {results[best_strategy]['cost']:.6f})")
    
    return results

def test_variation_vs_no_variation():
    """Compare PSO with and without variation"""
    print("\n⚔️ Variation vs No Variation Comparison")
    print("=" * 50)
    
    # Test parameters
    n_particles = 15
    dims = 3
    epochs = 40
    obj_func = ackley  # Challenging function with many local optima
    
    # Test without variation
    print("Testing PSO without variation...")
    swarm_no_variation = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global'
    )
    swarm_no_variation.optimize()
    
    # Test with hybrid variation
    print("Testing PSO with hybrid variation...")
    swarm_with_variation = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global',
        variation_strategy='hybrid',
        variation_rate=0.15,
        variation_strength=0.1
    )
    swarm_with_variation.optimize()
    
    print(f"\n📊 Results:")
    print(f"   No Variation:     Cost = {swarm_no_variation.best_cost:.6f}, Time = {swarm_no_variation.runtime:.3f}s")
    print(f"   With Variation:   Cost = {swarm_with_variation.best_cost:.6f}, Time = {swarm_with_variation.runtime:.3f}s")
    
    improvement = ((swarm_no_variation.best_cost - swarm_with_variation.best_cost) / 
                  swarm_no_variation.best_cost) * 100
    print(f"   Improvement:     {improvement:.1f}% better with variation")
    
    return {
        'no_variation': swarm_no_variation.best_cost,
        'with_variation': swarm_with_variation.best_cost,
        'improvement': improvement
    }

def test_variation_on_multimodal_functions():
    """Test variation on functions with multiple local optima"""
    print("\n🎯 Testing Variation on Multimodal Functions")
    print("=" * 50)
    
    functions = [
        (rastrigin, 'Rastrigin Function'),
        (ackley, 'Ackley Function'),
        (rosenbrock, 'Rosenbrock Function')
    ]
    
    results = {}
    
    for func, name in functions:
        print(f"\nTesting {name}...")
        
        # Without variation
        swarm_no_mut = Swarm(
            n_particles=20, dims=2, c1=2.0, c2=2.0, w=0.9,
            epochs=30, obj_func=func, algo='global'
        )
        swarm_no_mut.optimize()
        
        # With hybrid variation
        swarm_mut = Swarm(
            n_particles=20, dims=2, c1=2.0, c2=2.0, w=0.9,
            epochs=30, obj_func=func, algo='global',
            variation_strategy='hybrid', variation_rate=0.2, variation_strength=0.1
        )
        swarm_mut.optimize()
        
        improvement = ((swarm_no_mut.best_cost - swarm_mut.best_cost) / 
                      swarm_no_mut.best_cost) * 100
        
        results[name] = {
            'no_variation': swarm_no_mut.best_cost,
            'with_variation': swarm_mut.best_cost,
            'improvement': improvement
        }
        
        print(f"   No Variation:     {swarm_no_mut.best_cost:.6f}")
        print(f"   With Variation:   {swarm_mut.best_cost:.6f}")
        print(f"   Improvement:     {improvement:.1f}%")
    
    return results

def test_adaptive_variation_strength():
    """Test adaptive variation strength over time"""
    print("\n📈 Testing Adaptive Variation Strength")
    print("=" * 50)
    
    # Test on challenging function
    swarm = Swarm(
        n_particles=15,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=rastrigin,
        algo='global',
        variation_strategy='adaptive_strength',
        variation_rate=0.2,
        variation_strength=0.05
    )
    
    swarm.optimize()
    
    print(f"✅ Adaptive variation results:")
    print(f"   Final cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Strategy: Adaptive strength increases over time")
    
    return swarm.best_cost

def test_escape_local_optima():
    """Test escape local optima variation"""
    print("\n🚀 Testing Escape Local Optima Variation")
    print("=" * 50)
    
    # Test on function known to have local optima
    swarm = Swarm(
        n_particles=20,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=ackley,
        algo='global',
        variation_strategy='escape_local_optima',
        variation_rate=0.1,
        variation_strength=0.2
    )
    
    swarm.optimize()
    
    print(f"✅ Escape local optima results:")
    print(f"   Final cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Strategy: Strong displacement to escape local optima")
    
    return swarm.best_cost

def test_opposition_based_variation():
    """Test opposition-based variation"""
    print("\n🔄 Testing Opposition-Based Variation")
    print("=" * 50)
    
    # Test opposition-based variation
    swarm = Swarm(
        n_particles=15,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=sphere,
        algo='global',
        variation_strategy='opposition_based',
        variation_rate=0.15,
        variation_strength=0.1
    )
    
    swarm.optimize()
    
    print(f"✅ Opposition-based variation results:")
    print(f"   Final cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Strategy: Explore opposite regions of search space")
    
    return swarm.best_cost

def main():
    """Run all variation operator tests"""
    print("🎯 Variation Operators Test Suite")
    print("=" * 60)
    print("Testing variation operators designed to help particles")
    print("escape local optima and prevent premature convergence.")
    
    # Run all tests
    test_variation_operators()
    test_variation_vs_no_variation()
    test_variation_on_multimodal_functions()
    test_adaptive_variation_strength()
    test_escape_local_optima()
    test_opposition_based_variation()
    
    print("\n" + "=" * 60)
    print("🎉 Variation Operators Testing Complete!")
    print("=" * 60)
    print("\n✨ New Variation Features:")
    print("✅ Gaussian, adaptive, escape local optima variations")
    print("✅ Diversity preserving and opposition-based variations")
    print("✅ Hybrid variation with multiple strategies")
    print("✅ Stagnation detection and adaptive response")
    print("✅ Strong variations for stuck particles")
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., variation_strategy='hybrid', variation_rate=0.1)")

if __name__ == "__main__":
    main()
