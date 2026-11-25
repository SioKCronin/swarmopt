#!/usr/bin/env python3
"""
Test Swarm Diversity Measurement and Management System

This script tests the comprehensive diversity measurement system
that monitors swarm diversity and applies interventions when needed.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin

def test_diversity_measurement():
    """Test basic diversity measurement functions"""
    print("🧪 Testing Swarm Diversity Measurement")
    print("=" * 50)
    
    from swarmopt.utils.diversity import (
        calculate_swarm_diversity, calculate_dimension_diversity,
        calculate_convergence_diversity, calculate_velocity_diversity,
        calculate_fitness_diversity, detect_diversity_crisis
    )
    
    # Create test particles
    particles = [
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
        np.array([3.0, 4.0]),
        np.array([4.0, 5.0]),
        np.array([5.0, 6.0])
    ]
    
    # Test different diversity metrics
    methods = ['euclidean', 'manhattan', 'variance', 'radius', 'entropy']
    
    print("Testing diversity calculation methods:")
    for method in methods:
        diversity = calculate_swarm_diversity(particles, method)
        print(f"   {method:10}: {diversity:.4f}")
    
    # Test dimension diversity
    dim_diversity = calculate_dimension_diversity(particles)
    print(f"\nDimension diversity: {dim_diversity}")
    
    # Test convergence diversity
    conv_metrics = calculate_convergence_diversity(particles)
    print(f"Convergence metrics: {conv_metrics}")
    
    # Test crisis detection
    crisis = detect_diversity_crisis(particles, diversity_threshold=0.1)
    for method in methods:
        value = calculate_swarm_diversity(particles, method)
        assert np.isfinite(value) and value > 0, (method, value)
    print(f"Diversity crisis detected: {crisis}")
    
    return True

def test_diversity_monitoring():
    """Test diversity monitoring with PSO"""
    print("\n📊 Testing Diversity Monitoring with PSO")
    print("=" * 50)
    
    # Test with diversity monitoring enabled
    swarm = Swarm(
        n_particles=20,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=30,
        obj_func=rastrigin,  # Challenging function with many local optima
        algo='global',
        diversity_monitoring=True,
        diversity_threshold=0.1
    )
    
    print("Running PSO with diversity monitoring...")
    swarm.optimize()
    
    print(f"✅ Results:")
    print(f"   Best cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Diversity interventions: {len([d for d in swarm.diversity_history if d['needs_intervention']])}")
    
    # Analyze diversity history
    if swarm.diversity_history:
        final_diversity = swarm.diversity_history[-1]['stats']['diversity']
        print(f"   Final diversity: {final_diversity:.4f}")
        
        # Show intervention types
        interventions = [d['recommended_intervention'] for d in swarm.diversity_history if d['needs_intervention']]
        if interventions:
            print(f"   Interventions applied: {set(interventions)}")
    
    return swarm

def test_diversity_vs_no_diversity():
    """Compare PSO with and without diversity monitoring"""
    print("\n⚔️ Diversity Monitoring vs No Monitoring")
    print("=" * 50)
    
    # Test parameters
    n_particles = 15
    dims = 2
    epochs = 40
    obj_func = ackley  # Function with many local optima
    
    # Without diversity monitoring
    print("Testing PSO without diversity monitoring...")
    swarm_no_div = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global'
    )
    swarm_no_div.optimize()
    
    # With diversity monitoring
    print("Testing PSO with diversity monitoring...")
    swarm_with_div = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global',
        diversity_monitoring=True,
        diversity_threshold=0.15
    )
    swarm_with_div.optimize()
    
    print(f"\n📊 Results:")
    print(f"   No Diversity Monitoring:    Cost = {swarm_no_div.best_cost:.6f}, Time = {swarm_no_div.runtime:.3f}s")
    print(f"   With Diversity Monitoring:  Cost = {swarm_with_div.best_cost:.6f}, Time = {swarm_with_div.runtime:.3f}s")
    
    improvement = ((swarm_no_div.best_cost - swarm_with_div.best_cost) / 
                  swarm_no_div.best_cost) * 100
    print(f"   Improvement: {improvement:.1f}% better with diversity monitoring")
    
    return {
        'no_diversity': swarm_no_div.best_cost,
        'with_diversity': swarm_with_div.best_cost,
        'improvement': improvement
    }

def test_diversity_interventions():
    """Test different diversity intervention strategies"""
    print("\n🔧 Testing Diversity Intervention Strategies")
    print("=" * 50)
    
    # Test on challenging function
    obj_func = rastrigin
    n_particles = 20
    epochs = 35
    
    strategies = [
        ('restart', 'Particle Restart'),
        ('escape_local_optima', 'Escape Local Optima'),
        ('diversity_preserving', 'Diversity Preserving'),
        ('opposition_based', 'Opposition-Based'),
        ('adaptive_strength', 'Adaptive Strength')
    ]
    
    results = {}
    
    for strategy, name in strategies:
        print(f"\nTesting {name} intervention...")
        
        swarm = Swarm(
            n_particles=n_particles,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=epochs,
            obj_func=obj_func,
            algo='global',
            diversity_monitoring=True,
            diversity_threshold=0.1,
            variation_strategy=strategy if strategy != 'restart' else None
        )
        
        swarm.optimize()
        results[name] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime,
            'interventions': len([d for d in swarm.diversity_history if d['needs_intervention']])
        }
        
        print(f"   {name}: Cost = {swarm.best_cost:.6f}, Interventions = {results[name]['interventions']}")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best intervention strategy: {best_strategy}")
    
    return results

def test_diversity_metrics_comparison():
    """Compare different diversity metrics"""
    print("\n📏 Testing Diversity Metrics Comparison")
    print("=" * 50)
    
    from swarmopt.utils.diversity import calculate_swarm_diversity
    
    # Create different swarm configurations
    configs = [
        ('High Diversity', [
            np.array([1.0, 1.0]),
            np.array([5.0, 5.0]),
            np.array([-2.0, 3.0]),
            np.array([4.0, -1.0])
        ]),
        ('Medium Diversity', [
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            np.array([3.0, 3.0]),
            np.array([4.0, 4.0])
        ]),
        ('Low Diversity', [
            np.array([1.0, 1.0]),
            np.array([1.1, 1.1]),
            np.array([0.9, 0.9]),
            np.array([1.05, 1.05])
        ])
    ]
    
    methods = ['euclidean', 'manhattan', 'variance', 'radius', 'entropy']
    measured = {}
    
    print("Diversity metrics comparison:")
    print(f"{'Config':<15} {'Euclidean':<10} {'Manhattan':<10} {'Variance':<10} {'Radius':<10} {'Entropy':<10}")
    print("-" * 70)
    
    for config_name, particles in configs:
        metrics = []
        for method in methods:
            diversity = calculate_swarm_diversity(particles, method)
            metrics.append(f"{diversity:.4f}")
            measured[(config_name, method)] = diversity
        
        print(f"{config_name:<15} {metrics[0]:<10} {metrics[1]:<10} {metrics[2]:<10} {metrics[3]:<10} {metrics[4]:<10}")
    
    # 'entropy' is scale-invariant (its bins span the swarm's own range), so
    # it is left out here; see tests/unit/test_diversity_metrics.py.
    for method in [m for m in methods if m != 'entropy']:
        assert measured[('High Diversity', method)] > measured[('Low Diversity', method)], method
    return True

def test_diversity_trend_analysis():
    """Test diversity trend analysis over time"""
    print("\n📈 Testing Diversity Trend Analysis")
    print("=" * 50)
    
    # Run PSO with diversity monitoring
    swarm = Swarm(
        n_particles=15,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=25,
        obj_func=rosenbrock,
        algo='global',
        diversity_monitoring=True,
        diversity_threshold=0.1
    )
    
    swarm.optimize()
    
    # Analyze diversity trends
    if swarm.diversity_history:
        diversities = [d['stats']['diversity'] for d in swarm.diversity_history]
        interventions = [d['needs_intervention'] for d in swarm.diversity_history]
        
        print(f"✅ Diversity trend analysis:")
        print(f"   Initial diversity: {diversities[0]:.4f}")
        print(f"   Final diversity: {diversities[-1]:.4f}")
        print(f"   Average diversity: {np.mean(diversities):.4f}")
        print(f"   Diversity variance: {np.var(diversities):.4f}")
        print(f"   Interventions applied: {sum(interventions)}")
        
        # Check for diversity trends
        if len(diversities) >= 3:
            recent_trend = diversities[-3:]
            if recent_trend[-1] > recent_trend[0] * 1.1:
                trend = "increasing"
            elif recent_trend[-1] < recent_trend[0] * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
            print(f"   Recent trend: {trend}")
    
    assert len(swarm.diversity_history) == swarm.epochs
    diversities = [d['stats']['diversity'] for d in swarm.diversity_history]
    assert all(np.isfinite(d) and d >= 0 for d in diversities)
    return True

def main():
    """Run all diversity system tests"""
    print("🎯 Swarm Diversity Measurement and Management Test Suite")
    print("=" * 70)
    print("Testing comprehensive diversity measurement system that monitors")
    print("swarm diversity and applies interventions to prevent convergence.")
    
    # Run all tests
    test_diversity_measurement()
    test_diversity_monitoring()
    test_diversity_vs_no_diversity()
    test_diversity_interventions()
    test_diversity_metrics_comparison()
    test_diversity_trend_analysis()
    
    print("\n" + "=" * 70)
    print("🎉 Diversity System Testing Complete!")
    print("=" * 70)
    print("\n✨ New Diversity Features:")
    print("✅ Multiple diversity metrics (Euclidean, Manhattan, Variance, Radius, Entropy)")
    print("✅ Real-time diversity monitoring during optimization")
    print("✅ Automatic intervention when diversity crisis detected")
    print("✅ Particle restart, escape variations, diversity preservation")
    print("✅ Opposition-based and adaptive strength interventions")
    print("✅ Comprehensive diversity statistics and trend analysis")
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., diversity_monitoring=True, diversity_threshold=0.1)")

if __name__ == "__main__":
    main()
