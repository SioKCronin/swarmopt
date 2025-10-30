#!/usr/bin/env python3
"""
Test Proactive Particle Swarm Optimization (PPSO)

This script tests the PPSO implementation with mixed swarms containing
both proactive and reactive particles. Proactive particles use knowledge
gain metrics to explore regions with low sample density.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin, griewank, weierstrass

def test_ppso_basic():
    """Test basic PPSO functionality"""
    print("🧪 Testing Basic PPSO Functionality")
    print("=" * 50)
    
    # Test PPSO on simple function
    swarm = Swarm(
        n_particles=20,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=sphere,
        algo='global',
        ppso_enabled=True,
        proactive_ratio=0.25,
        knowledge_method='gaussian',
        exploration_weight=0.5
    )
    
    print("Running PPSO optimization...")
    swarm.optimize()
    
    print(f"✅ Results:")
    print(f"   Best cost: {swarm.best_cost:.6f}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    print(f"   Proactive particles: {swarm.ppso.n_proactive}")
    print(f"   Reactive particles: {swarm.ppso.n_reactive}")
    
    # Get PPSO statistics
    stats = swarm.ppso.get_statistics()
    print(f"   Final knowledge gain: {stats['final_knowledge_gain']:.4f}")
    print(f"   Average knowledge gain: {stats['avg_knowledge_gain']:.4f}")
    print(f"   Proactive contribution: {stats['proactive_contribution']:.2f}")
    
    return swarm

def test_knowledge_gain_methods():
    """Test different knowledge gain calculation methods"""
    print("\n📊 Testing Knowledge Gain Methods")
    print("=" * 50)
    
    methods = ['gaussian', 'inverse_distance', 'entropy', 'acquisition']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} knowledge gain method...")
        
        swarm = Swarm(
            n_particles=15,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=30,
            obj_func=ackley,  # Challenging function
            algo='global',
            ppso_enabled=True,
            proactive_ratio=0.3,
            knowledge_method=method,
            exploration_weight=0.6
        )
        
        swarm.optimize()
        results[method] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime,
            'stats': swarm.ppso.get_statistics()
        }
        
        print(f"   {method}: Cost = {swarm.best_cost:.6f}, "
              f"Knowledge Gain = {results[method]['stats']['final_knowledge_gain']:.4f}")
    
    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best knowledge gain method: {best_method}")
    
    return results

def test_proactive_ratios():
    """Test different proactive particle ratios"""
    print("\n⚖️ Testing Proactive Particle Ratios")
    print("=" * 50)
    
    ratios = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    results = {}
    
    for ratio in ratios:
        print(f"\nTesting {ratio*100:.0f}% proactive particles...")
        
        swarm = Swarm(
            n_particles=20,
            dims=3,
            c1=2.0, c2=2.0, w=0.9,
            epochs=40,
            obj_func=rastrigin,  # Multimodal function
            algo='global',
            ppso_enabled=True,
            proactive_ratio=ratio,
            knowledge_method='gaussian',
            exploration_weight=0.5
        )
        
        swarm.optimize()
        results[ratio] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime,
            'proactive_contribution': swarm.ppso.get_statistics()['proactive_contribution']
        }
        
        print(f"   {ratio*100:.0f}% proactive: Cost = {swarm.best_cost:.6f}, "
              f"Contribution = {results[ratio]['proactive_contribution']:.2f}")
    
    # Find best ratio
    best_ratio = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best proactive ratio: {best_ratio*100:.0f}%")
    
    return results

def test_ppso_vs_standard_pso():
    """Compare PPSO with standard PSO"""
    print("\n⚔️ PPSO vs Standard PSO Comparison")
    print("=" * 50)
    
    # Test parameters
    n_particles = 20
    dims = 2
    epochs = 50
    obj_func = ackley  # Challenging function with many local optima
    
    # Standard PSO
    print("Testing Standard PSO...")
    swarm_standard = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global'
    )
    swarm_standard.optimize()
    
    # PPSO
    print("Testing PPSO...")
    swarm_ppso = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo='global',
        ppso_enabled=True,
        proactive_ratio=0.25,
        knowledge_method='gaussian',
        exploration_weight=0.5
    )
    swarm_ppso.optimize()
    
    print(f"\n📊 Results:")
    print(f"   Standard PSO:    Cost = {swarm_standard.best_cost:.6f}, Time = {swarm_standard.runtime:.3f}s")
    print(f"   PPSO:            Cost = {swarm_ppso.best_cost:.6f}, Time = {swarm_ppso.runtime:.3f}s")
    
    improvement = ((swarm_standard.best_cost - swarm_ppso.best_cost) / 
                  swarm_standard.best_cost) * 100
    print(f"   Improvement: {improvement:.1f}% better with PPSO")
    
    # PPSO statistics
    ppso_stats = swarm_ppso.ppso.get_statistics()
    print(f"   PPSO Knowledge Gain: {ppso_stats['final_knowledge_gain']:.4f}")
    print(f"   Proactive Contribution: {ppso_stats['proactive_contribution']:.2f}")
    
    return {
        'standard_cost': swarm_standard.best_cost,
        'ppso_cost': swarm_ppso.best_cost,
        'improvement': improvement
    }

def test_exploration_weights():
    """Test different exploration weights"""
    print("\n🎯 Testing Exploration Weights")
    print("=" * 50)
    
    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for weight in weights:
        print(f"\nTesting exploration weight {weight}...")
        
        swarm = Swarm(
            n_particles=15,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=35,
            obj_func=rosenbrock,  # Valley function
            algo='global',
            ppso_enabled=True,
            proactive_ratio=0.3,
            knowledge_method='gaussian',
            exploration_weight=weight
        )
        
        swarm.optimize()
        results[weight] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime,
            'exploration': swarm.ppso.get_statistics()['avg_exploration']
        }
        
        print(f"   Weight {weight}: Cost = {swarm.best_cost:.6f}, "
              f"Exploration = {results[weight]['exploration']:.4f}")
    
    # Find best weight
    best_weight = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best exploration weight: {best_weight}")
    
    return results

def test_multimodal_functions():
    """Test PPSO on multimodal functions"""
    print("\n🌊 Testing PPSO on Multimodal Functions")
    print("=" * 50)
    
    functions = [
        ('Sphere', sphere),
        ('Rosenbrock', rosenbrock),
        ('Ackley', ackley),
        ('Rastrigin', rastrigin),
        ('Griewank', griewank),
        ('Weierstrass', weierstrass)
    ]
    
    results = {}
    
    for name, func in functions:
        print(f"\nTesting {name} function...")
        
        swarm = Swarm(
            n_particles=20,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=40,
            obj_func=func,
            algo='global',
            ppso_enabled=True,
            proactive_ratio=0.25,
            knowledge_method='gaussian',
            exploration_weight=0.5
        )
        
        swarm.optimize()
        results[name] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime,
            'stats': swarm.ppso.get_statistics()
        }
        
        print(f"   {name}: Cost = {swarm.best_cost:.6f}, "
              f"Knowledge Gain = {results[name]['stats']['final_knowledge_gain']:.4f}")
    
    return results

def test_knowledge_gain_visualization():
    """Test knowledge gain visualization (if matplotlib available)"""
    print("\n📈 Testing Knowledge Gain Visualization")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        # Run PPSO and collect knowledge gain history
        swarm = Swarm(
            n_particles=15,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=30,
            obj_func=ackley,
            algo='global',
            ppso_enabled=True,
            proactive_ratio=0.3,
            knowledge_method='gaussian',
            exploration_weight=0.5
        )
        
        swarm.optimize()
        
        # Plot knowledge gain over time
        knowledge_history = swarm.ppso.knowledge_gain_history
        exploration_history = swarm.ppso.exploration_history
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(knowledge_history)
        plt.title('Knowledge Gain Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Average Knowledge Gain')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(exploration_history)
        plt.title('Exploration Activity Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Average Velocity Magnitude')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ppso_knowledge_gain.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Knowledge gain visualization saved as 'ppso_knowledge_gain.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

def main():
    """Run all PPSO tests"""
    print("🎯 Proactive Particle Swarm Optimization (PPSO) Test Suite")
    print("=" * 70)
    print("Testing mixed swarms with proactive particles that use knowledge")
    print("gain metrics to explore regions with low sample density.")
    
    # Run all tests
    test_ppso_basic()
    test_knowledge_gain_methods()
    test_proactive_ratios()
    test_ppso_vs_standard_pso()
    test_exploration_weights()
    test_multimodal_functions()
    test_knowledge_gain_visualization()
    
    print("\n" + "=" * 70)
    print("🎉 PPSO Testing Complete!")
    print("=" * 70)
    print("\n✨ PPSO Features:")
    print("✅ Mixed swarms with proactive and reactive particles")
    print("✅ Knowledge gain metrics for exploration")
    print("✅ Gaussian Process-inspired acquisition functions")
    print("✅ Adaptive exploration weights")
    print("✅ Multiple knowledge gain calculation methods")
    print("✅ Comprehensive statistics and analysis")
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., ppso_enabled=True, proactive_ratio=0.25)")

if __name__ == "__main__":
    main()
