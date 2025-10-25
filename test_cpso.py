#!/usr/bin/env python3
"""
Test Cooperative PSO (CPSO) Implementation

This script tests the new Cooperative PSO algorithm with multiple
collaborating swarms.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin

def test_cpso_basic():
    """Test basic CPSO functionality"""
    print("🧪 Testing Cooperative PSO (CPSO)")
    print("=" * 50)
    
    # Test parameters
    n_particles = 20
    dims = 6
    c1 = 2.0
    c2 = 2.0
    w = 0.9
    epochs = 50
    
    print(f"Testing CPSO on Sphere function ({dims}D)")
    print(f"Parameters: {n_particles} particles, {epochs} epochs")
    
    # Test CPSO
    swarm = Swarm(
        n_particles=n_particles,
        dims=dims,
        c1=c1, c2=c2, w=w,
        epochs=epochs,
        obj_func=sphere,
        algo='cpso',
        n_swarms=3,  # 3 collaborating swarms
        communication_strategy='best'
    )
    
    swarm.optimize()
    
    print(f"✅ CPSO Results:")
    print(f"   Best cost: {swarm.best_cost:.6f}")
    print(f"   Best position: {swarm.best_pos}")
    print(f"   Runtime: {swarm.runtime:.3f}s")
    
    return swarm

def test_cpso_communication_strategies():
    """Test different communication strategies"""
    print("\n🔄 Testing CPSO Communication Strategies")
    print("=" * 50)
    
    strategies = ['best', 'random', 'tournament']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} communication strategy...")
        
        swarm = Swarm(
            n_particles=15,
            dims=4,
            c1=2.0, c2=2.0, w=0.9,
            epochs=30,
            obj_func=rosenbrock,
            algo='cpso',
            n_swarms=2,
            communication_strategy=strategy
        )
        
        swarm.optimize()
        results[strategy] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime
        }
        
        print(f"   {strategy}: Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['cost'])
    print(f"\n🏆 Best strategy: {best_strategy} (cost: {results[best_strategy]['cost']:.6f})")
    
    return results

def test_cpso_vs_standard():
    """Compare CPSO with standard PSO algorithms"""
    print("\n⚔️ CPSO vs Standard PSO Comparison")
    print("=" * 50)
    
    # Test parameters
    n_particles = 20
    dims = 4
    epochs = 40
    obj_func = ackley
    
    algorithms = [
        ('global', 'Global Best PSO'),
        ('local', 'Local Best PSO'),
        ('unified', 'Unified PSO'),
        ('cpso', 'Cooperative PSO')
    ]
    
    results = {}
    
    for algo, name in algorithms:
        print(f"\nTesting {name}...")
        
        if algo == 'cpso':
            swarm = Swarm(
                n_particles=n_particles,
                dims=dims,
                c1=2.0, c2=2.0, w=0.9,
                epochs=epochs,
                obj_func=obj_func,
                algo=algo,
                n_swarms=2,
                communication_strategy='best'
            )
        else:
            swarm = Swarm(
                n_particles=n_particles,
                dims=dims,
                c1=2.0, c2=2.0, w=0.9,
                epochs=epochs,
                obj_func=obj_func,
                algo=algo
            )
        
        swarm.optimize()
        results[name] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime
        }
        
        print(f"   {name}: Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s")
    
    # Rank algorithms
    print(f"\n📊 Algorithm Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cost'])
    for i, (name, result) in enumerate(sorted_results, 1):
        rank = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {rank} {name:20} | Cost: {result['cost']:8.6f} | Time: {result['runtime']:6.3f}s")
    
    return results

def test_cpso_scalability():
    """Test CPSO performance on different problem sizes"""
    print("\n📈 Testing CPSO Scalability")
    print("=" * 50)
    
    dimensions = [4, 8, 12, 16]
    results = {}
    
    for dims in dimensions:
        print(f"\nTesting {dims}D problem...")
        
        # Calculate appropriate number of swarms
        n_swarms = max(2, dims // 4)
        
        swarm = Swarm(
            n_particles=20,
            dims=dims,
            c1=2.0, c2=2.0, w=0.9,
            epochs=30,
            obj_func=sphere,
            algo='cpso',
            n_swarms=n_swarms,
            communication_strategy='best'
        )
        
        swarm.optimize()
        results[dims] = {
            'cost': swarm.best_cost,
            'runtime': swarm.runtime,
            'n_swarms': n_swarms
        }
        
        print(f"   {dims}D: Cost = {swarm.best_cost:.6f}, Time = {swarm.runtime:.3f}s, Swarms = {n_swarms}")
    
    return results

def test_cpso_swarm_statistics():
    """Test CPSO swarm statistics and analysis"""
    print("\n📊 Testing CPSO Swarm Statistics")
    print("=" * 50)
    
    swarm = Swarm(
        n_particles=15,
        dims=6,
        c1=2.0, c2=2.0, w=0.9,
        epochs=25,
        obj_func=rastrigin,
        algo='cpso',
        n_swarms=3,
        communication_strategy='best'
    )
    
    swarm.optimize()
    
    # Get swarm statistics
    stats = swarm.cpso.get_swarm_statistics()
    
    print(f"✅ CPSO Statistics:")
    print(f"   Number of swarms: {stats['n_swarms']}")
    print(f"   Total particles: {stats['total_particles']}")
    print(f"   Dimension assignments: {stats['dimension_assignments']}")
    
    print(f"\n📈 Swarm Performance:")
    for swarm_perf in stats['swarm_performance']:
        print(f"   Swarm {swarm_perf['swarm_id']}: "
              f"Dims {swarm_perf['dimensions']}, "
              f"Cost {swarm_perf['best_cost']:.6f}, "
              f"Particles {swarm_perf['n_particles']}")
    
    return stats

def main():
    """Run all CPSO tests"""
    print("🎯 Cooperative PSO (CPSO) Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_cpso_basic()
    test_cpso_communication_strategies()
    test_cpso_vs_standard()
    test_cpso_scalability()
    test_cpso_swarm_statistics()
    
    print("\n" + "=" * 60)
    print("🎉 CPSO Testing Complete!")
    print("=" * 60)
    print("\n✨ New CPSO Features:")
    print("✅ Multiple collaborating swarms")
    print("✅ Dimension-based swarm assignment")
    print("✅ Communication strategies (best, random, tournament)")
    print("✅ Scalable to high-dimensional problems")
    print("✅ Detailed swarm statistics and analysis")
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., algo='cpso', n_swarms=3, communication_strategy='best')")

if __name__ == "__main__":
    main()
