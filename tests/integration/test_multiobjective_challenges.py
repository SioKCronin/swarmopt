#!/usr/bin/env python3
"""
Challenging Multiobjective Test Suite

Tests MOPSO on the most difficult benchmark problems from literature:
- ZDT4: 21^9 local Pareto fronts (highly multimodal)
- ZDT6: Non-uniform, biased search space
- DTLZ3: 3^k local fronts (extremely multimodal)
- DTLZ4: Biased density
- DTLZ7: 2^(M-1) disconnected regions
- Kursawe: Non-convex, disconnected
- Viennet: 3 objectives, complex geometry
- Many-objective: 5+ objectives

References:
- Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of Multiobjective EAs
- Deb, K., et al. (2005). Scalable Test Problems for Evolutionary Multiobjective Optimization
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.utils.mo_benchmarks import (
    zdt4, zdt6, dtlz3, dtlz4, dtlz7, dtlz5_degenerate,
    fonseca_fleming, kursawe, viennet, many_objective_dtlz2,
    inverted_generational_distance, spacing_metric, hypervolume_wfg, spread_metric,
    generate_true_pareto_front, get_benchmark_info
)

def test_zdt4_multimodal():
    """Test ZDT4 - 21^9 local Pareto fronts"""
    print("=" * 70)
    print("🔥 CHALLENGE 1: ZDT4 - Highly Multimodal (21^9 local fronts)")
    print("=" * 70)
    
    info = get_benchmark_info('zdt4')
    print(f"Difficulty: {'⭐' * info['difficulty']} ({info['difficulty']}/5)")
    print(f"Challenges: {', '.join(info['challenges'])}")
    print()
    
    swarm = Swarm(
        n_particles=50,  # More particles for multimodal
        dims=10,
        c1=2.0, c2=2.0, w=0.9,
        epochs=100,  # More epochs for difficult problem
        obj_func=zdt4,
        multiobjective=True,
        archive_size=100
    )
    
    print("Running optimization...")
    swarm.optimize()
    
    # Get results
    pareto_front = swarm.mo_optimizer.archive
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    
    # Calculate metrics
    true_front = generate_true_pareto_front('zdt1', 100)  # Similar shape to ZDT1
    igd = inverted_generational_distance(objectives, true_front)
    spacing = spacing_metric(objectives)
    
    print(f"\n📊 Results:")
    print(f"   Pareto front size: {len(pareto_front)}")
    print(f"   IGD (convergence): {igd:.6f}")
    print(f"   Spacing (diversity): {spacing:.6f}")
    print(f"   Runtime: {swarm.runtime:.2f}s")
    print(f"   Objective ranges: f1=[{np.min(objectives[:, 0]):.4f}, {np.max(objectives[:, 0]):.4f}], "
          f"f2=[{np.min(objectives[:, 1]):.4f}, {np.max(objectives[:, 1]):.4f}]")
    
    return swarm, igd, spacing

def test_zdt6_biased():
    """Test ZDT6 - Non-uniform, biased search space"""
    print("\n" + "=" * 70)
    print("🔥 CHALLENGE 2: ZDT6 - Non-uniform Search Space")
    print("=" * 70)
    
    info = get_benchmark_info('zdt6')
    print(f"Difficulty: {'⭐' * info['difficulty']} ({info['difficulty']}/5)")
    print(f"Challenges: {', '.join(info['challenges'])}")
    print()
    
    swarm = Swarm(
        n_particles=40,
        dims=10,
        c1=2.0, c2=2.0, w=0.9,
        epochs=80,
        obj_func=zdt6,
        multiobjective=True,
        archive_size=80
    )
    
    print("Running optimization...")
    swarm.optimize()
    
    pareto_front = swarm.mo_optimizer.archive
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    spacing = spacing_metric(objectives)
    
    print(f"\n📊 Results:")
    print(f"   Pareto front size: {len(pareto_front)}")
    print(f"   Spacing (diversity): {spacing:.6f}")
    print(f"   Runtime: {swarm.runtime:.2f}s")
    
    return swarm, spacing

def test_dtlz3_extreme_multimodal():
    """Test DTLZ3 - Extremely multimodal (3^k local fronts)"""
    print("\n" + "=" * 70)
    print("🔥 CHALLENGE 3: DTLZ3 - Extreme Multimodality (3^k fronts)")
    print("=" * 70)
    
    info = get_benchmark_info('dtlz3')
    print(f"Difficulty: {'⭐' * info['difficulty']} ({info['difficulty']}/5)")
    print(f"Challenges: {', '.join(info['challenges'])}")
    print()
    
    n_obj = 3
    k = 5
    dims = n_obj + k - 1
    
    def dtlz3_wrapper(x):
        return dtlz3(x, n_obj)
    
    swarm = Swarm(
        n_particles=60,  # Even more particles
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=120,  # More epochs
        obj_func=dtlz3_wrapper,
        multiobjective=True,
        archive_size=100
    )
    
    print(f"Dimensions: {dims}, Objectives: {n_obj}, Number of local fronts: 3^{k} = {3**k}")
    print("Running optimization...")
    swarm.optimize()
    
    pareto_front = swarm.mo_optimizer.archive
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    spacing = spacing_metric(objectives)
    
    print(f"\n📊 Results:")
    print(f"   Pareto front size: {len(pareto_front)}")
    print(f"   Spacing (diversity): {spacing:.6f}")
    print(f"   Runtime: {swarm.runtime:.2f}s")
    print(f"   Objective ranges:")
    for i in range(n_obj):
        print(f"      f{i+1}: [{np.min(objectives[:, i]):.4f}, {np.max(objectives[:, i]):.4f}]")
    
    return swarm, spacing

def test_dtlz7_disconnected():
    """Test DTLZ7 - Disconnected Pareto regions"""
    print("\n" + "=" * 70)
    print("🔥 CHALLENGE 4: DTLZ7 - Disconnected Pareto Front (2^(M-1) regions)")
    print("=" * 70)
    
    info = get_benchmark_info('dtlz7')
    print(f"Difficulty: {'⭐' * info['difficulty']} ({info['difficulty']}/5)")
    print(f"Challenges: {', '.join(info['challenges'])}")
    print()
    
    n_obj = 3
    dims = n_obj + 19  # k = 20
    
    def dtlz7_wrapper(x):
        return dtlz7(x, n_obj)
    
    swarm = Swarm(
        n_particles=50,
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=100,
        obj_func=dtlz7_wrapper,
        multiobjective=True,
        archive_size=100
    )
    
    print(f"Dimensions: {dims}, Objectives: {n_obj}, Disconnected regions: 2^{n_obj-1} = {2**(n_obj-1)}")
    print("Running optimization...")
    swarm.optimize()
    
    pareto_front = swarm.mo_optimizer.archive
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    spacing = spacing_metric(objectives)
    
    print(f"\n📊 Results:")
    print(f"   Pareto front size: {len(pareto_front)}")
    print(f"   Spacing (diversity): {spacing:.6f}")
    print(f"   Runtime: {swarm.runtime:.2f}s")
    
    return swarm, spacing

def test_kursawe_nonconvex():
    """Test Kursawe - Non-convex, disconnected"""
    print("\n" + "=" * 70)
    print("🔥 CHALLENGE 5: Kursawe - Non-convex & Disconnected")
    print("=" * 70)
    
    info = get_benchmark_info('kursawe')
    print(f"Difficulty: {'⭐' * info['difficulty']} ({info['difficulty']}/5)")
    print(f"Challenges: {', '.join(info['challenges'])}")
    print()
    
    swarm = Swarm(
        n_particles=40,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=80,
        obj_func=kursawe,
        multiobjective=True,
        archive_size=60
    )
    
    print("Running optimization...")
    swarm.optimize()
    
    pareto_front = swarm.mo_optimizer.archive
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    spacing = spacing_metric(objectives)
    
    print(f"\n📊 Results:")
    print(f"   Pareto front size: {len(pareto_front)}")
    print(f"   Spacing (diversity): {spacing:.6f}")
    print(f"   Runtime: {swarm.runtime:.2f}s")
    
    return swarm, spacing

def test_many_objective():
    """Test 5-objective optimization"""
    print("\n" + "=" * 70)
    print("🔥 CHALLENGE 6: Many-Objective (5 objectives)")
    print("=" * 70)
    
    info = get_benchmark_info('many_objective')
    print(f"Difficulty: {'⭐' * info['difficulty']} ({info['difficulty']}/5)")
    print(f"Challenges: {', '.join(info['challenges'])}")
    print()
    
    n_obj = 5
    dims = n_obj + 9
    
    def many_obj_wrapper(x):
        return many_objective_dtlz2(x, n_obj)
    
    swarm = Swarm(
        n_particles=80,  # More particles for many objectives
        dims=dims,
        c1=2.0, c2=2.0, w=0.9,
        epochs=100,
        obj_func=many_obj_wrapper,
        multiobjective=True,
        archive_size=150
    )
    
    print(f"Dimensions: {dims}, Objectives: {n_obj}")
    print("Running optimization...")
    swarm.optimize()
    
    pareto_front = swarm.mo_optimizer.archive
    objectives = np.array([sol['objectives'] for sol in pareto_front])
    spacing = spacing_metric(objectives)
    
    print(f"\n📊 Results:")
    print(f"   Pareto front size: {len(pareto_front)}")
    print(f"   Spacing (diversity): {spacing:.6f}")
    print(f"   Runtime: {swarm.runtime:.2f}s")
    print(f"   Objective ranges:")
    for i in range(n_obj):
        print(f"      f{i+1}: [{np.min(objectives[:, i]):.4f}, {np.max(objectives[:, i]):.4f}]")
    
    return swarm, spacing

def performance_summary():
    """Run all challenges and create performance summary"""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE MOPSO STRESS TEST")
    print("=" * 70)
    print("Testing on most challenging multiobjective benchmarks from literature\n")
    
    results = {}
    
    # Run all challenges
    print("Running all challenges...\n")
    
    swarm1, igd1, spacing1 = test_zdt4_multimodal()
    results['ZDT4'] = {'igd': igd1, 'spacing': spacing1, 'runtime': swarm1.runtime}
    
    swarm2, spacing2 = test_zdt6_biased()
    results['ZDT6'] = {'spacing': spacing2, 'runtime': swarm2.runtime}
    
    swarm3, spacing3 = test_dtlz3_extreme_multimodal()
    results['DTLZ3'] = {'spacing': spacing3, 'runtime': swarm3.runtime}
    
    swarm4, spacing4 = test_dtlz7_disconnected()
    results['DTLZ7'] = {'spacing': spacing4, 'runtime': swarm4.runtime}
    
    swarm5, spacing5 = test_kursawe_nonconvex()
    results['Kursawe'] = {'spacing': spacing5, 'runtime': swarm5.runtime}
    
    swarm6, spacing6 = test_many_objective()
    results['Many-Objective'] = {'spacing': spacing6, 'runtime': swarm6.runtime}
    
    # Summary table
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\n{'Benchmark':<20} {'Difficulty':<12} {'Spacing':<15} {'Runtime':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        info = get_benchmark_info(name.lower())
        difficulty = '⭐' * info['difficulty']
        spacing = result.get('spacing', 0)
        runtime = result.get('runtime', 0)
        print(f"{name:<20} {difficulty:<12} {spacing:<15.6f} {runtime:<12.2f}s")
    
    print("\n💡 Key Insights:")
    print("• Lower spacing = better diversity (more uniform distribution)")
    print("• Runtime increases with problem difficulty and multimodality")
    print("• MOPSO handles up to 5 objectives reasonably well")
    print("• Disconnected fronts (DTLZ7) are particularly challenging")
    
    print("\n📚 Literature Context:")
    print("• ZDT4: Hardest ZDT function (Zitzler et al., 2000)")
    print("• DTLZ3: Hardest DTLZ function (Deb et al., 2005)")
    print("• DTLZ7: Tests ability to find disconnected regions")
    print("• Many-objective: Tests curse of dimensionality")
    
    return results

def main():
    """Run comprehensive MOPSO stress test"""
    print("🎯 Multiobjective PSO - Challenging Benchmark Suite")
    print("=" * 70)
    print("Based on standard test problems from multiobjective optimization literature")
    
    results = performance_summary()
    
    print("\n" + "=" * 70)
    print("✅ All Challenges Complete!")
    print("=" * 70)
    
    print("\n🏆 Algorithm Performance:")
    avg_spacing = np.mean([r['spacing'] for r in results.values()])
    avg_runtime = np.mean([r['runtime'] for r in results.values()])
    
    print(f"   Average spacing: {avg_spacing:.6f}")
    print(f"   Average runtime: {avg_runtime:.2f}s")
    
    print("\n📖 References:")
    print("   [1] Zitzler, E., Deb, K., & Thiele, L. (2000)")
    print("       'Comparison of Multiobjective Evolutionary Algorithms'")
    print("   [2] Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005)")
    print("       'Scalable Test Problems for Evolutionary Multiobjective Optimization'")
    print("   [3] Coello Coello, C. A., et al. (2004)")
    print("       'Handling multiple objectives with particle swarm optimization'")

if __name__ == "__main__":
    main()

