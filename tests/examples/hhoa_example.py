#!/usr/bin/env python3
"""
Horse Herd Optimization Algorithm (HHOA) Example

This example demonstrates the Horse Herd Optimization Algorithm based on:
"A high-speed MPPT based horse herd optimization algorithm with dynamic linear 
active disturbance rejection control for PV battery charging system"
https://www.nature.com/articles/s41598-025-85481-6

The HHOA mimics the motion cycles of a horse herd with three behaviors:
1. Grazing (exploration)
2. Leadership (exploitation)
3. Following (social learning)
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley


def main():
    """Demonstrate HHOA on various benchmark functions"""
    
    print("=" * 70)
    print("🐴 Horse Herd Optimization Algorithm (HHOA) Demo")
    print("=" * 70)
    print("\nBased on: https://www.nature.com/articles/s41598-025-85481-6")
    print("\nHHOA models three horse behaviors:")
    print("  • Grazing: Exploration of search space")
    print("  • Leadership: Exploitation around best solutions")
    print("  • Following: Social learning from leaders")
    print()
    
    # Example 1: Sphere function
    print("-" * 70)
    print("Example 1: Sphere Function (2D)")
    print("-" * 70)
    
    swarm1 = Swarm(
        n_particles=30,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,  # These are ignored for HHOA but kept for compatibility
        epochs=100,
        obj_func=sphere,
        algo='hhoa',  # Enable HHOA
        velocity_clamp=(-5, 5)
    )
    
    print("Running HHOA optimization...")
    swarm1.optimize()
    
    print(f"✅ Best cost: {swarm1.best_cost:.6f}")
    print(f"✅ Best position: {swarm1.best_pos}")
    print(f"✅ Runtime: {swarm1.runtime:.4f}s")
    
    # Example 2: Rosenbrock function
    print("\n" + "-" * 70)
    print("Example 2: Rosenbrock Function (3D)")
    print("-" * 70)
    
    swarm2 = Swarm(
        n_particles=40,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=150,
        obj_func=rosenbrock,
        algo='hhoa',
        velocity_clamp=(-5, 5)
    )
    
    print("Running HHOA optimization...")
    swarm2.optimize()
    
    print(f"✅ Best cost: {swarm2.best_cost:.6f}")
    print(f"✅ Best position: {swarm2.best_pos}")
    print(f"✅ Runtime: {swarm2.runtime:.4f}s")
    
    # Example 3: Ackley function
    print("\n" + "-" * 70)
    print("Example 3: Ackley Function (5D)")
    print("-" * 70)
    
    swarm3 = Swarm(
        n_particles=50,
        dims=5,
        c1=2.0, c2=2.0, w=0.9,
        epochs=200,
        obj_func=ackley,
        algo='hhoa',
        velocity_clamp=(-32, 32)
    )
    
    print("Running HHOA optimization...")
    swarm3.optimize()
    
    print(f"✅ Best cost: {swarm3.best_cost:.6f}")
    print(f"✅ Best position: {swarm3.best_pos}")
    print(f"✅ Runtime: {swarm3.runtime:.4f}s")
    
    # Comparison with standard PSO
    print("\n" + "=" * 70)
    print("Comparison: HHOA vs Standard PSO (Sphere Function)")
    print("=" * 70)
    
    # HHOA
    swarm_hhoa = Swarm(
        n_particles=30,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=100,
        obj_func=sphere,
        algo='hhoa',
        velocity_clamp=(-5, 5)
    )
    swarm_hhoa.optimize()
    
    # Standard PSO
    swarm_pso = Swarm(
        n_particles=30,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=100,
        obj_func=sphere,
        algo='global',
        velocity_clamp=(-5, 5)
    )
    swarm_pso.optimize()
    
    print(f"\nHHOA Results:")
    print(f"  Best cost: {swarm_hhoa.best_cost:.6f}")
    print(f"  Runtime: {swarm_hhoa.runtime:.4f}s")
    
    print(f"\nStandard PSO Results:")
    print(f"  Best cost: {swarm_pso.best_cost:.6f}")
    print(f"  Runtime: {swarm_pso.runtime:.4f}s")
    
    improvement = ((swarm_pso.best_cost - swarm_hhoa.best_cost) / swarm_pso.best_cost) * 100
    print(f"\n📊 HHOA improvement: {improvement:.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ HHOA Demo Complete!")
    print("=" * 70)
    print("\n💡 Key Features of HHOA:")
    print("  • Bio-inspired algorithm mimicking horse herd behavior")
    print("  • Three-phase behavior: grazing, leadership, following")
    print("  • Adaptive exploration-exploitation balance")
    print("  • Effective for MPPT and other optimization problems")
    print("\n📚 Reference:")
    print("  https://www.nature.com/articles/s41598-025-85481-6")


if __name__ == "__main__":
    main()

