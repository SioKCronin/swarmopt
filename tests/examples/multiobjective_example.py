#!/usr/bin/env python3
"""
Multiobjective Optimization Example

This script demonstrates how to use SwarmOpt for multiobjective optimization
with the ZDT1 benchmark function.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.utils.simple_multiobjective import zdt1, zdt2, zdt3

def main():
    """Run multiobjective optimization example"""
    print("🎯 Multiobjective Optimization Example")
    print("=" * 50)
    
    # Example 1: ZDT1 function
    print("\n📊 Example 1: ZDT1 Function")
    print("-" * 30)
    
    swarm = Swarm(
        n_particles=20,
        dims=5,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=zdt1,
        multiobjective=True,
        archive_size=50
    )
    
    print("Running optimization...")
    swarm.optimize()
    
    # Get results
    pareto_front = swarm.mo_optimizer.archive
    print(f"✅ Found {len(pareto_front)} Pareto-optimal solutions")
    print(f"✅ Runtime: {swarm.runtime:.3f}s")
    
    # Show some solutions
    if pareto_front:
        objectives = np.array([sol['objectives'] for sol in pareto_front])
        print(f"✅ Objective ranges:")
        print(f"   f1: [{np.min(objectives[:, 0]):.4f}, {np.max(objectives[:, 0]):.4f}]")
        print(f"   f2: [{np.min(objectives[:, 1]):.4f}, {np.max(objectives[:, 1]):.4f}]")
        
        # Show first few solutions
        print(f"\n📋 First 5 Pareto-optimal solutions:")
        for i in range(min(5, len(pareto_front))):
            sol = pareto_front[i]
            print(f"   Solution {i+1}: f1={sol['objectives'][0]:.4f}, f2={sol['objectives'][1]:.4f}")
    
    # Example 2: ZDT2 function
    print("\n📊 Example 2: ZDT2 Function")
    print("-" * 30)
    
    swarm2 = Swarm(
        n_particles=15,
        dims=4,
        c1=2.0, c2=2.0, w=0.9,
        epochs=30,
        obj_func=zdt2,
        multiobjective=True,
        archive_size=30
    )
    
    print("Running optimization...")
    swarm2.optimize()
    
    pareto_front2 = swarm2.mo_optimizer.archive
    print(f"✅ Found {len(pareto_front2)} Pareto-optimal solutions")
    print(f"✅ Runtime: {swarm2.runtime:.3f}s")
    
    # Example 3: Custom multiobjective function
    print("\n📊 Example 3: Custom Multiobjective Function")
    print("-" * 30)
    
    def custom_multiobjective(x):
        """Custom function with two conflicting objectives"""
        # Objective 1: Minimize sum of squares
        f1 = np.sum(x**2)
        
        # Objective 2: Minimize sum of absolute values
        f2 = np.sum(np.abs(x))
        
        return np.array([f1, f2])
    
    swarm3 = Swarm(
        n_particles=20,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=custom_multiobjective,
        multiobjective=True,
        archive_size=40
    )
    
    print("Running optimization...")
    swarm3.optimize()
    
    pareto_front3 = swarm3.mo_optimizer.archive
    print(f"✅ Found {len(pareto_front3)} Pareto-optimal solutions")
    print(f"✅ Runtime: {swarm3.runtime:.3f}s")
    
    if pareto_front3:
        objectives3 = np.array([sol['objectives'] for sol in pareto_front3])
        print(f"✅ Objective ranges:")
        print(f"   f1 (sum of squares): [{np.min(objectives3[:, 0]):.4f}, {np.max(objectives3[:, 0]):.4f}]")
        print(f"   f2 (sum of abs): [{np.min(objectives3[:, 1]):.4f}, {np.max(objectives3[:, 1]):.4f}]")
    
    print("\n" + "=" * 50)
    print("🎉 Multiobjective Optimization Examples Complete!")
    print("=" * 50)
    print("\n💡 Key Points:")
    print("• Multiobjective optimization finds Pareto-optimal solutions")
    print("• Solutions represent trade-offs between conflicting objectives")
    print("• Use multiobjective=True in Swarm constructor")
    print("• Access results via swarm.mo_optimizer.archive")
    print("• Archive size controls number of solutions maintained")

if __name__ == "__main__":
    main()
