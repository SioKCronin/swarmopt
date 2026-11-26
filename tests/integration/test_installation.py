#!/usr/bin/env python3
"""
Test script to verify SwarmOpt installation
Run this from anywhere to test your SwarmOpt installation
"""

def test_installation():
    """Test that SwarmOpt is properly installed and working"""
    print("🧪 Testing SwarmOpt Installation...")
    print("=" * 50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from swarmopt import Swarm
        from swarmopt.functions import sphere, rosenbrock, ackley
        print("   ✅ All imports successful")
        
        # Test basic functionality
        print("\n2. Testing basic PSO...")
        swarm = Swarm(
            n_particles=20,
            dims=2,
            c1=2.0,
            c2=2.0,
            w=0.9,
            epochs=20,
            obj_func=sphere,
            algo='global',
            velocity_clamp=(-5, 5)
        )
        swarm.optimize()
        print(f"   ✅ Sphere optimization: Cost = {swarm.best_cost:.6f}")
        
        # Test different algorithms
        print("\n3. Testing different algorithms...")
        algorithms = ['global', 'local', 'unified']
        for algo in algorithms:
            swarm = Swarm(10, 2, 2.0, 2.0, 0.9, 10, sphere, algo, velocity_clamp=(-2, 2))
            swarm.optimize()
            print(f"   ✅ {algo.upper()} PSO: Cost = {swarm.best_cost:.6f}")
        
        # Test different functions
        print("\n4. Testing different functions...")
        functions = [
            (sphere, "Sphere"),
            (rosenbrock, "Rosenbrock"), 
            (ackley, "Ackley")
        ]
        
        for func, name in functions:
            swarm = Swarm(10, 2, 2.0, 2.0, 0.9, 10, func, 'global', velocity_clamp=(-2, 2))
            swarm.optimize()
            print(f"   ✅ {name} function: Cost = {swarm.best_cost:.6f}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ SwarmOpt is properly installed and working")
        print("✅ You can now use it from anywhere!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("❌ SwarmOpt installation failed")
        return False

if __name__ == "__main__":
    success = test_installation()
    exit(0 if success else 1)
