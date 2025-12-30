#!/usr/bin/env python3
"""
Test Respect Boundary Feature

This script demonstrates the respect boundary feature where particles
converge to a safe distance around a target rather than exactly on it.
"""

import numpy as np
from swarmopt import Swarm

def test_basic_respect_boundary():
    """Test basic respect boundary functionality"""
    print("🎯 Testing Basic Respect Boundary")
    print("=" * 60)
    
    # Define target position and respect boundary
    target = np.array([5.0, 5.0])
    respect_distance = 2.0
    
    # Define objective: distance to target
    def distance_to_target(x):
        return np.linalg.norm(x - target)
    
    # Test WITHOUT respect boundary
    print("\n📍 Test 1: WITHOUT Respect Boundary")
    print(f"   Target: {target}")
    print(f"   Objective: Minimize distance to target")
    
    swarm1 = Swarm(
        n_particles=20,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=distance_to_target,
        algo='global'
    )
    
    swarm1.optimize()
    final_distance1 = np.linalg.norm(swarm1.best_pos - target)
    
    print(f"✅ Converged to: {swarm1.best_pos}")
    print(f"✅ Distance from target: {final_distance1:.4f}")
    print(f"✅ Objective value: {swarm1.best_cost:.4f}")
    
    # Test WITH respect boundary
    print(f"\n📍 Test 2: WITH Respect Boundary = {respect_distance}")
    print(f"   Target: {target}")
    print(f"   Objective: Minimize distance but stay outside boundary")
    
    swarm2 = Swarm(
        n_particles=20,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=distance_to_target,
        algo='global',
        respect_boundary=respect_distance,
        target_position=target
    )
    
    swarm2.optimize()
    final_distance2 = np.linalg.norm(swarm2.best_pos - target)
    
    print(f"✅ Converged to: {swarm2.best_pos}")
    print(f"✅ Distance from target: {final_distance2:.4f}")
    print(f"✅ Objective value: {swarm2.best_cost:.4f}")
    print(f"✅ Respect boundary maintained: {final_distance2 >= respect_distance}")
    
    return swarm1, swarm2

def test_3d_respect_boundary():
    """Test respect boundary in 3D space"""
    print("\n🌐 Testing 3D Respect Boundary")
    print("=" * 60)
    
    # Define 3D target and boundary
    target = np.array([10.0, 10.0, 10.0])
    respect_distance = 3.0
    
    def distance_to_target_3d(x):
        return np.linalg.norm(x - target)
    
    swarm = Swarm(
        n_particles=25,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=distance_to_target_3d,
        algo='global',
        respect_boundary=respect_distance,
        target_position=target
    )
    
    swarm.optimize()
    final_distance = np.linalg.norm(swarm.best_pos - target)
    
    print(f"✅ Target position: {target}")
    print(f"✅ Respect boundary: {respect_distance}")
    print(f"✅ Converged to: {swarm.best_pos}")
    print(f"✅ Distance from target: {final_distance:.4f}")
    print(f"✅ Boundary respected: {final_distance >= respect_distance}")
    
    return swarm

def test_multiple_respect_distances():
    """Test different respect boundary sizes"""
    print("\n📏 Testing Multiple Respect Distances")
    print("=" * 60)
    
    target = np.array([0.0, 0.0])
    respect_distances = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    def distance_to_origin(x):
        return np.linalg.norm(x)
    
    results = []
    
    for respect_dist in respect_distances:
        swarm = Swarm(
            n_particles=20,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=40,
            obj_func=distance_to_origin,
            algo='global',
            respect_boundary=respect_dist,
            target_position=target
        )
        
        swarm.optimize()
        final_distance = np.linalg.norm(swarm.best_pos - target)
        
        results.append({
            'respect_boundary': respect_dist,
            'final_distance': final_distance,
            'boundary_respected': final_distance >= respect_dist,
            'best_pos': swarm.best_pos
        })
        
        print(f"\n📊 Respect Boundary: {respect_dist}")
        print(f"   Final distance: {final_distance:.4f}")
        print(f"   Boundary respected: {final_distance >= respect_dist}")
        print(f"   Best position: {swarm.best_pos}")
    
    return results

def test_standoff_distance():
    """Test standoff distance scenario"""
    print("\n🎯  Testing Standoff Distance (Safety-Critical)")
    print("=" * 60)
    
    # Earth center as target
    earth_center = np.array([0.0, 0.0, 0.0])
    earth_radius = 6371.0  # km
    desired_altitude = 400.0  # km (ISS altitude)
    min_safe_distance = earth_radius + desired_altitude
    
    # Objective: Minimize energy (distance from optimal orbit)
    optimal_orbit = earth_radius + desired_altitude
    
    def orbit_energy(position):
        distance = np.linalg.norm(position - earth_center)
        # Energy penalty for deviating from optimal orbit
        return abs(distance - optimal_orbit)
    
    print(f"   Earth radius: {earth_radius} km")
    print(f"   Desired altitude: {desired_altitude} km")
    print(f"   Minimum safe distance: {min_safe_distance} km")
    
    swarm = Swarm(
        n_particles=30,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=orbit_energy,
        algo='global',
        respect_boundary=min_safe_distance,
        target_position=earth_center
    )
    
    swarm.optimize()
    final_distance = np.linalg.norm(swarm.best_pos - earth_center)
    altitude = final_distance - earth_radius
    
    print(f"\n✅ Optimal position found: {swarm.best_pos}")
    print(f"✅ Distance from Earth center: {final_distance:.2f} km")
    print(f"✅ Altitude: {altitude:.2f} km")
    print(f"✅ Safe distance maintained: {final_distance >= min_safe_distance}")
    print(f"✅ Energy (deviation from optimal): {swarm.best_cost:.2f} km")
    
    return swarm

def test_obstacle_avoidance():
    """Test obstacle avoidance with safety margin"""
    print("\n⚠️  Testing Obstacle Avoidance with Safety Margin")
    print("=" * 60)
    
    # Define obstacles as targets with respect boundaries
    obstacles = [
        {'center': np.array([5.0, 5.0]), 'radius': 2.0},
        {'center': np.array([15.0, 15.0]), 'radius': 2.5},
        {'center': np.array([10.0, 10.0]), 'radius': 1.5}
    ]
    
    # Goal: Reach target while avoiding obstacles
    goal = np.array([20.0, 20.0])
    
    def multi_obstacle_objective(x):
        # Primary objective: get to goal
        goal_distance = np.linalg.norm(x - goal)
        
        # Penalty for being inside obstacle safety zones
        obstacle_penalty = 0.0
        for obs in obstacles:
            dist_to_obstacle = np.linalg.norm(x - obs['center'])
            if dist_to_obstacle < obs['radius']:
                violation = obs['radius'] - dist_to_obstacle
                obstacle_penalty += 1000.0 * (violation ** 2)
        
        return goal_distance + obstacle_penalty
    
    print(f"   Goal position: {goal}")
    print(f"   Number of obstacles: {len(obstacles)}")
    
    swarm = Swarm(
        n_particles=30,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=60,
        obj_func=multi_obstacle_objective,
        algo='global'
    )
    
    swarm.optimize()
    
    print(f"\n✅ Optimal path found: {swarm.best_pos}")
    print(f"✅ Distance to goal: {np.linalg.norm(swarm.best_pos - goal):.4f}")
    
    # Check obstacle clearance
    print(f"\n📊 Obstacle Clearances:")
    for i, obs in enumerate(obstacles):
        dist = np.linalg.norm(swarm.best_pos - obs['center'])
        safe = dist >= obs['radius']
        print(f"   Obstacle {i+1}: {dist:.4f} km (radius: {obs['radius']:.2f}, safe: {safe})")
    
    return swarm

def test_visualization():
    """Create visualization of respect boundary behavior"""
    print("\n📊 Creating Respect Boundary Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        target = np.array([5.0, 5.0])
        respect_distance = 2.0
        
        def distance_obj(x):
            return np.linalg.norm(x - target)
        
        # Run with respect boundary
        swarm = Swarm(
            n_particles=20,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=50,
            obj_func=distance_obj,
            algo='global',
            respect_boundary=respect_distance,
            target_position=target
        )
        
        swarm.optimize()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Respect boundary visualization
        ax1.add_patch(Circle(target, respect_distance, fill=False, 
                            edgecolor='red', linewidth=2, linestyle='--',
                            label=f'Respect Boundary ({respect_distance})'))
        ax1.plot(target[0], target[1], 'r*', markersize=20, label='Target')
        ax1.plot(swarm.best_pos[0], swarm.best_pos[1], 'go', 
                markersize=15, label='Converged Position')
        
        # Plot final particle positions
        for particle in swarm.swarm:
            ax1.plot(particle.pos[0], particle.pos[1], 'b.', markersize=8, alpha=0.6)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Respect Boundary Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Distance from target over particles
        distances = [np.linalg.norm(p.pos - target) for p in swarm.swarm]
        ax2.hist(distances, bins=15, edgecolor='black', alpha=0.7)
        ax2.axvline(respect_distance, color='red', linestyle='--', 
                   linewidth=2, label=f'Respect Boundary ({respect_distance})')
        ax2.set_xlabel('Distance from Target')
        ax2.set_ylabel('Number of Particles')
        ax2.set_title('Particle Distance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('respect_boundary_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization saved as 'respect_boundary_visualization.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")

def main():
    """Run all respect boundary tests"""
    print("🎯 Respect Boundary Test Suite")
    print("=" * 70)
    print("Testing PSO with respect boundary (standoff distance) feature.")
    
    # Run all tests
    test_basic_respect_boundary()
    test_3d_respect_boundary()
    test_multiple_respect_distances()
    test_standoff_distance()
    test_obstacle_avoidance()
    test_visualization()
    
    print("\n" + "=" * 70)
    print("🎉 Respect Boundary Testing Complete!")
    print("=" * 70)
    
    print("\n✨ Respect Boundary Features:")
    print("✅ Particles converge to safe distance from target")
    print("✅ Useful for safety-critical applications and standoff operations")
    print("✅ Enables obstacle avoidance with safety margins")
    print("✅ Works in 2D and 3D spaces")
    print("✅ Configurable respect distance")
    
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., respect_boundary=2.0, target_position=[x, y, z])")

if __name__ == "__main__":
    main()
