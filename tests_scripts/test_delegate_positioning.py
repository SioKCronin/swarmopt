#!/usr/bin/env python3
"""
Test Delegate Positioning

Demonstrates positioning delegate agents (like repair drones or service satellites)
at specific polar positions around a target with automatic respect boundary.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarmopt import Swarm

def test_2d_delegates():
    """Test 2 delegates in 2D (satellite repair scenario)"""
    print("🛰️  Test 1: 2D Satellite Repair with 2 Delegates")
    print("=" * 60)
    
    # Target satellite position
    target = np.array([10.0, 10.0])
    
    def distance_to_target(x):
        return np.linalg.norm(x - target)
    
    # Create swarm with 2 delegate positions for redundancy
    swarm = Swarm(
        n_particles=20,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=30,
        obj_func=distance_to_target,
        algo='global',
        target_position=target,
        n_delegates=2,  # 2 repair drones for redundancy
        delegate_spread='opposite'  # Position opposite each other
    )
    
    print(f"✅ Target: {target}")
    print(f"✅ Respect boundary: {swarm.respect_boundary:.4f}")
    print(f"✅ Number of delegates: {len(swarm.delegate_positions)}")
    print(f"\n📍 Delegate Positions:")
    for i, pos in enumerate(swarm.delegate_positions):
        dist = np.linalg.norm(pos - target)
        angle = np.arctan2(pos[1] - target[1], pos[0] - target[0]) * 180 / np.pi
        print(f"   Delegate {i+1}: {pos} (distance: {dist:.4f}, angle: {angle:.1f}°)")
    
    # Run optimization
    swarm.optimize()
    
    # Get delegate assignments
    assignments = swarm.get_delegate_assignments()
    print(f"\n🎯 Optimized Delegate Assignments:")
    for delegate_idx, info in assignments.items():
        print(f"   Delegate {delegate_idx+1} → Particle {info['particle_index']}")
        print(f"      Target position: {info['target_pos']}")
        print(f"      Particle position: {info['particle_pos']}")
        print(f"      Distance to target: {info['distance']:.4f}")
    
    return swarm

def test_3d_satellite_servicing():
    """Test 3D satellite servicing with multiple delegates"""
    print("\n🛰️  Test 2: 3D Satellite Servicing with 4 Delegates")
    print("=" * 60)
    
    # ISS-like scenario
    earth_center = np.array([0.0, 0.0, 0.0])
    iss_position = np.array([6771.0, 0.0, 0.0])  # ISS orbital distance
    
    def approach_iss(x):
        # Objective: Get close to ISS while staying outside respect boundary
        return np.linalg.norm(x - iss_position)
    
    swarm = Swarm(
        n_particles=30,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=approach_iss,
        algo='global',
        target_position=iss_position,
        n_delegates=4,  # 4 service satellites around ISS
        delegate_spread='uniform'  # Evenly distributed
    )
    
    print(f"✅ ISS Position: {iss_position}")
    print(f"✅ Respect boundary: {swarm.respect_boundary:.2f} km")
    print(f"✅ Number of service satellites: {len(swarm.delegate_positions)}")
    print(f"\n📍 Service Satellite Positions:")
    for i, pos in enumerate(swarm.delegate_positions):
        dist_from_iss = np.linalg.norm(pos - iss_position)
        dist_from_earth = np.linalg.norm(pos - earth_center)
        print(f"   Satellite {i+1}: distance from ISS: {dist_from_iss:.2f} km, from Earth: {dist_from_earth:.2f} km")
    
    swarm.optimize()
    
    assignments = swarm.get_delegate_assignments()
    print(f"\n🎯 Service Satellite Assignments:")
    for delegate_idx, info in assignments.items():
        print(f"   Service Sat {delegate_idx+1} → Particle {info['particle_index']} (error: {info['distance']:.2f} km)")
    
    return swarm

def test_drone_formation():
    """Test drone formation around target"""
    print("\n🚁 Test 3: Drone Formation with 6 Delegates")
    print("=" * 60)
    
    # Target building or area
    target = np.array([50.0, 50.0, 20.0])  # x, y, altitude
    
    def observe_target(x):
        # Drones want to observe target from optimal distance
        return abs(np.linalg.norm(x - target) - 30.0)  # Optimal at 30m
    
    swarm = Swarm(
        n_particles=25,
        dims=3,
        c1=2.0, c2=2.0, w=0.9,
        epochs=35,
        obj_func=observe_target,
        algo='global',
        target_position=target,
        n_delegates=6,  # 6 observation drones
        delegate_spread='uniform'  # Uniform coverage
    )
    
    print(f"✅ Target location: {target}")
    print(f"✅ Respect boundary: {swarm.respect_boundary:.2f} m")
    print(f"✅ Number of drones: {len(swarm.delegate_positions)}")
    print(f"\n📍 Drone Positions (relative to target):")
    for i, pos in enumerate(swarm.delegate_positions):
        relative_pos = pos - target
        dist = np.linalg.norm(relative_pos)
        print(f"   Drone {i+1}: offset {relative_pos}, distance: {dist:.2f} m")
    
    swarm.optimize()
    
    assignments = swarm.get_delegate_assignments()
    print(f"\n🎯 Drone Formation Achieved:")
    print(f"   {len(assignments)} drones positioned around target")
    
    return swarm

def test_different_spreads():
    """Compare different delegate spread strategies"""
    print("\n📊 Test 4: Comparing Delegate Spread Strategies")
    print("=" * 60)
    
    target = np.array([0.0, 0.0])
    
    def dummy_objective(x):
        return np.linalg.norm(x)
    
    spreads = ['uniform', 'opposite', 'random']
    
    for spread in spreads:
        print(f"\n🔹 Spread strategy: {spread}")
        swarm = Swarm(
            n_particles=10,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=1,  # Just for initialization
            obj_func=dummy_objective,
            target_position=target,
            n_delegates=4,
            delegate_spread=spread
        )
        
        print(f"   Delegate positions:")
        for i, pos in enumerate(swarm.delegate_positions):
            angle = np.arctan2(pos[1], pos[0]) * 180 / np.pi
            print(f"      {i+1}. {pos} (angle: {angle:.1f}°)")

def visualize_delegates():
    """Create visualization of delegate positioning"""
    print("\n📊 Creating Delegate Positioning Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        target = np.array([0.0, 0.0])
        
        def distance_obj(x):
            return np.linalg.norm(x - target)
        
        # Create swarm with delegates
        swarm = Swarm(
            n_particles=20,
            dims=2,
            c1=2.0, c2=2.0, w=0.9,
            epochs=40,
            obj_func=distance_obj,
            algo='global',
            target_position=target,
            n_delegates=4,
            delegate_spread='uniform'
        )
        
        swarm.optimize()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot respect boundary
        boundary_circle = Circle(target, swarm.respect_boundary, fill=False,
                                edgecolor='red', linewidth=2, linestyle='--',
                                label=f'Respect Boundary ({swarm.respect_boundary:.2f})')
        ax.add_patch(boundary_circle)
        
        # Plot target
        ax.plot(target[0], target[1], 'r*', markersize=30, label='Target', zorder=10)
        
        # Plot delegate positions
        for i, delegate_pos in enumerate(swarm.delegate_positions):
            ax.plot(delegate_pos[0], delegate_pos[1], 'gs', 
                   markersize=20, label=f'Delegate {i+1}' if i < 4 else '',
                   markeredgecolor='darkgreen', markeredgewidth=2, zorder=9)
            
            # Draw line from target to delegate
            ax.plot([target[0], delegate_pos[0]], 
                   [target[1], delegate_pos[1]],
                   'g--', alpha=0.5, linewidth=1)
        
        # Plot particles
        for particle in swarm.swarm:
            dist = np.linalg.norm(particle.pos - target)
            color = 'blue' if dist >= swarm.respect_boundary else 'orange'
            ax.plot(particle.pos[0], particle.pos[1], 'o',
                   color=color, markersize=8, alpha=0.6, zorder=3)
        
        # Plot delegate assignments
        assignments = swarm.get_delegate_assignments()
        for delegate_idx, info in assignments.items():
            ax.plot([info['target_pos'][0], info['particle_pos'][0]],
                   [info['target_pos'][1], info['particle_pos'][1]],
                   'g-', linewidth=2, alpha=0.7, zorder=8)
        
        ax.set_xlabel('X Position', fontsize=14)
        ax.set_ylabel('Y Position', fontsize=14)
        ax.set_title('Delegate Positioning with Respect Boundary\n' +
                    f'{len(swarm.delegate_positions)} Delegates for Redundant Coverage',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig('delegate_positioning.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization saved as 'delegate_positioning.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")

def main():
    """Run all delegate positioning tests"""
    print("=" * 70)
    print("🎯 Delegate Positioning Test Suite")
    print("=" * 70)
    print("Testing polar positioning of delegates around targets")
    print("Use cases: Satellite servicing, drone formations, redundant coverage\n")
    
    test_2d_delegates()
    test_3d_satellite_servicing()
    test_drone_formation()
    test_different_spreads()
    visualize_delegates()
    
    print("\n" + "=" * 70)
    print("✅ Delegate Positioning Tests Complete!")
    print("=" * 70)
    
    print("\n✨ Features Demonstrated:")
    print("✓ 2D and 3D delegate positioning")
    print("✓ Uniform, opposite, and random spread strategies")
    print("✓ Automatic respect boundary enforcement")
    print("✓ Delegate-to-particle assignment")
    print("✓ Satellite servicing scenarios")
    print("✓ Drone formation control")
    
    print("\n🎯 Usage:")
    print("swarm = Swarm(..., target_position=[x,y,z], n_delegates=2, delegate_spread='opposite')")

if __name__ == "__main__":
    main()

