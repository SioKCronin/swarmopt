#!/usr/bin/env python3
"""
Respect Boundary Visualization

Creates beautiful visualizations demonstrating the respect boundary feature,
including a satellite positioning example.
"""

import numpy as np
import sys
import os
# Add parent directory to path to import swarmopt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarmopt.swarm import Swarm

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation, PillowWriter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

def create_2d_respect_boundary_viz():
    """Create 2D visualization of respect boundary convergence"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print("🎨 Creating 2D Respect Boundary Visualization...")
    
    # Define target and objective
    target = np.array([5.0, 5.0])
    
    def distance_to_target(x):
        return np.linalg.norm(x - target)
    
    # Run optimization WITH respect boundary
    swarm = Swarm(
        n_particles=25,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=50,
        obj_func=distance_to_target,
        algo='global',
        target_position=target,
        velocity_clamp=(-10, 10)
    )
    
    swarm.optimize()
    respect_distance = swarm.respect_boundary
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot respect boundary circle
    boundary_circle = Circle(target, respect_distance, fill=False, 
                            edgecolor='red', linewidth=3, linestyle='--',
                            label=f'Respect Boundary ({respect_distance:.2f})',
                            zorder=5)
    ax.add_patch(boundary_circle)
    
    # Plot target
    ax.plot(target[0], target[1], 'r*', markersize=30, 
           label='Target (Unsafe Zone)', zorder=10)
    
    # Plot final particle positions
    particle_distances = []
    for i, particle in enumerate(swarm.swarm):
        dist = np.linalg.norm(particle.pos - target)
        particle_distances.append(dist)
        
        # Color by distance
        if dist >= respect_distance:
            color = 'green'
            alpha = 0.7
        else:
            color = 'orange'
            alpha = 0.5
        
        ax.plot(particle.pos[0], particle.pos[1], 'o', 
               color=color, markersize=10, alpha=alpha, zorder=3)
    
    # Plot best position
    ax.plot(swarm.best_pos[0], swarm.best_pos[1], 'go', 
           markersize=20, label='Best Position', 
           markeredgecolor='darkgreen', markeredgewidth=2, zorder=8)
    
    # Add distance annotation
    best_dist = np.linalg.norm(swarm.best_pos - target)
    ax.annotate(f'Distance: {best_dist:.2f}',
               xy=swarm.best_pos, xytext=(swarm.best_pos[0]+1, swarm.best_pos[1]+1),
               fontsize=12, fontweight='bold',
               arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    # Styling
    ax.set_xlabel('X Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=14, fontweight='bold')
    ax.set_title('Respect Boundary: Particles Converge to Safe Distance\n' +
                f'Target at {target}, Respect Boundary = {respect_distance:.2f}',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add statistics text
    violations = sum(1 for d in particle_distances if d < respect_distance)
    stats_text = f'Statistics:\n'
    stats_text += f'Particles: {len(swarm.swarm)}\n'
    stats_text += f'Boundary violations: {violations}\n'
    stats_text += f'Compliance rate: {(1 - violations/len(swarm.swarm))*100:.1f}%\n'
    stats_text += f'Min distance: {min(particle_distances):.2f}\n'
    stats_text += f'Avg distance: {np.mean(particle_distances):.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = 'swarm_visualizations/respect_boundary_2d.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: respect_boundary_2d.png")
    return swarm

def create_comparison_viz():
    """Create comparison: with vs without respect boundary"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    print("📊 Creating Comparison Visualization...")
    
    target = np.array([10.0, 10.0])
    
    def distance_obj(x):
        return np.linalg.norm(x - target)
    
    # WITHOUT respect boundary
    swarm_without = Swarm(
        n_particles=20,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=distance_obj,
        algo='global',
        velocity_clamp=(-10, 10)
    )
    swarm_without.optimize()
    
    # WITH respect boundary
    swarm_with = Swarm(
        n_particles=20,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=40,
        obj_func=distance_obj,
        algo='global',
        target_position=target,
        velocity_clamp=(-10, 10)
    )
    swarm_with.optimize()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # LEFT: Without respect boundary
    ax1.plot(target[0], target[1], 'r*', markersize=30, label='Target', zorder=10)
    
    for particle in swarm_without.swarm:
        ax1.plot(particle.pos[0], particle.pos[1], 'bo', 
                markersize=10, alpha=0.6, zorder=3)
    
    ax1.plot(swarm_without.best_pos[0], swarm_without.best_pos[1], 'go',
            markersize=20, label='Best Position',
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=8)
    
    dist_without = np.linalg.norm(swarm_without.best_pos - target)
    ax1.set_title(f'WITHOUT Respect Boundary\nConverged Distance: {dist_without:.4f}',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # RIGHT: With respect boundary
    respect_distance = swarm_with.respect_boundary
    boundary_circle = Circle(target, respect_distance, fill=False,
                            edgecolor='red', linewidth=3, linestyle='--',
                            label=f'Respect Boundary ({respect_distance:.2f})',
                            zorder=5)
    ax2.add_patch(boundary_circle)
    
    ax2.plot(target[0], target[1], 'r*', markersize=30, label='Target', zorder=10)
    
    for particle in swarm_with.swarm:
        dist = np.linalg.norm(particle.pos - target)
        color = 'green' if dist >= respect_distance else 'orange'
        ax2.plot(particle.pos[0], particle.pos[1], 'o',
                color=color, markersize=10, alpha=0.6, zorder=3)
    
    ax2.plot(swarm_with.best_pos[0], swarm_with.best_pos[1], 'go',
            markersize=20, label='Best Position',
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=8)
    
    dist_with = np.linalg.norm(swarm_with.best_pos - target)
    ax2.set_title(f'WITH Respect Boundary\nConverged Distance: {dist_with:.4f}',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.suptitle('Respect Boundary Comparison\n' +
                 'Safety-Critical Distance Enforcement',
                 fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    output_path = 'swarm_visualizations/respect_boundary_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: respect_boundary_comparison.png")
    return swarm_without, swarm_with

def main():
    """Generate all respect boundary visualizations"""
    print("=" * 70)
    print("🎯 Respect Boundary Visualization Generator")
    print("=" * 70)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\n❌ Matplotlib is required for visualizations")
        print("Install with: pip install matplotlib")
        return
    
    print("\nGenerating visualizations...\n")
    
    # Create visualizations
    create_2d_respect_boundary_viz()
    create_comparison_viz()
    
    print("\n" + "=" * 70)
    print("✅ All Visualizations Generated!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • respect_boundary_2d.png - Basic 2D convergence")
    print("  • respect_boundary_comparison.png - With vs without comparison")
    print("\n💡 These visualizations demonstrate:")
    print("  ✓ Safe distance enforcement")
    print("  ✓ Real-world safety-critical applications")
    print("  ✓ Comparison of convergence behavior")

if __name__ == "__main__":
    main()

