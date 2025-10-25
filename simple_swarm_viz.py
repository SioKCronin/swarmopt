#!/usr/bin/env python3
"""
Simple Swarm Visualization - Create videos of PSO optimization

This is a simplified version that focuses on creating beautiful,
educational videos of particle swarm optimization.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def create_swarm_video(obj_func, func_name, algo='global', inertia='linear', 
                      clamping='basic', n_particles=20, epochs=50, 
                      filename=None):
    """
    Create a video of particle swarm optimization
    
    Parameters:
    -----------
    obj_func : function
        Objective function to optimize
    func_name : str
        Name of the function for display
    algo : str
        PSO algorithm ('global', 'local', 'unified')
    inertia : str
        Inertia weight strategy
    clamping : str
        Velocity clamping strategy
    n_particles : int
        Number of particles
    epochs : int
        Number of epochs
    filename : str
        Output filename (auto-generated if None)
    """
    
    if not HAS_MATPLOTLIB:
        print("❌ Matplotlib not available!")
        print("Install with: pip install matplotlib")
        return None
    
    if filename is None:
        filename = f"swarm_{func_name}_{algo}_{inertia}_{clamping}.mp4"
    
    print(f"🎬 Creating {func_name} optimization video...")
    print(f"   Algorithm: {algo}, Inertia: {inertia}, Clamping: {clamping}")
    
    # Create swarm
    swarm = Swarm(
        n_particles=n_particles,
        dims=2,
        c1=2.0, c2=2.0, w=0.9,
        epochs=epochs,
        obj_func=obj_func,
        algo=algo,
        velocity_clamp=(-5, 5),
        inertia_func=inertia,
        velocity_clamp_func=clamping
    )
    
    # Capture optimization process
    positions_history = []
    best_positions_history = []
    costs_history = []
    
    # Store initial state
    positions_history.append([p.pos.copy() for p in swarm.swarm])
    best_positions_history.append(swarm.best_pos.copy())
    costs_history.append(swarm.best_cost)
    
    # Run optimization with tracking
    for epoch in range(epochs):
        for particle in swarm.swarm:
            particle.update(epoch)
        
        swarm.update_local_best_pos()
        swarm.update_global_best_pos()
        swarm.update_global_worst_pos()
        
        # Store state
        positions_history.append([p.pos.copy() for p in swarm.swarm])
        best_positions_history.append(swarm.best_pos.copy())
        costs_history.append(swarm.best_cost)
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Determine bounds
    all_positions = np.concatenate(positions_history)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    
    # Add padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Setup main plot
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Setup cost plot
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Best Cost')
    ax2.grid(True, alpha=0.3)
    
    # Initialize plot elements
    particles_scatter = ax1.scatter([], [], c='blue', s=60, alpha=0.8, label='Particles')
    best_scatter = ax1.scatter([], [], c='red', s=150, marker='*', label='Best Position', zorder=5)
    cost_line, = ax2.plot([], [], 'b-', linewidth=2)
    
    # Add optimal point (if known)
    if func_name == 'Sphere':
        ax1.scatter(0, 0, c='gold', s=200, marker='X', label='Optimal', zorder=5)
    elif func_name == 'Rosenbrock':
        ax1.scatter(1, 1, c='gold', s=200, marker='X', label='Optimal', zorder=5)
    elif func_name == 'Ackley':
        ax1.scatter(0, 0, c='gold', s=200, marker='X', label='Optimal', zorder=5)
    
    ax1.legend()
    
    def animate(frame):
        if frame >= len(positions_history):
            return particles_scatter, best_scatter, cost_line
        
        # Current state
        current_positions = np.array(positions_history[frame])
        best_pos = best_positions_history[frame]
        current_cost = costs_history[frame]
        
        # Update particles
        particles_scatter.set_offsets(current_positions)
        best_scatter.set_offsets([best_pos])
        
        # Update cost plot
        epochs = list(range(frame + 1))
        costs = costs_history[:frame+1]
        cost_line.set_data(epochs, costs)
        ax2.set_xlim(0, len(positions_history))
        if costs:
            ax2.set_ylim(min(costs) * 0.9, max(costs) * 1.1)
        
        # Update title
        ax1.set_title(f'{func_name} - {algo.upper()} PSO (Epoch {frame}, Cost: {current_cost:.6f})')
        
        return particles_scatter, best_scatter, cost_line
    
    # Create and save animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(positions_history),
        interval=200, blit=False, repeat=True
    )
    
    print(f"💾 Saving to {filename}...")
    anim.save(filename, writer='ffmpeg', fps=5, bitrate=1800)
    print(f"✅ Video saved!")
    
    return anim

def create_demo_videos():
    """Create a set of demonstration videos"""
    print("🎬 Creating SwarmOpt Demo Videos")
    print("=" * 50)
    
    # Demo configurations
    demos = [
        # (function, name, algorithm, inertia, clamping)
        (sphere, 'Sphere', 'global', 'linear', 'basic'),
        (sphere, 'Sphere', 'global', 'adaptive', 'hybrid'),
        (rosenbrock, 'Rosenbrock', 'global', 'exponential', 'adaptive'),
        (ackley, 'Ackley', 'local', 'chaotic', 'exponential'),
        (sphere, 'Sphere', 'unified', 'linear', 'soft'),
    ]
    
    for i, (func, name, algo, inertia, clamping) in enumerate(demos):
        print(f"\n{i+1}. Creating {name} with {algo} PSO...")
        create_swarm_video(
            func, name, algo, inertia, clamping,
            n_particles=15, epochs=40,
            filename=f"demo_{i+1}_{name.lower()}_{algo}_{inertia}.mp4"
        )
    
    print("\n" + "=" * 50)
    print("🎉 Demo videos created!")
    print("📁 Files created:")
    for i in range(len(demos)):
        print(f"  - demo_{i+1}_*.mp4")

def create_algorithm_comparison():
    """Create side-by-side comparison of algorithms"""
    print("🎬 Creating Algorithm Comparison")
    print("=" * 50)
    
    algorithms = ['global', 'local', 'unified']
    algorithm_names = ['Global Best', 'Local Best', 'Unified']
    
    for algo, name in zip(algorithms, algorithm_names):
        print(f"Creating {name} PSO video...")
        create_swarm_video(
            sphere, 'Sphere', algo, 'linear', 'basic',
            n_particles=20, epochs=50,
            filename=f"comparison_{algo}_pso.mp4"
        )
    
    print("✅ Algorithm comparison videos created!")

def create_inertia_comparison():
    """Create comparison of inertia weight strategies"""
    print("🎬 Creating Inertia Weight Comparison")
    print("=" * 50)
    
    inertia_types = ['linear', 'exponential', 'adaptive', 'chaotic']
    inertia_names = ['Linear', 'Exponential', 'Adaptive', 'Chaotic']
    
    for inertia, name in zip(inertia_types, inertia_names):
        print(f"Creating {name} inertia video...")
        create_swarm_video(
            rosenbrock, 'Rosenbrock', 'global', inertia, 'basic',
            n_particles=15, epochs=40,
            filename=f"inertia_{inertia}_comparison.mp4"
        )
    
    print("✅ Inertia comparison videos created!")

def main():
    """Main function"""
    print("🎬 SwarmOpt Visualization Suite")
    print("=" * 50)
    
    if not HAS_MATPLOTLIB:
        print("❌ Matplotlib not available!")
        print("Install with: pip install matplotlib")
        print("\nAlternatively, you can install all visualization dependencies with:")
        print("pip install matplotlib ffmpeg-python")
        return
    
    print("Choose visualization type:")
    print("1. Demo videos (5 different configurations)")
    print("2. Algorithm comparison (Global vs Local vs Unified)")
    print("3. Inertia weight comparison (4 different strategies)")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        create_demo_videos()
    elif choice == '2':
        create_algorithm_comparison()
    elif choice == '3':
        create_inertia_comparison()
    elif choice == '4':
        create_demo_videos()
        create_algorithm_comparison()
        create_inertia_comparison()
    else:
        print("Invalid choice. Creating demo videos...")
        create_demo_videos()
    
    print("\n🎉 Visualization complete!")
    print("💡 These videos are perfect for:")
    print("  - Educational demonstrations")
    print("  - Algorithm comparisons")
    print("  - Understanding PSO behavior")
    print("  - Sharing with colleagues and students")

if __name__ == "__main__":
    main()
