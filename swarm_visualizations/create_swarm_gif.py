#!/usr/bin/env python3
"""
Create Swarm GIF - Create animated GIFs of particle swarm optimization

This creates GIF animations that work without ffmpeg.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley

def create_swarm_gif(obj_func, func_name, algo='global', inertia='linear', 
                     clamping='basic', n_particles=15, epochs=30, 
                     filename=None):
    """Create a GIF animation of particle swarm optimization"""
    
    if filename is None:
        filename = f"swarm_{func_name}_{algo}_{inertia}_{clamping}.gif"
    
    print(f"🎬 Creating {func_name} optimization GIF...")
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
    
    print(f"   Initial best cost: {swarm.best_cost:.6f}")
    
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
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch:2d}: Best cost = {swarm.best_cost:.6f}")
    
    print(f"   Final best cost: {swarm.best_cost:.6f}")
    
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
    
    # Create and save animation as GIF
    anim = animation.FuncAnimation(
        fig, animate, frames=len(positions_history),
        interval=300, blit=False, repeat=True
    )
    
    print(f"💾 Saving GIF to {filename}...")
    anim.save(filename, writer='pillow', fps=3)
    print(f"✅ GIF saved!")
    
    return anim

def create_demo_gifs():
    """Create demonstration GIFs"""
    print("🎬 Creating SwarmOpt Demo GIFs")
    print("=" * 50)
    
    # Demo configurations
    demos = [
        (sphere, 'Sphere', 'global', 'linear', 'basic'),
        (sphere, 'Sphere', 'global', 'adaptive', 'hybrid'),
        (rosenbrock, 'Rosenbrock', 'global', 'exponential', 'adaptive'),
        (ackley, 'Ackley', 'local', 'chaotic', 'exponential'),
    ]
    
    for i, (func, name, algo, inertia, clamping) in enumerate(demos):
        print(f"\n{i+1}. Creating {name} GIF...")
        create_swarm_gif(
            func, name, algo, inertia, clamping,
            n_particles=12, epochs=25,
            filename=f"demo_{i+1}_{name.lower()}_{algo}_{inertia}.gif"
        )
    
    print("\n" + "=" * 50)
    print("🎉 Demo GIFs created!")
    print("📁 Files created:")
    for i in range(len(demos)):
        print(f"  - demo_{i+1}_*.gif")

def main():
    """Main function"""
    print("🎬 SwarmOpt GIF Creator")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("❌ Matplotlib not available!")
        print("Install with: pip install matplotlib")
        return
    
    print("Creating demonstration GIFs...")
    create_demo_gifs()
    
    print("\n🎉 GIF creation complete!")
    print("💡 These GIFs are perfect for:")
    print("  - Educational demonstrations")
    print("  - Algorithm comparisons")
    print("  - Understanding PSO behavior")
    print("  - Sharing with colleagues and students")

if __name__ == "__main__":
    main()
