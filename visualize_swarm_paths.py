#!/usr/bin/env python3
"""
Swarm Path Visualization - Create videos of particle swarm optimization paths

This script creates animated visualizations of PSO algorithms showing:
- Particle trajectories over time
- Best position evolution
- Convergence patterns
- Different algorithm behaviors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley, rastrigin

class SwarmVisualizer:
    def __init__(self, swarm, save_path="swarm_animation.mp4", fps=10):
        """
        Initialize the swarm visualizer
        
        Parameters:
        -----------
        swarm : Swarm
            The swarm object to visualize
        save_path : str
            Path to save the animation
        fps : int
            Frames per second for the animation
        """
        self.swarm = swarm
        self.save_path = save_path
        self.fps = fps
        
        # Store particle positions over time
        self.positions_history = []
        self.best_positions_history = []
        self.costs_history = []
        
        # Capture the optimization process
        self._capture_optimization()
        
    def _capture_optimization(self):
        """Capture particle positions during optimization"""
        print("🎬 Capturing optimization process...")
        
        # Store initial state
        initial_positions = [particle.pos.copy() for particle in self.swarm.swarm]
        self.positions_history.append(initial_positions)
        self.best_positions_history.append(self.swarm.best_pos.copy())
        self.costs_history.append(self.swarm.best_cost)
        
        # Run optimization with position tracking
        for epoch in range(self.swarm.epochs):
            # Update particles
            for particle in self.swarm.swarm:
                particle.update(epoch)
            
            # Update swarm state
            self.swarm.update_local_best_pos()
            self.swarm.update_global_best_pos()
            self.swarm.update_global_worst_pos()
            
            # Store current state
            current_positions = [particle.pos.copy() for particle in self.swarm.swarm]
            self.positions_history.append(current_positions)
            self.best_positions_history.append(self.swarm.best_pos.copy())
            self.costs_history.append(self.swarm.best_cost)
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{self.swarm.epochs} - Best cost: {self.swarm.best_cost:.6f}")
    
    def create_animation(self, show_trails=True, trail_length=20):
        """
        Create animated visualization of the swarm optimization
        
        Parameters:
        -----------
        show_trails : bool
            Whether to show particle trails
        trail_length : int
            Length of particle trails
        """
        print(f"🎨 Creating animation: {self.save_path}")
        
        # Set up the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Determine plot bounds
        all_positions = np.concatenate(self.positions_history)
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        
        # Add some padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Set up the main plot (particle positions)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Particle Swarm Optimization - {self.swarm.algo.upper()}')
        ax1.grid(True, alpha=0.3)
        
        # Set up the cost plot
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Best Cost')
        ax2.set_title('Convergence')
        ax2.grid(True, alpha=0.3)
        
        # Initialize plot elements
        particles_scatter = ax1.scatter([], [], c='blue', s=50, alpha=0.7, label='Particles')
        best_scatter = ax1.scatter([], [], c='red', s=100, marker='*', label='Best Position')
        
        # Trail lines for particles
        trail_lines = []
        if show_trails:
            for i in range(len(self.swarm.swarm)):
                line, = ax1.plot([], [], alpha=0.3, linewidth=1)
                trail_lines.append(line)
        
        # Global best trail
        global_best_line, = ax1.plot([], [], 'r-', linewidth=2, alpha=0.8, label='Best Path')
        
        # Cost line
        cost_line, = ax2.plot([], [], 'b-', linewidth=2)
        
        # Add optimal point if known
        if hasattr(self.swarm, 'optimal_pos'):
            ax1.scatter(self.swarm.optimal_pos[0], self.swarm.optimal_pos[1], 
                       c='gold', s=200, marker='X', label='Optimal', zorder=5)
        
        ax1.legend()
        
        def animate(frame):
            """Animation function"""
            if frame >= len(self.positions_history):
                return particles_scatter, best_scatter, global_best_line, cost_line
            
            # Current positions
            current_positions = np.array(self.positions_history[frame])
            best_pos = self.best_positions_history[frame]
            current_cost = self.costs_history[frame]
            
            # Update particle positions
            particles_scatter.set_offsets(current_positions)
            
            # Update best position
            best_scatter.set_offsets([best_pos])
            
            # Update trails
            if show_trails:
                start_frame = max(0, frame - trail_length)
                for i, line in enumerate(trail_lines):
                    trail_x = [pos[i][0] for pos in self.positions_history[start_frame:frame+1]]
                    trail_y = [pos[i][1] for pos in self.positions_history[start_frame:frame+1]]
                    line.set_data(trail_x, trail_y)
            
            # Update global best trail
            best_trail_x = [pos[0] for pos in self.best_positions_history[:frame+1]]
            best_trail_y = [pos[1] for pos in self.best_positions_history[:frame+1]]
            global_best_line.set_data(best_trail_x, best_trail_y)
            
            # Update cost plot
            epochs = list(range(frame + 1))
            costs = self.costs_history[:frame+1]
            cost_line.set_data(epochs, costs)
            ax2.set_xlim(0, len(self.positions_history))
            ax2.set_ylim(min(costs) * 0.9, max(costs) * 1.1)
            
            # Update title with current info
            ax1.set_title(f'PSO - {self.swarm.algo.upper()} (Epoch {frame}, Cost: {current_cost:.6f})')
            
            return particles_scatter, best_scatter, global_best_line, cost_line
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.positions_history),
            interval=1000//self.fps, blit=False, repeat=True
        )
        
        # Save animation
        print(f"💾 Saving animation to {self.save_path}...")
        anim.save(self.save_path, writer='ffmpeg', fps=self.fps, bitrate=1800)
        print(f"✅ Animation saved!")
        
        return anim

def create_comparison_video():
    """Create a comparison video showing different algorithms"""
    print("🎬 Creating Algorithm Comparison Video")
    print("=" * 50)
    
    # Test parameters
    n_particles = 20
    dims = 2
    epochs = 50
    obj_func = sphere
    
    algorithms = ['global', 'local', 'unified']
    algorithm_names = ['Global Best PSO', 'Local Best PSO', 'Unified PSO']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (algo, name) in enumerate(zip(algorithms, algorithm_names)):
        print(f"  Processing {name}...")
        
        # Create swarm
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0, c2=2.0, w=0.9,
            epochs=epochs,
            obj_func=obj_func,
            algo=algo,
            velocity_clamp=(-5, 5),
            inertia_func='linear',
            velocity_clamp_func='basic'
        )
        
        # Create visualizer
        visualizer = SwarmVisualizer(swarm, f"comparison_{algo}.mp4")
        
        # Create subplot
        ax = axes[i]
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        # Plot final result
        final_positions = np.array(visualizer.positions_history[-1])
        best_pos = visualizer.best_positions_history[-1]
        
        ax.scatter(final_positions[:, 0], final_positions[:, 1], 
                  c='blue', s=50, alpha=0.7, label='Final Positions')
        ax.scatter(best_pos[0], best_pos[1], c='red', s=100, 
                  marker='*', label='Best Position')
        ax.scatter(0, 0, c='gold', s=200, marker='X', label='Optimal')
        
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Comparison plot saved as 'algorithm_comparison.png'")

def create_inertia_comparison():
    """Create comparison of different inertia weight strategies"""
    print("🎬 Creating Inertia Weight Comparison")
    print("=" * 50)
    
    # Test parameters
    n_particles = 15
    dims = 2
    epochs = 40
    obj_func = rosenbrock
    
    inertia_types = ['linear', 'exponential', 'adaptive', 'chaotic']
    inertia_names = ['Linear', 'Exponential', 'Adaptive', 'Chaotic']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (inertia_type, name) in enumerate(zip(inertia_types, inertia_names)):
        print(f"  Processing {name} Inertia...")
        
        # Create swarm
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0, c2=2.0, w=0.9,
            epochs=epochs,
            obj_func=obj_func,
            algo='global',
            velocity_clamp=(-2, 2),
            inertia_func=inertia_type,
            w_start=0.9, w_end=0.4,
            velocity_clamp_func='basic'
        )
        
        # Create visualizer
        visualizer = SwarmVisualizer(swarm, f"inertia_{inertia_type}.mp4")
        
        # Plot trajectories
        ax = axes[i]
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f'{name} Inertia Weight')
        ax.grid(True, alpha=0.3)
        
        # Plot particle trails
        for j in range(min(5, n_particles)):  # Show first 5 particles
            trail_x = [pos[j][0] for pos in visualizer.positions_history]
            trail_y = [pos[j][1] for pos in visualizer.positions_history]
            ax.plot(trail_x, trail_y, alpha=0.5, linewidth=1)
        
        # Plot best path
        best_trail_x = [pos[0] for pos in visualizer.best_positions_history]
        best_trail_y = [pos[1] for pos in visualizer.best_positions_history]
        ax.plot(best_trail_x, best_trail_y, 'r-', linewidth=2, label='Best Path')
        
        # Mark start and end
        start_pos = visualizer.positions_history[0][0]
        end_pos = visualizer.positions_history[-1][0]
        ax.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', label='Start')
        ax.scatter(end_pos[0], end_pos[1], c='red', s=100, marker='x', label='End')
        ax.scatter(1, 1, c='gold', s=200, marker='*', label='Optimal')
        
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('inertia_comparison.png', dpi=150, bbox_inings='tight')
    print("✅ Inertia comparison saved as 'inertia_comparison.png'")

def create_velocity_clamping_comparison():
    """Create comparison of different velocity clamping strategies"""
    print("🎬 Creating Velocity Clamping Comparison")
    print("=" * 50)
    
    # Test parameters
    n_particles = 15
    dims = 2
    epochs = 40
    obj_func = ackley
    
    clamping_types = ['none', 'basic', 'adaptive', 'exponential']
    clamping_names = ['No Clamping', 'Basic', 'Adaptive', 'Exponential']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (clamp_type, name) in enumerate(zip(clamping_types, clamping_names)):
        print(f"  Processing {name} Clamping...")
        
        # Create swarm
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0, c2=2.0, w=0.9,
            epochs=epochs,
            obj_func=obj_func,
            algo='global',
            velocity_clamp=(-5, 5),
            inertia_func='linear',
            velocity_clamp_func=clamp_type
        )
        
        # Create visualizer
        visualizer = SwarmVisualizer(swarm, f"clamping_{clamp_type}.mp4")
        
        # Plot trajectories
        ax = axes[i]
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f'{name} Velocity Clamping')
        ax.grid(True, alpha=0.3)
        
        # Plot particle trails
        for j in range(min(5, n_particles)):  # Show first 5 particles
            trail_x = [pos[j][0] for pos in visualizer.positions_history]
            trail_y = [pos[j][1] for pos in visualizer.positions_history]
            ax.plot(trail_x, trail_y, alpha=0.5, linewidth=1)
        
        # Plot best path
        best_trail_x = [pos[0] for pos in visualizer.best_positions_history]
        best_trail_y = [pos[1] for pos in visualizer.best_positions_history]
        ax.plot(best_trail_x, best_trail_y, 'r-', linewidth=2, label='Best Path')
        
        # Mark start and end
        start_pos = visualizer.positions_history[0][0]
        end_pos = visualizer.positions_history[-1][0]
        ax.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', label='Start')
        ax.scatter(end_pos[0], end_pos[1], c='red', s=100, marker='x', label='End')
        ax.scatter(0, 0, c='gold', s=200, marker='*', label='Optimal')
        
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('clamping_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Clamping comparison saved as 'clamping_comparison.png'")

def main():
    """Main function to create all visualizations"""
    print("🎬 SwarmOpt Visualization Suite")
    print("=" * 50)
    
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    os.chdir('visualizations')
    
    print("\n1. Creating basic swarm animation...")
    # Basic swarm animation
    swarm = Swarm(
        n_particles=20, dims=2, c1=2.0, c2=2.0, w=0.9,
        epochs=50, obj_func=sphere, algo='global',
        velocity_clamp=(-5, 5), inertia_func='linear',
        velocity_clamp_func='basic'
    )
    visualizer = SwarmVisualizer(swarm, 'basic_swarm.mp4')
    visualizer.create_animation()
    
    print("\n2. Creating algorithm comparison...")
    create_comparison_video()
    
    print("\n3. Creating inertia weight comparison...")
    create_inertia_comparison()
    
    print("\n4. Creating velocity clamping comparison...")
    create_velocity_clamping_comparison()
    
    print("\n" + "=" * 50)
    print("🎉 All visualizations complete!")
    print("📁 Check the 'visualizations' directory for:")
    print("  - basic_swarm.mp4")
    print("  - comparison_*.mp4")
    print("  - inertia_*.mp4") 
    print("  - clamping_*.mp4")
    print("  - *_comparison.png")
    print("\n💡 These videos are perfect for:")
    print("  - Educational demonstrations")
    print("  - Algorithm comparisons")
    print("  - Understanding PSO behavior")
    print("  - Sharing with colleagues")

if __name__ == "__main__":
    main()
