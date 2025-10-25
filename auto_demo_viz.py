#!/usr/bin/env python3
"""
Auto Demo Visualization - Automatically create swarm path demonstrations

This script automatically creates visualizations without user interaction.
"""

import numpy as np
import json
import os
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley

def capture_swarm_paths(obj_func, func_name, algo='global', inertia='linear', 
                       clamping='basic', n_particles=10, epochs=30):
    """Capture particle paths during optimization"""
    print(f"🎬 Capturing {func_name} optimization paths...")
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
    print(f"   Final best position: [{swarm.best_pos[0]:.3f}, {swarm.best_pos[1]:.3f}]")
    
    return {
        'function': func_name,
        'algorithm': algo,
        'inertia': inertia,
        'clamping': clamping,
        'n_particles': n_particles,
        'epochs': epochs,
        'positions': positions_history,
        'best_positions': best_positions_history,
        'costs': costs_history,
        'final_cost': swarm.best_cost,
        'final_position': swarm.best_pos.tolist()
    }

def create_visualization_files(data, prefix=""):
    """Create all visualization files for the data"""
    if prefix:
        prefix = f"{prefix}_"
    
    # Text visualization
    filename = f"{prefix}swarm_paths_{data['function']}_{data['algorithm']}.txt"
    print(f"📝 Creating text visualization: {filename}")
    
    with open(filename, 'w') as f:
        f.write(f"SwarmOpt Path Visualization\n")
        f.write(f"Function: {data['function']}\n")
        f.write(f"Algorithm: {data['algorithm']}\n")
        f.write(f"Inertia: {data['inertia']}\n")
        f.write(f"Clamping: {data['clamping']}\n")
        f.write(f"Particles: {data['n_particles']}\n")
        f.write(f"Epochs: {data['epochs']}\n")
        f.write(f"Final Cost: {data['final_cost']:.6f}\n")
        f.write(f"Final Position: [{data['final_position'][0]:.3f}, {data['final_position'][1]:.3f}]\n")
        f.write("=" * 50 + "\n\n")
        
        # Show particle paths
        for epoch in range(0, len(data['positions']), 5):  # Every 5th epoch
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Best Cost: {data['costs'][epoch]:.6f}\n")
            f.write(f"  Best Position: [{data['best_positions'][epoch][0]:.3f}, {data['best_positions'][epoch][1]:.3f}]\n")
            f.write(f"  Particle Positions:\n")
            
            for i, pos in enumerate(data['positions'][epoch]):
                f.write(f"    Particle {i:2d}: [{pos[0]:6.3f}, {pos[1]:6.3f}]\n")
            f.write("\n")
    
    # JSON data
    filename = f"{prefix}swarm_data_{data['function']}_{data['algorithm']}.json"
    print(f"💾 Saving JSON data: {filename}")
    
    json_data = {
        'function': data['function'],
        'algorithm': data['algorithm'],
        'inertia': data['inertia'],
        'clamping': data['clamping'],
        'n_particles': data['n_particles'],
        'epochs': data['epochs'],
        'positions': [[pos.tolist() for pos in epoch] for epoch in data['positions']],
        'best_positions': [pos.tolist() for pos in data['best_positions']],
        'costs': data['costs'],
        'final_cost': data['final_cost'],
        'final_position': data['final_position']
    }
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # CSV analysis
    filename = f"{prefix}swarm_analysis_{data['function']}_{data['algorithm']}.csv"
    print(f"📊 Creating CSV analysis: {filename}")
    
    with open(filename, 'w') as f:
        f.write("epoch,best_cost,best_x,best_y\n")
        for epoch, (cost, pos) in enumerate(zip(data['costs'], data['best_positions'])):
            f.write(f"{epoch},{cost:.6f},{pos[0]:.6f},{pos[1]:.6f}\n")

def main():
    """Main function - automatically create demonstrations"""
    print("🎬 SwarmOpt Automatic Visualization Demo")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('swarm_visualizations', exist_ok=True)
    os.chdir('swarm_visualizations')
    
    print("Creating demonstration visualizations...")
    
    # Demo configurations
    demos = [
        (sphere, 'Sphere', 'global', 'linear', 'basic'),
        (sphere, 'Sphere', 'global', 'adaptive', 'hybrid'),
        (rosenbrock, 'Rosenbrock', 'global', 'exponential', 'adaptive'),
        (ackley, 'Ackley', 'local', 'chaotic', 'exponential'),
    ]
    
    all_data = []
    
    for i, (func, name, algo, inertia, clamping) in enumerate(demos):
        print(f"\n{i+1}. Processing {name} with {algo} PSO...")
        
        # Capture paths
        data = capture_swarm_paths(
            func, name, algo, inertia, clamping,
            n_particles=8, epochs=25
        )
        all_data.append(data)
        
        # Create visualizations
        create_visualization_files(data, f"demo_{i+1}")
    
    # Create comparison summary
    print("\n📊 Creating comparison summary...")
    with open('comparison_summary.txt', 'w') as f:
        f.write("SwarmOpt Algorithm Comparison Summary\n")
        f.write("=" * 40 + "\n\n")
        
        for i, data in enumerate(all_data):
            f.write(f"Demo {i+1}: {data['function']} - {data['algorithm']} PSO\n")
            f.write(f"  Inertia: {data['inertia']}, Clamping: {data['clamping']}\n")
            f.write(f"  Final Cost: {data['final_cost']:.6f}\n")
            f.write(f"  Final Position: [{data['final_position'][0]:.3f}, {data['final_position'][1]:.3f}]\n")
            f.write(f"  Convergence: {data['costs'][-1]:.6f} (started at {data['costs'][0]:.6f})\n\n")
    
    print("\n" + "=" * 50)
    print("🎉 Demo visualizations created!")
    print("📁 Files created in 'swarm_visualizations' directory:")
    print("  - demo_*_swarm_paths_*.txt (text visualizations)")
    print("  - demo_*_swarm_data_*.json (JSON data for external tools)")
    print("  - demo_*_swarm_analysis_*.csv (CSV for analysis)")
    print("  - comparison_summary.txt (overview)")
    print("\n💡 These files can be used with:")
    print("  - External visualization tools (D3.js, Plotly, etc.)")
    print("  - Data analysis tools (pandas, R, etc.)")
    print("  - Custom visualization scripts")
    print("\n🎬 For video creation, install matplotlib and run:")
    print("  pip install matplotlib")
    print("  python simple_swarm_viz.py")

if __name__ == "__main__":
    main()
