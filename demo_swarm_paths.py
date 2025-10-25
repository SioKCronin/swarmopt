#!/usr/bin/env python3
"""
Demo Swarm Paths - Demonstrate particle swarm optimization paths

This script shows how particles move during optimization without requiring matplotlib.
It creates text-based visualizations and data files that can be used for visualization.
"""

import numpy as np
import json
import os
from swarmopt import Swarm
from swarmopt.functions import sphere, rosenbrock, ackley

def capture_swarm_paths(obj_func, func_name, algo='global', inertia='linear', 
                       clamping='basic', n_particles=10, epochs=30):
    """
    Capture particle paths during optimization
    
    Returns:
    --------
    dict : Contains positions, best positions, and costs over time
    """
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

def create_text_visualization(data, filename=None):
    """Create a text-based visualization of particle paths"""
    if filename is None:
        filename = f"swarm_paths_{data['function']}_{data['algorithm']}.txt"
    
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
    
    print(f"✅ Text visualization saved to {filename}")

def create_json_data(data, filename=None):
    """Save optimization data as JSON for external visualization"""
    if filename is None:
        filename = f"swarm_data_{data['function']}_{data['algorithm']}.json"
    
    print(f"💾 Saving data as JSON: {filename}")
    
    # Convert numpy arrays to lists for JSON serialization
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
    
    print(f"✅ JSON data saved to {filename}")

def create_csv_data(data, filename=None):
    """Create CSV files for easy analysis"""
    if filename is None:
        filename = f"swarm_analysis_{data['function']}_{data['algorithm']}.csv"
    
    print(f"📊 Creating CSV analysis: {filename}")
    
    with open(filename, 'w') as f:
        f.write("epoch,best_cost,best_x,best_y\n")
        for epoch, (cost, pos) in enumerate(zip(data['costs'], data['best_positions'])):
            f.write(f"{epoch},{cost:.6f},{pos[0]:.6f},{pos[1]:.6f}\n")
    
    print(f"✅ CSV analysis saved to {filename}")

def create_demo_visualizations():
    """Create demonstration visualizations"""
    print("🎬 Creating SwarmOpt Demo Visualizations")
    print("=" * 50)
    
    # Demo configurations
    demos = [
        (sphere, 'Sphere', 'global', 'linear', 'basic'),
        (sphere, 'Sphere', 'global', 'adaptive', 'hybrid'),
        (rosenbrock, 'Rosenbrock', 'global', 'exponential', 'adaptive'),
        (ackley, 'Ackley', 'local', 'chaotic', 'exponential'),
    ]
    
    # Create output directory
    os.makedirs('swarm_visualizations', exist_ok=True)
    os.chdir('swarm_visualizations')
    
    for i, (func, name, algo, inertia, clamping) in enumerate(demos):
        print(f"\n{i+1}. Processing {name} with {algo} PSO...")
        
        # Capture paths
        data = capture_swarm_paths(
            func, name, algo, inertia, clamping,
            n_particles=8, epochs=25
        )
        
        # Create visualizations
        create_text_visualization(data, f"demo_{i+1}_{name.lower()}_paths.txt")
        create_json_data(data, f"demo_{i+1}_{name.lower()}_data.json")
        create_csv_data(data, f"demo_{i+1}_{name.lower()}_analysis.csv")
    
    print("\n" + "=" * 50)
    print("🎉 Demo visualizations created!")
    print("📁 Files created in 'swarm_visualizations' directory:")
    print("  - *_paths.txt (text visualizations)")
    print("  - *_data.json (JSON data for external tools)")
    print("  - *_analysis.csv (CSV for analysis)")
    print("\n💡 These files can be used with:")
    print("  - External visualization tools (D3.js, Plotly, etc.)")
    print("  - Data analysis tools (pandas, R, etc.)")
    print("  - Custom visualization scripts")

def create_algorithm_comparison():
    """Create comparison of different algorithms"""
    print("\n🎬 Creating Algorithm Comparison")
    print("=" * 50)
    
    algorithms = ['global', 'local', 'unified']
    algorithm_names = ['Global Best', 'Local Best', 'Unified']
    
    comparison_data = []
    
    for algo, name in zip(algorithms, algorithm_names):
        print(f"Processing {name} PSO...")
        data = capture_swarm_paths(
            sphere, 'Sphere', algo, 'linear', 'basic',
            n_particles=6, epochs=20
        )
        comparison_data.append(data)
        
        # Save individual data
        create_json_data(data, f"comparison_{algo}_data.json")
        create_csv_data(data, f"comparison_{algo}_analysis.csv")
    
    # Create comparison summary
    with open('algorithm_comparison_summary.txt', 'w') as f:
        f.write("Algorithm Comparison Summary\n")
        f.write("=" * 30 + "\n\n")
        
        for i, (data, name) in enumerate(zip(comparison_data, algorithm_names)):
            f.write(f"{name} PSO:\n")
            f.write(f"  Final Cost: {data['final_cost']:.6f}\n")
            f.write(f"  Final Position: [{data['final_position'][0]:.3f}, {data['final_position'][1]:.3f}]\n")
            f.write(f"  Convergence: {data['costs'][-1]:.6f} (started at {data['costs'][0]:.6f})\n\n")
    
    print("✅ Algorithm comparison created!")

def main():
    """Main function"""
    print("🎬 SwarmOpt Path Visualization Demo")
    print("=" * 50)
    print("This creates text-based visualizations and data files")
    print("that can be used for creating videos with external tools.")
    print()
    
    print("Choose visualization type:")
    print("1. Demo visualizations (4 different configurations)")
    print("2. Algorithm comparison (Global vs Local vs Unified)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        create_demo_visualizations()
    elif choice == '2':
        create_algorithm_comparison()
    elif choice == '3':
        create_demo_visualizations()
        create_algorithm_comparison()
    else:
        print("Invalid choice. Creating demo visualizations...")
        create_demo_visualizations()
    
    print("\n🎉 Visualization demo complete!")
    print("\n💡 Next steps for creating videos:")
    print("  1. Install matplotlib: pip install matplotlib")
    print("  2. Run: python simple_swarm_viz.py")
    print("  3. Or use the JSON data with external visualization tools")

if __name__ == "__main__":
    main()
