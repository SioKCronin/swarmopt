#!/usr/bin/env python3
"""
View GIFs - Display information about the created GIF animations
"""

import os

def show_gif_info():
    """Display information about the created GIFs"""
    print("🎬 SwarmOpt GIF Animations")
    print("=" * 50)
    
    gifs = [
        ("demo_1_sphere_global_linear.gif", "Sphere Function - Global PSO - Linear Inertia - Basic Clamping"),
        ("demo_2_sphere_global_adaptive.gif", "Sphere Function - Global PSO - Adaptive Inertia - Hybrid Clamping"),
        ("demo_3_rosenbrock_global_exponential.gif", "Rosenbrock Function - Global PSO - Exponential Inertia - Adaptive Clamping"),
        ("demo_4_ackley_local_chaotic.gif", "Ackley Function - Local PSO - Chaotic Inertia - Exponential Clamping")
    ]
    
    for i, (filename, description) in enumerate(gifs, 1):
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            print(f"\n{i}. {filename}")
            print(f"   📊 {description}")
            print(f"   📁 Size: {size_mb:.1f} MB")
            print(f"   🎯 Shows: Particle movement, convergence, cost evolution")
        else:
            print(f"\n{i}. {filename} - NOT FOUND")
    
    print("\n" + "=" * 50)
    print("🎉 What Each GIF Shows:")
    print("  🔵 Blue dots = Particles moving through the search space")
    print("  🔴 Red star = Best position found so far")
    print("  🟡 Gold X = Optimal solution (if known)")
    print("  📈 Right plot = Cost function evolution over time")
    print("\n💡 These GIFs demonstrate:")
    print("  - How particles explore the search space")
    print("  - How different algorithms behave")
    print("  - How inertia weights affect movement")
    print("  - How velocity clamping controls exploration")
    print("  - Real-time convergence to optimal solutions")

if __name__ == "__main__":
    show_gif_info()
