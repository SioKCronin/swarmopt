# SwarmOpt Visualizations

This directory contains all visualization tools and outputs for SwarmOpt particle swarm optimization.

## 🎬 Animated GIFs

### Demo Animations
- `demo_1_sphere_global_linear.gif` - Classic PSO approach
- `demo_2_sphere_global_adaptive.gif` - Advanced adaptive approach  
- `demo_3_rosenbrock_global_exponential.gif` - Challenging Rosenbrock function
- `demo_4_ackley_local_chaotic.gif` - Complex Ackley function with chaotic inertia

### What the GIFs Show
- 🔵 **Blue dots** = Particles moving through the search space
- 🔴 **Red star** = Best position found so far
- 🟡 **Gold X** = Optimal solution (if known)
- 📈 **Right plot** = Cost function evolution over time

## 🛠️ Visualization Scripts

### Video Creation (requires matplotlib)
- `simple_swarm_viz.py` - Interactive video creation
- `visualize_swarm_paths.py` - Advanced visualization suite
- `create_swarm_gif.py` - GIF animation creator

### Data Capture (works without matplotlib)
- `auto_demo_viz.py` - Automatic demonstration generation
- `demo_swarm_paths.py` - Text-based visualizations

### Viewing Tools
- `view_gifs.py` - Display GIF information
- `view_animations.html` - Browser-based viewer

## 📊 Data Files

### Demo Data
- `demo_*_swarm_paths_*.txt` - Text visualizations
- `demo_*_swarm_data_*.json` - JSON data for external tools
- `demo_*_swarm_analysis_*.csv` - CSV analysis files
- `comparison_summary.txt` - Algorithm comparison overview

## 🚀 Usage

### Create GIFs
```bash
cd swarm_visualizations
python create_swarm_gif.py
```

### Create Videos (requires matplotlib)
```bash
cd swarm_visualizations
python simple_swarm_viz.py
```

### View Animations
```bash
cd swarm_visualizations
python view_gifs.py
open view_animations.html
```

## 💡 Perfect For

- Educational demonstrations
- Algorithm comparisons
- Understanding PSO behavior
- Sharing with colleagues and students
- GitHub README demonstrations
- Research presentations

## 🎯 Features

- **Particle trajectory animation** over time
- **Best position evolution** tracking
- **Convergence pattern** visualization
- **Side-by-side algorithm** comparison
- **Real-time cost function** plotting
- **Multiple output formats** (GIF, MP4, JSON, CSV, HTML)
