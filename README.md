![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/media/swarmopt_lateral.png)

# SwarmOpt

SwarmOpt is a library of swarm optimization algorithms implemented in Python. 

Swarm intelligence leverages population-based search solutions to balance exploration and exploitation with respect 
to specified cost functions. The PSO lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, 
and the intervening years have seen many variations spring from their central idea. 

## Installation

To install SwarmOpt, run this command in your terminal:

```shell
$ pip install swarmopt
```

## Quick Start

```python
from swarmopt import Swarm
from swarmopt.functions import sphere

# Basic usage
swarm = Swarm(
    n_particles=30,
    dims=2,
    c1=2.0,
    c2=2.0,
    w=0.9,
    epochs=100,
    obj_func=sphere,
    algo='global'
)

swarm.optimize()
print(f"Best cost: {swarm.best_cost}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Or run specific tests
python tests_scripts/index.py list          # Show all available tests
python tests_scripts/index.py all           # Run all tests
python tests_scripts/index.py test_name    # Run specific test
```

## 🎬 Visualizations

See SwarmOpt in action with our animated demonstrations:

**[🎬 View Particle Swarm Animations](https://htmlpreview.github.io/?https://github.com/SioKCronin/swarmopt/blob/update2/swarm_visualizations/view_animations.html)**

The visualizations show:
- 🔵 **Blue dots** = Particles moving through the search space
- 🔴 **Red star** = Best position found so far  
- 🟡 **Gold X** = Optimal solution (if known)
- 📈 **Right plot** = Cost function evolution over time

### Demo Animations
- **Demo 1**: Classic PSO approach
- **Demo 2**: Advanced adaptive approach  
- **Demo 3**: Challenging Rosenbrock function
- **Demo 4**: Complex Ackley function with chaotic inertia

> 💡 **Tip**: The animations are also available as GIF files in the [`swarm_visualizations/`](https://github.com/SioKCronin/swarmopt/tree/update2/swarm_visualizations) directory for download and offline viewing.

## Advanced Usage

### Inertia Weight Variations
```python
# Use adaptive inertia weight
swarm = Swarm(
    n_particles=30, dims=2, c1=2.0, c2=2.0, w=0.9, epochs=100,
    obj_func=sphere, algo='global',
    inertia_func='adaptive',  # Try: linear, exponential, chaotic, random, adaptive
    w_start=0.9, w_end=0.4
)
```

### Velocity Clamping Variations
```python
# Use hybrid velocity clamping
swarm = Swarm(
    n_particles=30, dims=2, c1=2.0, c2=2.0, w=0.9, epochs=100,
    obj_func=sphere, algo='global',
    velocity_clamp_func='hybrid'  # Try: basic, adaptive, exponential, chaotic, soft
)
```

### Combined Advanced Features
```python
# Combine inertia and velocity clamping
swarm = Swarm(
    n_particles=50, dims=3, c1=2.0, c2=2.0, w=0.9, epochs=200,
    obj_func=sphere, algo='global',
    inertia_func='exponential', w_start=0.9, w_end=0.4,
    velocity_clamp_func='adaptive'
)
```

### Cooperative PSO (CPSO)
```python
# Multiple collaborating swarms
swarm = Swarm(
    n_particles=20, dims=6, c1=2.0, c2=2.0, w=0.9, epochs=100,
    obj_func=sphere, algo='cpso',
    n_swarms=3,  # 3 collaborating swarms
    communication_strategy='best'  # Try: best, random, tournament
)
```

## Algorithms

### Single-Objective
* Global Best PSO - Kennedy & Eberhart 1995
* Local Best PSO - Kennedy & Eberhart 1995
* Unified PSO - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO - Liang & Suganthan 2005
* Simulated Annealing PSO - Mu, Cao, & Wang 2009
* **Cooperative PSO (CPSO)** - Van den Bergh & Engelbrecht 2004 ⭐

## Benchmark Functions

Single objective test functions:
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

## ✅ Implemented Features

### Inertia Weight Variations
* **Constant** - Traditional fixed inertia weight
* **Linear Decreasing** - Classic linear decay (default)
* **Chaotic** - Chaotic inertia using logistic map
* **Random** - Random inertia between 0.5-1.0
* **Adaptive** - Adapts based on convergence progress ⭐
* **Chaotic-Random** - Combination of chaotic and random
* **Exponential Decreasing** - Exponential decay ⭐
* **Sigmoid Decreasing** - Sigmoid decay curve

### Velocity Clamping Variations
* **No Clamping** - Particles can move freely
* **Basic Clamping** - Standard velocity bounds
* **Adaptive Clamping** - Decreases over time
* **Exponential Clamping** - Exponential decay
* **Sigmoid Clamping** - Sigmoid decay
* **Random Clamping** - Random bounds
* **Chaotic Clamping** - Chaotic bounds using logistic map
* **Soft Clamping** - Soft bounds using tanh
* **Hybrid Clamping** - Adaptive + exponential
* **Convergence-Based** - Based on optimization progress

## 🚧 On Deck

* Cooperative Approach to PSO (CPSO)(multiple collaborating swarms)
* Proactive Particles in Swarm Optimization (PPSO) (self-tuning swarms)
* Mutation operator variations
* Multiobjective variations
* Benchmark on something canonical like MNIST

## Performance

### Inertia Weight Performance
- **Adaptive Inertia**: Best performer on most functions
- **Exponential Decreasing**: Excellent convergence
- **Linear Decreasing**: Reliable baseline
- **Chaotic Inertia**: Good for exploration

### Velocity Clamping Performance
- **Hybrid Clamping**: Best overall performance
- **Exponential Clamping**: Excellent convergence
- **Adaptive Clamping**: Good balance of exploration/exploitation
- **Soft Clamping**: Smooth convergence

### Combined Performance
- **Exponential Inertia + Hybrid Clamping**: Optimal for most problems
- **Adaptive Inertia + Adaptive Clamping**: Best for complex landscapes
- **Linear Inertia + Basic Clamping**: Reliable baseline

## Applications

* Neural network number of layers and weight optimization
* Grid scheduling (load balancing)
* Routing in communication networks
* Anomaly detection

## Citation

Siobhán K Cronin, SwarmOpt (2018), GitHub repository, https://github.com/SioKCronin/SwarmOpt
