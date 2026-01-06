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

# Or run specific test categories
python run_tests.py --unit                 # Fast unit tests only
python run_tests.py --show                 # Show all available tests
python tests/index.py                      # Interactive test runner
```

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

### Horse Herd Optimization Algorithm (HHOA)

```python
from swarmopt import Swarm
from swarmopt.functions import sphere

# HHOA mimics horse herd behavior with three phases:
# 1. Grazing (exploration)
# 2. Leadership (exploitation)
# 3. Following (social learning)

swarm = Swarm(
    n_particles=30,
    dims=2,
    c1=2.0, c2=2.0, w=0.9,  # Parameters ignored for HHOA but kept for compatibility
    epochs=100,
    obj_func=sphere,
    algo='hhoa',  # Enable Horse Herd Optimization Algorithm
    velocity_clamp=(-5, 5)
)

swarm.optimize()
print(f"Best cost: {swarm.best_cost}")
```

**Reference:** [A high-speed MPPT based horse herd optimization algorithm](https://www.nature.com/articles/s41598-025-85481-6)

### Multiobjective Optimization

```python
from swarmopt import Swarm
from swarmopt.utils.simple_multiobjective import zdt1

# Define multiobjective function (returns array of objectives)
def multiobjective_function(x):
    return zdt1(x)  # Returns [f1, f2]

# Create multiobjective swarm
swarm = Swarm(
    n_particles=20,
    dims=5,
    c1=2.0, c2=2.0, w=0.9,
    epochs=50,
    obj_func=multiobjective_function,
    multiobjective=True,  # Enable multiobjective optimization
    archive_size=50       # Size of Pareto front archive
)

# Run optimization
swarm.optimize()

# Access results
pareto_front = swarm.mo_optimizer.archive
print(f"Found {len(pareto_front)} Pareto-optimal solutions")
```

### Horse Herd Optimization Algorithm (HHOA)

```python
from swarmopt import Swarm
from swarmopt.functions import sphere

# HHOA mimics horse herd behavior with three phases:
# 1. Grazing (exploration)
# 2. Leadership (exploitation)
# 3. Following (social learning)

swarm = Swarm(
    n_particles=30,
    dims=2,
    c1=2.0, c2=2.0, w=0.9,  # Parameters ignored for HHOA but kept for compatibility
    epochs=100,
    obj_func=sphere,
    algo='hhoa',  # Enable Horse Herd Optimization Algorithm
    velocity_clamp=(-5, 5)
)

swarm.optimize()
print(f"Best cost: {swarm.best_cost}")
```

**Reference:** [A high-speed MPPT based horse herd optimization algorithm](https://www.nature.com/articles/s41598-025-85481-6)

### Respect Boundary (Safety-Critical Applications)

```python
import numpy as np
from swarmopt import Swarm

# Example: Respect boundary for safety-critical applications
target = np.array([10.0, 10.0])

def distance_objective(position):
    return np.linalg.norm(position - target)

# Create swarm with automatic respect boundary
swarm = Swarm(
    n_particles=30,
    dims=2,
    c1=2.0, c2=2.0, w=0.9,
    epochs=50,
    obj_func=distance_objective,
    target_position=target  # Respect boundary automatically enforced!
)
# ⚠️ Particles will maintain safe distance from target

swarm.optimize()
distance = np.linalg.norm(swarm.best_pos - target)
print(f"Optimal distance from target: {distance:.2f}")
```

## Algorithms

### Single-Objective
* Global Best PSO - Kennedy & Eberhart 1995
* Local Best PSO - Kennedy & Eberhart 1995
* Unified PSO - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO - Liang & Suganthan 2005
* Simulated Annealing PSO - Mu, Cao, & Wang 2009
* Cooperative PSO (CPSO) - Van den Bergh & Engelbrecht 2004
* **Horse Herd Optimization Algorithm (HHOA)** - Ibrahim et al. 2025 ⭐

### Multiobjective
* Multiobjective PSO - Handles multiple conflicting objectives simultaneously 

## Benchmark Functions

### Single Objective
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

### Multiobjective
* **ZDT1, ZDT2, ZDT3** - Zitzler-Deb-Thiele test functions
* **DTLZ1, DTLZ2** - Deb-Thiele-Laumanns-Zitzler test functions

## Implemented Features

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

## On Deck

* Cooperative Approach to PSO (CPSO)(multiple collaborating swarms)
* Proactive Particles in Swarm Optimization (PPSO) (self-tuning swarms)
* Variation operator variations
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
* Satelite positioning
* Routing in communication networks
* Anomaly detection

## Citation

Siobhan K Cronin, SwarmOpt (2018), GitHub repository, https://github.com/SioKCronin/SwarmOpt
