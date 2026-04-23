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

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Or run specific test categories
python run_tests.py --unit                 # Fast unit tests only
python run_tests.py --show                 # Show all available tests
python tests/index.py                      # Interactive test runner
```

## Algorithms

### Single-Objective
* Global Best PSO - Kennedy & Eberhart 1995
* Local Best PSO - Kennedy & Eberhart 1995
* Unified PSO - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO - Liang & Suganthan 2005
* Simulated Annealing PSO - Mu, Cao, & Wang 2009
* Cooperative PSO (CPSO) - Van den Bergh & Engelbrecht 2004
* Horse Herd Optimization Algorithm (HHOA) - Ibrahim et al. 2025

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
* **Adaptive** - Adapts based on convergence progress
* **Chaotic-Random** - Combination of chaotic and random
* **Exponential Decreasing** - Exponential decay
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

## Sample Applications

* Neural network number of layers and weight optimization
* Routing in communication networks
* Satellite repair helper-swarm standoff positioning

## Citation

Siobhan K Cronin, SwarmOpt (2018), GitHub repository, https://github.com/SioKCronin/SwarmOpt
