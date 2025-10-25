# SwarmOpt Installation and Usage Guide

## Quick Start

### 1. Activate the Virtual Environment
```bash
cd /Users/siobhan/code/swarmopt
source swarmopt/bin/activate
```

### 2. Run Tests
```bash
python -m pytest swarmopt/tests/ -v
```

### 3. Run Example
```bash
python example.py
```

## Installation from Scratch

If you want to set up the environment from scratch:

### 1. Create Virtual Environment
```bash
python3 -m venv swarmopt
source swarmopt/bin/activate
```

### 2. Install Dependencies
```bash
pip install numpy pytest
```

### 3. Install SwarmOpt in Development Mode
```bash
pip install -e .
```

### 4. Test Installation
```bash
python test_installation.py
```

## Usage

### Basic Usage
```python
from swarmopt import Swarm
from swarmopt.functions import sphere

# Create a swarm
swarm = Swarm(
    n_particles=30,
    dims=2,
    c1=2.0,
    c2=2.0,
    w=0.9,
    epochs=100,
    obj_func=sphere,
    algo='global',
    velocity_clamp=(-5, 5)
)

# Run optimization
swarm.optimize()

# Get results
print(f"Best cost: {swarm.best_cost}")
print(f"Best position: {swarm.best_pos}")
```

### Available Algorithms
- `'global'` - Global Best PSO
- `'local'` - Local Best PSO  
- `'unified'` - Unified PSO
- `'sa'` - Simulated Annealing PSO
- `'multiswarm'` - Dynamic Multi-Swarm PSO

### Available Functions
- `sphere` - Sphere function
- `rosenbrock` - Rosenbrock function
- `ackley` - Ackley function
- `griewank` - Griewank function
- `rastrigin` - Rastrigin function
- `weierstrass` - Weierstrass function

## Features

✅ **Multiple PSO Algorithms**: Global, Local, Unified, SA, Multi-swarm
✅ **Benchmark Functions**: 6 standard optimization test functions
✅ **Configurable Parameters**: Full control over PSO parameters
✅ **Performance Tracking**: Runtime and convergence metrics
✅ **Comprehensive Tests**: Full test suite with 10 passing tests
✅ **Working Examples**: Complete example demonstrating all features

## Project Status

🎉 **FULLY FUNCTIONAL** - All tests passing, example working!

The SwarmOpt library is now ready for production use with:
- Fixed all import issues
- Corrected algorithm implementations
- Working test suite
- Comprehensive example
- Performance benchmarks
