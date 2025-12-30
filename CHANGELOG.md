# Changelog

All notable changes to SwarmOpt will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-14

### 🎯 Major Features Added

#### Safety & Control
- **Respect Boundary**: Safety-critical standoff distance optimization
  - Automatic enforcement when target position specified
  - Default: 10% of search space diagonal
  - Applications: Satellites, robots, collision avoidance
  - Cannot be disabled for safety-critical applications

#### Multiobjective Optimization
- **Multiobjective PSO**: Handle multiple conflicting objectives
  - NSGA-II inspired algorithm
  - SPEA2 inspired algorithm
  - Pareto dominance and ranking
  - Hypervolume indicator
  - External archive management
  - ZDT and DTLZ benchmark functions

#### Advanced PSO Algorithms
- **Cooperative PSO (CPSO)**: Multiple collaborating swarms
  - Dimension-based swarm assignment
  - Communication strategies: best, random, tournament
  - Scalable for high-dimensional problems

- **Proactive PSO (PPSO)**: Knowledge gain guided exploration
  - Mixed swarms (proactive + reactive particles)
  - Gaussian Process-inspired exploration
  - Sample density awareness
  - Adaptive exploration weights

#### Diversity & Variation
- **Diversity Monitoring**: Prevent premature convergence
  - 9 diversity metrics (Euclidean, Manhattan, Variance, etc.)
  - Automatic intervention triggers
  - Diversity-based particle restart

- **Variation Operators**: Escape local optima
  - 14 variation strategies
  - Gaussian, Uniform, Polynomial variations
  - Escape and diversity-preserving variations
  - Adaptive variation strength
  - Stagnation detection

#### Inertia & Velocity Control
- **8 Inertia Weight Variations**:
  - Constant, Linear Decreasing, Chaotic
  - Random, Adaptive, Chaotic-Random
  - Exponential Decreasing, Sigmoid Decreasing

- **11 Velocity Clamping Strategies**:
  - No Clamping, Basic, Adaptive
  - Exponential, Sigmoid, Random
  - Chaotic, Dimension-wise, Soft
  - Hybrid, Convergence-based

#### Application-Specific
- **Airfoil Optimization Suite**:
  - CST (Class-Shape Transformation) parameterization
  - Hicks-Henne bump functions
  - Bézier curve representation
  - NACA-k modifications
  - Multiple objective functions
  - Constraint handling

### 📊 Visualizations
- Interactive HTML animation viewer
- Respect boundary visualizations
- Satellite positioning examples
- Airfoil optimization viewer
- GIF generation for swarm paths
- Comparison visualizations

### 📚 Documentation
- Comprehensive README with examples
- Respect boundary guide (`docs/RESPECT_BOUNDARY_README.md`)
- Cancer TDA project plan (`docs/CANCER_TDA_PROJECT_PLAN.md`)
- Publishing guide (`PUBLISHING.md`)
- Test suite documentation
- Interactive visualization gallery

### 🧪 Testing
- 10+ comprehensive test scripts
- Test index for easy navigation (`tests_scripts/index.py`)
- Quick test runner (`run_tests.py`)
- Examples for all major features
- Installation verification scripts

### 🗂️ Project Organization
- Created `docs/` directory for documentation
- Organized `tests_scripts/` for all test files
- Centralized `swarm_visualizations/` for all visualizations
- Clean repository structure
- Improved navigation and discoverability

### 🔧 Improvements
- Better error handling
- Performance optimizations
- Enhanced code documentation
- Modular utility structure
- Backward compatibility maintained

### 🐛 Bug Fixes
- Fixed particle update logic in various algorithms
- Corrected local best position tracking
- Fixed SA algorithm implementation
- Improved multiswarm regrouping
- Various performance and stability fixes

### 📦 Dependencies
- Core: `numpy>=1.19.0`
- Optional: `matplotlib>=3.3.0` (for visualizations)
- Optional: `scipy>=1.5.0` (for multiobjective)
- Optional: `giotto-tda>=0.5.0` (for TDA applications)

## [0.1.0] - 2018-XX-XX

### Added
- Initial release
- Basic PSO algorithms:
  - Global Best PSO
  - Local Best PSO
  - Unified PSO
  - Dynamic Multi-Swarm PSO
  - Simulated Annealing PSO
- Benchmark functions:
  - Sphere, Rosenbrock, Ackley
  - Griewank, Rastrigin, Weierstrass
- Basic particle and swarm classes
- Initial documentation

---

## Version Numbering

- **Major version** (X.0.0): Breaking changes
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes only

## Links

- **PyPI**: https://pypi.org/project/swarmopt/
- **GitHub**: https://github.com/siokcronin/swarmopt
- **Documentation**: https://github.com/siokcronin/swarmopt/tree/main/docs
- **Issues**: https://github.com/siokcronin/swarmopt/issues

