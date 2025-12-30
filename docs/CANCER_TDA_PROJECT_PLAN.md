# TDA-Guided PSO for Cancer Growth Modeling and Control

## Project Overview

This project combines **Topological Data Analysis (TDA)** with **Particle Swarm Optimization (PSO)** to model and control 
cancer growth dynamics. The central innovation is using persistent homology to characterize tumor morphology and 
optimize treatment strategies that collapse invasive topological features.

## Core Concept

### Problem Statement
Current cancer models struggle to capture the complex spatial patterns and invasive behaviors that determine treatment outcomes. Topological features (loops, voids, connected components) provide robust descriptors of tumor morphology that are invariant to noise and deformations.

### Solution Approach
1. **Characterize** tumor morphology using persistent homology
2. **Optimize** simulator parameters to match observed topological signatures
3. **Search** for treatment policies that collapse invasive topology
4. **Control** tumor growth by targeting specific topological features

## Technical Architecture

### Phase 1: Parameter Fitting (Inverse Problem)
```
Observed Tumor → Persistence Diagram → PSO → Simulator Parameters
                       ↓
                 Wasserstein Distance
                       ↓
              Fitness Evaluation ← Simulated Tumor
```

**Objective Function:**
```python
minimize: Σ w_i * d_W(PD_obs^i, PD_sim^i)
where:
  - PD = persistence diagram
  - d_W = Wasserstein distance
  - i = homology dimension (H0, H1, H2)
  - w_i = importance weights (emphasize H1 for invasion)
```

### Phase 2: Control Policy Search (Forward Problem)
```
Initial Tumor → Treatment Policy → Simulator → Final Topology
                      ↓                             ↓
                    PSO ← Objective: Minimize H1 features
```

**Control Objective:**
```python
minimize: λ₁ * |H1| + λ₂ * persistence(H1) + λ₃ * |H2|
where:
  - |H1| = number of loops (invasive patterns)
  - persistence(H1) = lifetime of topological features
  - |H2| = number of cavities
```

## Implementation Plan

### Repository Structure
```
cancer-tda-pso/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .gitignore
├── cancer_tda/
│   ├── __init__.py
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── persistence.py        # Persistent homology
│   │   ├── diagrams.py           # Diagram utilities
│   │   ├── distance.py           # Wasserstein distance
│   │   └── filtrations.py        # Custom filtrations
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── reaction_diffusion.py # Basic RD model
│   │   ├── cellular_automata.py  # CA-based model
│   │   ├── agent_based.py        # ABM model
│   │   └── hybrid.py             # Hybrid approaches
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── tda_objective.py      # TDA-based objectives
│   │   ├── parameter_fitting.py  # Parameter estimation
│   │   ├── control_policy.py     # Treatment optimization
│   │   └── multiobjective.py     # Multi-objective (size + topology)
│   ├── control/
│   │   ├── __init__.py
│   │   ├── policies.py           # Treatment strategies
│   │   ├── scheduling.py         # Temporal scheduling
│   │   └── adaptive.py           # Adaptive control
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Image processing
│   │   ├── segmentation.py       # Tumor segmentation
│   │   └── time_series.py        # Temporal data handling
│   └── visualization/
│       ├── __init__.py
│       ├── persistence_plots.py  # PD visualization
│       ├── tumor_plots.py        # 3D tumor viz
│       └── optimization_viz.py   # PSO convergence
├── examples/
│   ├── 01_basic_persistence.py
│   ├── 02_parameter_fitting.py
│   ├── 03_control_policy.py
│   ├── 04_multiobjective.py
│   ├── 05_adaptive_control.py
│   └── 06_real_data.py
├── tests/
│   ├── test_topology.py
│   ├── test_simulator.py
│   ├── test_optimization.py
│   └── test_control.py
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_tda_basics.ipynb
│   ├── 03_parameter_fitting.ipynb
│   └── 04_treatment_optimization.ipynb
├── data/
│   ├── synthetic/
│   ├── observed/
│   └── results/
└── docs/
    ├── theory.md
    ├── api.md
    └── tutorials/
```

## 📚 Key Dependencies

### Required
```
numpy>=1.21.0
scipy>=1.7.0
swarmopt  # Your PSO library
```

### TDA Libraries (choose one or both)
```
# Option 1: Giotto-TDA (scikit-learn compatible)
giotto-tda>=0.6.0

# Option 2: GUDHI (comprehensive, C++ backend)
gudhi>=3.5.0

# Option 3: Ripser (fast persistence)
ripser>=0.6.0
persim>=0.3.0  # For Wasserstein distance
```

### Tumor Simulation
```
# For image-based models
scikit-image>=0.19.0
nibabel>=3.2.0  # Medical imaging

# For agent-based models
mesa>=1.0.0  # ABM framework
```

### Visualization
```
matplotlib>=3.5.0
plotly>=5.0.0
mayavi>=4.7.0  # 3D visualization
```

## 🎓 Mathematical Background

### Persistent Homology
Captures multi-scale topological features:
- **H0 (Betti-0)**: Connected components (tumor fragments)
- **H1 (Betti-1)**: Loops/cycles (invasive patterns, vascularization)
- **H2 (Betti-2)**: Voids/cavities (necrotic regions)

### Wasserstein Distance
Optimal transport distance between persistence diagrams:
```
d_W^p(D₁, D₂) = (inf_γ Σ ||x - γ(x)||_∞^p)^(1/p)
```
where γ is a matching between points in D₁ and D₂.

### PSO for Topological Optimization
**Advantages:**
- No gradient required (non-differentiable TDA objectives)
- Handles multi-modal objective landscapes
- Population-based exploration of parameter space
- Easily parallelizable

## 🔬 Research Applications

### 1. Parameter Estimation
**Goal:** Calibrate tumor growth models from imaging data

**Inputs:**
- Time series of tumor images
- Persistence diagrams at each timepoint

**Outputs:**
- Growth rate, diffusion coefficient, angiogenesis factors
- Model uncertainty quantification

### 2. Treatment Optimization
**Goal:** Find treatment schedules that minimize invasive topology

**Inputs:**
- Patient-specific tumor parameters
- Treatment constraints (dose limits, timing)

**Outputs:**
- Optimal treatment schedule
- Predicted topological evolution

### 3. Adaptive Control
**Goal:** Online adjustment of treatment based on topology changes

**Inputs:**
- Real-time imaging data
- Current treatment effects

**Outputs:**
- Treatment adjustments
- Topology-based early warnings

### 4. Drug Combination
**Goal:** Multi-drug strategies targeting different topological features

**Inputs:**
- Multiple drug mechanisms
- Topological objectives

**Outputs:**
- Optimal drug combinations
- Synergistic effects on topology

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up repository structure
- [ ] Implement basic persistence computation
- [ ] Create simple tumor simulator (reaction-diffusion)
- [ ] Integrate with SwarmOpt
- [ ] Basic parameter fitting example

### Phase 2: Core Functionality (Weeks 3-4)
- [ ] Wasserstein distance implementation
- [ ] Multi-scale persistence
- [ ] Treatment policy representation
- [ ] Control objective functions
- [ ] Visualization tools

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Multiobjective optimization (size + topology)
- [ ] Temporal dynamics tracking
- [ ] Adaptive control strategies
- [ ] Sensitivity analysis
- [ ] Uncertainty quantification

### Phase 4: Validation (Weeks 7-8)
- [ ] Synthetic data validation
- [ ] Benchmark against existing methods
- [ ] Case studies
- [ ] Documentation and tutorials
- [ ] Paper preparation

## 📊 Evaluation Metrics

### Model Fitting
- **Topological accuracy**: Wasserstein distance to observed diagrams
- **Morphological similarity**: Hausdorff distance, Dice coefficient
- **Temporal consistency**: Time-series correlation

### Treatment Optimization
- **Topology collapse**: Reduction in H1 features
- **Tumor size**: Volume reduction
- **Treatment efficiency**: Minimal dose for maximum effect
- **Side effects**: Constraints on healthy tissue

## 📖 Literature Foundation

### TDA for Cancer
- **Topological characterization of glioblastoma** (Adélie Garin et al.)
- **Persistent homology of tumor vasculature** (Paul Bendich et al.)
- **TDA for cancer cell morphology** (Lorin Crawford et al.)

### Tumor Growth Models
- **Reaction-diffusion models** (Murray, 2003)
- **Agent-based models** (An et al., 2017)
- **Hybrid multiscale models** (Cristini & Lowengrub, 2010)

### Optimization for Control
- **Swarm intelligence in medical optimization** (Kennedy & Eberhart, 2001)
- **Multi-objective cancer treatment** (Ledzewicz & Schättler, 2012)

## 🎯 Key Innovation

**Novel Contribution:**
Using **topological signatures** as the optimization objective, rather than traditional metrics (tumor size, cell count). This captures:
1. **Invasive patterns** through H1 features
2. **Morphological complexity** through persistence
3. **Treatment resistance** through topology persistence
4. **Multi-scale behavior** through filtration

## 🔗 Integration with SwarmOpt

### Example Usage
```python
from swarmopt import Swarm
from cancer_tda import PersistenceDiagram, TumorSimulator

# Set up TDA-guided optimization
pd_computer = PersistenceDiagram()
simulator = TumorSimulator()

def tda_objective(params):
    # Simulate tumor
    tumor = simulator.run(params)
    
    # Compute topology
    diagram = pd_computer.compute(tumor)
    
    # Compare to observed
    distance = pd_computer.wasserstein(diagram, observed_diagram)
    
    return distance

# Run PSO
swarm = Swarm(
    n_particles=30,
    dims=len(params),
    obj_func=tda_objective,
    epochs=100
)
swarm.optimize()
```

## 📝 Citation

If you use this work, please cite:
```bibtex
@software{cancer_tda_pso,
  title={TDA-Guided PSO for Cancer Growth Modeling and Control},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/cancer-tda-pso}
}
```

## 🤝 Collaboration Opportunities

This project would benefit from expertise in:
- **Oncology**: Clinical validation, data access
- **TDA**: Advanced persistence methods, stability theorems
- **Control Theory**: Optimal control, adaptive strategies
- **Imaging**: Medical image processing, segmentation
- **Computational Biology**: Multi-scale modeling, validation

## 📧 Contact

For questions, collaborations, or contributions, please open an issue or contact the maintainers.

---

**Built with [SwarmOpt](https://github.com/SioKCronin/swarmopt)**
