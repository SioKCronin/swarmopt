# Test Scripts Directory

This directory contains all test scripts, examples, and demonstration code for the SwarmOpt library.

## 📁 File Organization

### 🧪 **Core Test Scripts**
- `test_installation.py` - Verify SwarmOpt installation and basic functionality
- `test_inertia_variations.py` - Test and compare different inertia weight strategies
- `test_velocity_clamping.py` - Test all velocity clamping variations
- `test_cpso.py` - Comprehensive Cooperative PSO testing
- `test_mutation_operators.py` - Test mutation operators for local optima escape
- `test_diversity_system.py` - Test diversity measurement and intervention system

### 📚 **Examples and Demos**
- `example.py` - Comprehensive example showcasing all SwarmOpt features
- `INSTALL.md` - Installation and usage guide
- `PR_DESCRIPTION.md` - Detailed Pull Request description

## 🚀 **Quick Start**

### Run All Tests
```bash
# Test basic installation
python test_installation.py

# Test inertia weight variations
python test_inertia_variations.py

# Test velocity clamping strategies
python test_velocity_clamping.py

# Test Cooperative PSO
python test_cpso.py

# Test mutation operators
python test_mutation_operators.py

# Test diversity measurement system
python test_diversity_system.py

# Run comprehensive example
python example.py
```

### Run Specific Tests
```bash
# Test specific features
python test_inertia_variations.py --function sphere --epochs 50
python test_velocity_clamping.py --strategy adaptive
python test_cpso.py --communication best
python test_mutation_operators.py --strategy hybrid
```

## 📊 **Test Categories**

### **1. Installation & Basic Functionality**
- `test_installation.py`
  - Import testing
  - Basic PSO functionality
  - Algorithm compatibility
  - Function compatibility

### **2. Inertia Weight Variations**
- `test_inertia_variations.py`
  - Constant, Linear, Chaotic, Random, Adaptive
  - Chaotic-Random, Exponential, Sigmoid
  - Performance comparison across functions
  - Visualization (if matplotlib available)

### **3. Velocity Clamping Strategies**
- `test_velocity_clamping.py`
  - 11 different clamping strategies
  - Performance analysis
  - Convergence behavior
  - Multi-function testing

### **4. Cooperative PSO (CPSO)**
- `test_cpso.py`
  - Basic CPSO functionality
  - Communication strategies (best, random, tournament)
  - Comparison with standard PSO
  - Scalability testing

### **5. Mutation Operators**
- `test_mutation_operators.py`
  - 10+ mutation strategies
  - Local optima escape testing
  - Stagnation detection
  - Adaptive response testing

### **6. Diversity Measurement System**
- `test_diversity_system.py`
  - Multiple diversity metrics
  - Real-time monitoring
  - Automatic interventions
  - Performance comparisons

## 🎯 **Expected Outputs**

### **Test Results**
- ✅/❌ Pass/fail indicators
- Performance metrics (cost, runtime)
- Convergence analysis
- Diversity statistics

### **Visualizations** (if matplotlib available)
- Convergence curves
- Swarm path animations
- Diversity trend plots
- Performance comparisons

### **Data Files**
- CSV performance data
- JSON swarm statistics
- Text-based analysis reports

## 🔧 **Configuration**

### **Test Parameters**
Most test scripts accept command-line arguments:
```bash
python test_script.py --particles 20 --epochs 50 --dims 3
```

### **Common Options**
- `--particles N` - Number of particles
- `--epochs N` - Number of iterations
- `--dims N` - Problem dimensions
- `--function NAME` - Objective function
- `--algorithm NAME` - PSO algorithm
- `--visualize` - Enable visualizations

## 📈 **Performance Benchmarks**

### **Standard Test Functions**
- **Sphere** - Simple convex function
- **Rosenbrock** - Valley function
- **Ackley** - Many local optima
- **Griewank** - Product terms
- **Rastrigin** - Highly multimodal
- **Weierstrass** - Continuous but non-differentiable

### **Expected Performance**
- **Sphere**: Should converge to near-zero
- **Rosenbrock**: Should find global minimum
- **Ackley**: Should escape local optima
- **Rastrigin**: Should handle multimodality
- **Griewank**: Should handle product terms
- **Weierstrass**: Should handle non-differentiability

## 🐛 **Troubleshooting**

### **Common Issues**
1. **Import errors** - Ensure SwarmOpt is installed
2. **Matplotlib errors** - Install with `pip install matplotlib`
3. **Memory issues** - Reduce particles/epochs
4. **Convergence issues** - Try different parameters

### **Debug Mode**
```bash
# Enable verbose output
python test_script.py --verbose

# Enable debug logging
python test_script.py --debug
```

## 📝 **Adding New Tests**

### **Test Template**
```python
#!/usr/bin/env python3
"""
Test [Feature Name]

Description of what this test does.
"""

import numpy as np
from swarmopt import Swarm
from swarmopt.functions import sphere

def test_feature():
    """Test the feature"""
    print("🧪 Testing [Feature Name]")
    
    # Test implementation
    swarm = Swarm(...)
    swarm.optimize()
    
    print(f"✅ Results: {swarm.best_cost:.6f}")
    return True

if __name__ == "__main__":
    test_feature()
```

### **Best Practices**
- Use descriptive test names
- Include performance metrics
- Add error handling
- Provide clear output
- Include usage examples

## 🎉 **Contributing**

When adding new test scripts:
1. Follow the naming convention `test_*.py`
2. Include comprehensive docstrings
3. Add to this README
4. Test with multiple functions
5. Include performance benchmarks

---

**Happy Testing!** 🚀✨
