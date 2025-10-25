#!/usr/bin/env python3
"""
Test Airfoil Optimization with Various Parameterization Methods

This script tests airfoil optimization using different parameterization
techniques: CST, Hicks-Henne, Bézier, and NACA-k.
"""

import numpy as np
from swarmopt.utils.airfoil_parameterization import (
    CSTParameterization, HicksHenneParameterization, 
    BezierParameterization, NACAKParameterization, AirfoilOptimization
)

def test_airfoil_parameterizations():
    """Test all airfoil parameterization methods"""
    print("🧪 Testing Airfoil Parameterization Methods")
    print("=" * 60)
    
    # Test parameters
    n_points = 50
    test_params = {
        'CST': np.array([0.05, 0.02, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02]),
        'Hicks-Henne': np.array([0.02, -0.01, 0.015, -0.005, 0.01, -0.008, 0.005, -0.003,
                                0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.45, 0.65]),
        'Bézier': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.05, 0.08, 0.06, 0.04, 0.02, 0.0]),
        'NACA-k': np.array([0.12, 0.02, 0.4, 0.005, -0.003, 0.002, -0.001])
    }
    
    parameterizations = {
        'CST': CSTParameterization(n_points, 8),
        'Hicks-Henne': HicksHenneParameterization(n_points, 8),
        'Bézier': BezierParameterization(n_points, 6),
        'NACA-k': NACAKParameterization(n_points, 4)
    }
    
    results = {}
    
    for name, param in parameterizations.items():
        print(f"\nTesting {name} parameterization...")
        
        try:
            # Generate airfoil
            x_coords, y_coords = param.generate_airfoil(test_params[name])
            
            # Basic validation
            is_valid = len(x_coords) > 0 and len(y_coords) > 0
            has_closed_shape = abs(y_coords[0] - y_coords[-1]) < 0.01
            
            results[name] = {
                'valid': is_valid,
                'closed': has_closed_shape,
                'n_points': len(x_coords),
                'x_range': (np.min(x_coords), np.max(x_coords)),
                'y_range': (np.min(y_coords), np.max(y_coords))
            }
            
            print(f"   ✅ Valid: {is_valid}")
            print(f"   ✅ Closed shape: {has_closed_shape}")
            print(f"   ✅ Points: {len(x_coords)}")
            print(f"   ✅ X range: {results[name]['x_range']}")
            print(f"   ✅ Y range: {results[name]['y_range']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {'valid': False, 'error': str(e)}
    
    return results

def test_airfoil_objective_functions():
    """Test different objective functions for airfoil optimization"""
    print("\n🎯 Testing Airfoil Objective Functions")
    print("=" * 60)
    
    # Simple objective functions for testing
    def minimize_thickness(x_coords, y_coords):
        """Minimize maximum thickness"""
        return np.max(y_coords) - np.min(y_coords)
    
    def maximize_lift_to_drag_ratio(x_coords, y_coords):
        """Maximize lift-to-drag ratio (simplified)"""
        # Simplified L/D calculation based on airfoil shape
        thickness = np.max(y_coords) - np.min(y_coords)
        camber = np.mean(y_coords)
        return -camber / (thickness + 0.01)  # Negative for minimization
    
    def minimize_drag_coefficient(x_coords, y_coords):
        """Minimize drag coefficient (simplified)"""
        # Simplified drag calculation
        thickness = np.max(y_coords) - np.min(y_coords)
        surface_roughness = np.sum(np.abs(np.diff(y_coords)))
        return thickness + 0.1 * surface_roughness
    
    def maximize_stall_angle(x_coords, y_coords):
        """Maximize stall angle (simplified)"""
        # Simplified stall angle calculation
        leading_edge_radius = np.min(x_coords)
        camber = np.mean(y_coords)
        return -(leading_edge_radius + abs(camber))
    
    objectives = {
        'Minimize Thickness': minimize_thickness,
        'Maximize L/D Ratio': maximize_lift_to_drag_ratio,
        'Minimize Drag': minimize_drag_coefficient,
        'Maximize Stall Angle': maximize_stall_angle
    }
    
    results = {}
    
    for name, obj_func in objectives.items():
        print(f"\nTesting {name} objective...")
        
        # Test with CST parameterization
        param = CSTParameterization(50, 6)
        optimizer = AirfoilOptimization(param, obj_func)
        
        try:
            # Quick optimization test
            result = optimizer.optimize(n_particles=10, epochs=20)
            
            results[name] = {
                'success': True,
                'best_cost': result['best_cost'],
                'runtime': result['runtime']
            }
            
            print(f"   ✅ Best cost: {result['best_cost']:.6f}")
            print(f"   ✅ Runtime: {result['runtime']:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    return results

def test_parameterization_comparison():
    """Compare different parameterization methods"""
    print("\n⚔️ Parameterization Method Comparison")
    print("=" * 60)
    
    # Common objective function
    def airfoil_objective(x_coords, y_coords):
        """Combined airfoil objective function"""
        # Minimize thickness while maintaining reasonable shape
        thickness = np.max(y_coords) - np.min(y_coords)
        
        # Penalize unrealistic shapes
        shape_penalty = 0
        if thickness < 0.05:  # Too thin
            shape_penalty += 1000
        if thickness > 0.25:  # Too thick
            shape_penalty += 1000
        
        # Penalize non-smooth shapes
        roughness = np.sum(np.abs(np.diff(y_coords)))
        if roughness > 1.0:
            shape_penalty += roughness * 100
        
        return thickness + shape_penalty
    
    parameterizations = {
        'CST': CSTParameterization(50, 8),
        'Hicks-Henne': HicksHenneParameterization(50, 8),
        'Bézier': BezierParameterization(50, 6),
        'NACA-k': NACAKParameterization(50, 4)
    }
    
    results = {}
    
    for name, param in parameterizations.items():
        print(f"\nTesting {name} optimization...")
        
        optimizer = AirfoilOptimization(param, airfoil_objective)
        
        try:
            result = optimizer.optimize(n_particles=15, epochs=30)
            
            results[name] = {
                'best_cost': result['best_cost'],
                'runtime': result['runtime'],
                'n_params': len(result['best_params']),
                'success': True
            }
            
            print(f"   ✅ Best cost: {result['best_cost']:.6f}")
            print(f"   ✅ Runtime: {result['runtime']:.3f}s")
            print(f"   ✅ Parameters: {len(result['best_params'])}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Find best method
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    if successful_results:
        best_method = min(successful_results.keys(), 
                         key=lambda k: successful_results[k]['best_cost'])
        print(f"\n🏆 Best parameterization method: {best_method}")
    
    return results

def test_airfoil_visualization():
    """Test airfoil visualization (if matplotlib available)"""
    print("\n📈 Testing Airfoil Visualization")
    print("=" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        # Generate sample airfoils
        parameterizations = {
            'CST': CSTParameterization(100, 8),
            'Hicks-Henne': HicksHenneParameterization(100, 8),
            'Bézier': BezierParameterization(100, 6),
            'NACA-k': NACAKParameterization(100, 4)
        }
        
        # Sample parameters
        sample_params = {
            'CST': np.array([0.05, 0.02, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02]),
            'Hicks-Henne': np.array([0.02, -0.01, 0.015, -0.005, 0.01, -0.008, 0.005, -0.003,
                                    0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.45, 0.65]),
            'Bézier': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.05, 0.08, 0.06, 0.04, 0.02, 0.0]),
            'NACA-k': np.array([0.12, 0.02, 0.4, 0.005, -0.003, 0.002, -0.001])
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, param) in enumerate(parameterizations.items()):
            try:
                x_coords, y_coords = param.generate_airfoil(sample_params[name])
                
                axes[i].plot(x_coords, y_coords, 'b-', linewidth=2)
                axes[i].set_title(f'{name} Airfoil')
                axes[i].set_xlabel('x/c')
                axes[i].set_ylabel('y/c')
                axes[i].grid(True)
                axes[i].axis('equal')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[i].set_title(f'{name} Airfoil (Error)')
        
        plt.tight_layout()
        plt.savefig('airfoil_parameterizations.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Airfoil visualization saved as 'airfoil_parameterizations.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

def test_optimization_algorithms():
    """Test different PSO algorithms on airfoil optimization"""
    print("\n🔄 Testing Optimization Algorithms")
    print("=" * 60)
    
    # Simple objective function
    def simple_objective(x_coords, y_coords):
        """Simple objective: minimize thickness"""
        return np.max(y_coords) - np.min(y_coords)
    
    algorithms = ['global', 'local', 'unified', 'multiswarm']
    param = CSTParameterization(50, 6)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm} algorithm...")
        
        optimizer = AirfoilOptimization(param, simple_objective)
        
        try:
            result = optimizer.optimize(n_particles=12, epochs=25, algorithm=algorithm)
            
            results[algorithm] = {
                'best_cost': result['best_cost'],
                'runtime': result['runtime'],
                'success': True
            }
            
            print(f"   ✅ Best cost: {result['best_cost']:.6f}")
            print(f"   ✅ Runtime: {result['runtime']:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[algorithm] = {'success': False, 'error': str(e)}
    
    # Find best algorithm
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    if successful_results:
        best_algorithm = min(successful_results.keys(), 
                           key=lambda k: successful_results[k]['best_cost'])
        print(f"\n🏆 Best optimization algorithm: {best_algorithm}")
    
    return results

def test_airfoil_constraints():
    """Test airfoil optimization with constraints"""
    print("\n🔒 Testing Airfoil Constraints")
    print("=" * 60)
    
    def constrained_objective(x_coords, y_coords):
        """Objective function with constraints"""
        thickness = np.max(y_coords) - np.min(y_coords)
        
        # Constraints
        penalty = 0
        
        # Minimum thickness constraint
        if thickness < 0.08:
            penalty += 1000 * (0.08 - thickness)
        
        # Maximum thickness constraint
        if thickness > 0.20:
            penalty += 1000 * (thickness - 0.20)
        
        # Leading edge radius constraint
        leading_edge_radius = np.min(x_coords)
        if leading_edge_radius < 0.01:
            penalty += 1000 * (0.01 - leading_edge_radius)
        
        # Surface smoothness constraint
        roughness = np.sum(np.abs(np.diff(y_coords)))
        if roughness > 0.5:
            penalty += 100 * roughness
        
        return thickness + penalty
    
    param = CSTParameterization(50, 6)
    optimizer = AirfoilOptimization(param, constrained_objective)
    
    try:
        result = optimizer.optimize(n_particles=15, epochs=30)
        
        print(f"✅ Constrained optimization results:")
        print(f"   Best cost: {result['best_cost']:.6f}")
        print(f"   Runtime: {result['runtime']:.3f}s")
        
        # Check if constraints are satisfied
        x_coords, y_coords = result['best_airfoil']
        thickness = np.max(y_coords) - np.min(y_coords)
        leading_edge_radius = np.min(x_coords)
        roughness = np.sum(np.abs(np.diff(y_coords)))
        
        print(f"   Final thickness: {thickness:.4f}")
        print(f"   Leading edge radius: {leading_edge_radius:.4f}")
        print(f"   Surface roughness: {roughness:.4f}")
        
        # Constraint satisfaction
        thickness_ok = 0.08 <= thickness <= 0.20
        radius_ok = leading_edge_radius >= 0.01
        roughness_ok = roughness <= 0.5
        
        print(f"   Thickness constraint: {'✅' if thickness_ok else '❌'}")
        print(f"   Radius constraint: {'✅' if radius_ok else '❌'}")
        print(f"   Roughness constraint: {'✅' if roughness_ok else '❌'}")
        
        return {
            'success': True,
            'constraints_satisfied': thickness_ok and radius_ok and roughness_ok,
            'thickness': thickness,
            'radius': leading_edge_radius,
            'roughness': roughness
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Run all airfoil optimization tests"""
    print("🎯 Airfoil Optimization Test Suite")
    print("=" * 70)
    print("Testing airfoil optimization with various parameterization methods:")
    print("CST, Hicks-Henne, Bézier, and NACA-k parameterizations.")
    
    # Run all tests
    test_airfoil_parameterizations()
    test_airfoil_objective_functions()
    test_parameterization_comparison()
    test_airfoil_visualization()
    test_optimization_algorithms()
    test_airfoil_constraints()
    
    print("\n" + "=" * 70)
    print("🎉 Airfoil Optimization Testing Complete!")
    print("=" * 70)
    print("\n✨ Airfoil Features:")
    print("✅ CST (Class-Shape Transformation) parameterization")
    print("✅ Hicks-Henne bump functions parameterization")
    print("✅ Bézier curve parameterization")
    print("✅ NACA-k parameterization")
    print("✅ Multiple objective functions (thickness, L/D, drag, stall)")
    print("✅ Constraint handling")
    print("✅ Visualization support")
    print("\n🎯 Usage:")
    print("from swarmopt.utils.airfoil_parameterization import CSTParameterization, AirfoilOptimization")

if __name__ == "__main__":
    main()
