"""
Airfoil Parameterization Methods

This module implements various airfoil parameterization techniques for
aerodynamic optimization using swarm intelligence.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings

class AirfoilParameterization:
    """
    Base class for airfoil parameterization methods
    """
    
    def __init__(self, n_points: int = 100):
        self.n_points = n_points
        self.x_coords = np.linspace(0, 1, n_points)
    
    def generate_airfoil(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate airfoil coordinates from parameters
        
        Parameters:
        -----------
        params : np.ndarray
            Parameterization parameters
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (x_coords, y_coords)
        """
        raise NotImplementedError("Subclasses must implement generate_airfoil")
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get parameter bounds for optimization
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (lower_bounds, upper_bounds)
        """
        raise NotImplementedError("Subclasses must implement get_bounds")

class CSTParameterization(AirfoilParameterization):
    """
    Class-Shape Transformation (CST) parameterization
    
    The CST method represents airfoils using Bernstein polynomials
    with class and shape functions.
    """
    
    def __init__(self, n_points: int = 100, n_params: int = 8):
        super().__init__(n_points)
        self.n_params = n_params
        self.n_class = 2  # Leading edge radius and trailing edge thickness
        self.n_shape = n_params - self.n_class
    
    def generate_airfoil(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate airfoil using CST parameterization"""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        # Separate class and shape parameters
        class_params = params[:self.n_class]
        shape_params = params[self.n_class:]
        
        # Class function: C(x) = x^0.5 * (1-x)^1.0
        class_func = np.power(self.x_coords, 0.5) * np.power(1 - self.x_coords, 1.0)
        
        # Shape function using Bernstein polynomials
        shape_func = self._bernstein_polynomial(shape_params)
        
        # Combine class and shape functions
        y_upper = class_func * (1 + shape_func)
        y_lower = class_func * (1 - shape_func)
        
        # Apply leading edge radius and trailing edge thickness
        y_upper *= (1 + class_params[0] * (1 - self.x_coords))
        y_lower *= (1 + class_params[0] * (1 - self.x_coords))
        
        # Add trailing edge thickness
        te_thickness = class_params[1]
        y_upper += te_thickness * (1 - self.x_coords)
        y_lower -= te_thickness * (1 - self.x_coords)
        
        # Combine upper and lower surfaces
        x_coords = np.concatenate([self.x_coords[::-1], self.x_coords[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        return x_coords, y_coords
    
    def _bernstein_polynomial(self, coeffs: np.ndarray) -> np.ndarray:
        """Calculate Bernstein polynomial"""
        n = len(coeffs) - 1
        result = np.zeros_like(self.x_coords)
        
        for i, coeff in enumerate(coeffs):
            # Bernstein basis function
            basis = self._binomial_coefficient(n, i) * \
                   np.power(self.x_coords, i) * \
                   np.power(1 - self.x_coords, n - i)
            result += coeff * basis
        
        return result
    
    def _binomial_coefficient(self, n: int, k: int) -> int:
        """Calculate binomial coefficient C(n,k)"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for CST"""
        # Class parameters: leading edge radius [0, 0.1], trailing edge thickness [0, 0.05]
        # Shape parameters: Bernstein coefficients [-0.5, 0.5]
        lower = np.concatenate([[0.0, 0.0], [-0.5] * self.n_shape])
        upper = np.concatenate([[0.1, 0.05], [0.5] * self.n_shape])
        return lower, upper

class HicksHenneParameterization(AirfoilParameterization):
    """
    Hicks-Henne bump functions parameterization
    
    Uses a series of Hicks-Henne bump functions to modify
    a baseline airfoil shape.
    """
    
    def __init__(self, n_points: int = 100, n_bumps: int = 8):
        super().__init__(n_points)
        self.n_bumps = n_bumps
        self.n_params = n_bumps * 2  # amplitude and location for each bump
        
        # Baseline NACA 0012 airfoil
        self.baseline_x, self.baseline_y = self._generate_naca0012()
    
    def _generate_naca0012(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate baseline NACA 0012 airfoil"""
        # NACA 0012 coordinates (simplified)
        x = self.x_coords
        t = 0.12  # 12% thickness
        
        # NACA 4-digit series formula
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                     0.2843 * x**3 - 0.1036 * x**4)
        
        y_upper = yt
        y_lower = -yt
        
        x_coords = np.concatenate([x[::-1], x[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        return x_coords, y_coords
    
    def generate_airfoil(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate airfoil using Hicks-Henne parameterization"""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        # Separate amplitudes and locations
        amplitudes = params[:self.n_bumps]
        locations = params[self.n_bumps:]
        
        # Generate bump functions
        bump_upper = np.zeros_like(self.x_coords)
        bump_lower = np.zeros_like(self.x_coords)
        
        for i in range(self.n_bumps):
            amplitude = amplitudes[i]
            location = locations[i]
            
            # Hicks-Henne bump function
            bump = self._hicks_henne_bump(self.x_coords, location)
            
            # Apply to upper and lower surfaces
            bump_upper += amplitude * bump
            bump_lower -= amplitude * bump
        
        # Add bumps to baseline
        # Ensure baseline has the right number of points
        if len(self.baseline_y) != 2 * self.n_points - 1:
            # Regenerate baseline with correct number of points
            self.baseline_x, self.baseline_y = self._generate_naca0012()
        
        y_upper = self.baseline_y[:self.n_points] + bump_upper
        y_lower = self.baseline_y[self.n_points:] + bump_lower
        
        # Combine surfaces
        x_coords = np.concatenate([self.x_coords[::-1], self.x_coords[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower])
        
        return x_coords, y_coords
    
    def _hicks_henne_bump(self, x: np.ndarray, location: float) -> np.ndarray:
        """Generate Hicks-Henne bump function"""
        # Ensure location is in valid range
        location = np.clip(location, 0.1, 0.9)
        
        # Hicks-Henne formula
        bump = np.sin(np.pi * np.power(x / location, 0.5))**2
        
        # Apply only in the vicinity of the location
        mask = (x >= 0.05) & (x <= location * 1.5)
        bump = np.where(mask, bump, 0)
        
        return bump
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for Hicks-Henne"""
        # Amplitudes: [-0.05, 0.05], Locations: [0.1, 0.9]
        lower = np.concatenate([[-0.05] * self.n_bumps, [0.1] * self.n_bumps])
        upper = np.concatenate([[0.05] * self.n_bumps, [0.9] * self.n_bumps])
        return lower, upper

class BezierParameterization(AirfoilParameterization):
    """
    Bézier curve parameterization
    
    Uses Bézier curves to represent airfoil surfaces
    with control points as optimization parameters.
    """
    
    def __init__(self, n_points: int = 100, n_control_points: int = 6):
        super().__init__(n_points)
        self.n_control_points = n_control_points
        self.n_params = n_control_points * 2  # x and y for each control point
        
        # Fixed x-coordinates for control points
        self.control_x = np.linspace(0, 1, n_control_points)
    
    def generate_airfoil(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate airfoil using Bézier parameterization"""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        # Separate x and y coordinates
        control_x = params[:self.n_control_points]
        control_y = params[self.n_control_points:]
        
        # Generate Bézier curves for upper and lower surfaces
        upper_curve = self._bezier_curve(self.control_x, control_y, self.x_coords)
        lower_curve = self._bezier_curve(self.control_x, -control_y, self.x_coords)
        
        # Combine surfaces
        x_coords = np.concatenate([self.x_coords[::-1], self.x_coords[1:]])
        y_coords = np.concatenate([upper_curve[::-1], lower_curve[1:]])
        
        return x_coords, y_coords
    
    def _bezier_curve(self, control_x: np.ndarray, control_y: np.ndarray, 
                     t_values: np.ndarray) -> np.ndarray:
        """Generate Bézier curve from control points"""
        n = len(control_x) - 1
        curve_y = np.zeros_like(t_values)
        
        for i, t in enumerate(t_values):
            # De Casteljau's algorithm
            points_x = control_x.copy()
            points_y = control_y.copy()
            
            for r in range(1, n + 1):
                for j in range(n - r + 1):
                    points_x[j] = (1 - t) * points_x[j] + t * points_x[j + 1]
                    points_y[j] = (1 - t) * points_y[j] + t * points_y[j + 1]
            
            curve_y[i] = points_y[0]
        
        return curve_y
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for Bézier"""
        # X coordinates: [0, 1], Y coordinates: [-0.2, 0.2]
        lower = np.concatenate([[0.0] * self.n_control_points, [-0.2] * self.n_control_points])
        upper = np.concatenate([[1.0] * self.n_control_points, [0.2] * self.n_control_points])
        return lower, upper

class NACAKParameterization(AirfoilParameterization):
    """
    NACA-k parameterization
    
    Uses NACA 4-digit series with k-parameter modification
    for trailing edge shape control.
    """
    
    def __init__(self, n_points: int = 100, n_k_params: int = 4):
        super().__init__(n_points)
        self.n_k_params = n_k_params
        self.n_params = 3 + n_k_params  # thickness, camber, camber position, k-parameters
        
        # NACA 4-digit series parameters
        self.thickness = 0.12  # 12% thickness
        self.camber = 0.02     # 2% camber
        self.camber_pos = 0.4  # 40% chord position
    
    def generate_airfoil(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate airfoil using NACA-k parameterization"""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        # Extract parameters
        thickness = params[0]
        camber = params[1]
        camber_pos = params[2]
        k_params = params[3:]
        
        # Generate NACA 4-digit series
        x = self.x_coords
        yt = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                           0.2843 * x**3 - 0.1036 * x**4)
        
        # Camber line
        if camber > 0:
            yc = self._camber_line(x, camber, camber_pos)
            dyc_dx = self._camber_slope(x, camber, camber_pos)
            
            # Upper and lower surfaces
            theta = np.arctan(dyc_dx)
            y_upper = yc + yt * np.cos(theta)
            y_lower = yc - yt * np.cos(theta)
        else:
            y_upper = yt
            y_lower = -yt
        
        # Apply k-parameter modifications
        y_upper = self._apply_k_modifications(y_upper, x, k_params)
        y_lower = self._apply_k_modifications(y_lower, x, k_params)
        
        # Combine surfaces
        x_coords = np.concatenate([x[::-1], x[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        return x_coords, y_coords
    
    def _camber_line(self, x: np.ndarray, camber: float, camber_pos: float) -> np.ndarray:
        """Generate camber line"""
        if camber_pos == 0:
            return np.zeros_like(x)
        
        yc = np.zeros_like(x)
        mask1 = x <= camber_pos
        mask2 = x > camber_pos
        
        if np.any(mask1):
            yc[mask1] = (camber / camber_pos**2) * (2 * camber_pos * x[mask1] - x[mask1]**2)
        
        if np.any(mask2):
            yc[mask2] = (camber / (1 - camber_pos)**2) * ((1 - 2 * camber_pos) + 
                                                          2 * camber_pos * x[mask2] - x[mask2]**2)
        
        return yc
    
    def _camber_slope(self, x: np.ndarray, camber: float, camber_pos: float) -> np.ndarray:
        """Calculate camber line slope"""
        if camber_pos == 0:
            return np.zeros_like(x)
        
        dyc_dx = np.zeros_like(x)
        mask1 = x <= camber_pos
        mask2 = x > camber_pos
        
        if np.any(mask1):
            dyc_dx[mask1] = (2 * camber / camber_pos**2) * (camber_pos - x[mask1])
        
        if np.any(mask2):
            dyc_dx[mask2] = (2 * camber / (1 - camber_pos)**2) * (camber_pos - x[mask2])
        
        return dyc_dx
    
    def _apply_k_modifications(self, y: np.ndarray, x: np.ndarray, k_params: np.ndarray) -> np.ndarray:
        """Apply k-parameter modifications"""
        # Simple k-parameter modification (can be extended)
        k_modification = np.zeros_like(x)
        
        for i, k in enumerate(k_params):
            # Apply k-parameter at different chord positions
            position = (i + 1) / (len(k_params) + 1)
            weight = np.exp(-((x - position) / 0.1)**2)
            k_modification += k * weight
        
        return y + k_modification
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get parameter bounds for NACA-k"""
        # Thickness: [0.08, 0.20], Camber: [0, 0.05], Camber position: [0.2, 0.6]
        # K-parameters: [-0.02, 0.02]
        lower = np.concatenate([[0.08, 0.0, 0.2], [-0.02] * self.n_k_params])
        upper = np.concatenate([[0.20, 0.05, 0.6], [0.02] * self.n_k_params])
        return lower, upper

class AirfoilOptimization:
    """
    Airfoil optimization using swarm intelligence
    """
    
    def __init__(self, parameterization: AirfoilParameterization, 
                 objective_func: callable):
        self.parameterization = parameterization
        self.objective_func = objective_func
        self.bounds = parameterization.get_bounds()
    
    def optimize(self, n_particles: int = 20, epochs: int = 100, 
                algorithm: str = 'global') -> Dict:
        """
        Optimize airfoil using swarm intelligence
        
        Parameters:
        -----------
        n_particles : int
            Number of particles
        epochs : int
            Number of iterations
        algorithm : str
            PSO algorithm to use
            
        Returns:
        --------
        Dict : Optimization results
        """
        from swarmopt import Swarm
        
        # Create objective function wrapper
        def obj_func(params):
            try:
                x_coords, y_coords = self.parameterization.generate_airfoil(params)
                return self.objective_func(x_coords, y_coords)
            except Exception as e:
                return float('inf')  # Penalty for invalid airfoils
        
        # Get parameter dimensions
        dims = len(self.bounds[0])
        
        # Create swarm
        swarm = Swarm(
            n_particles=n_particles,
            dims=dims,
            c1=2.0, c2=2.0, w=0.9,
            epochs=epochs,
            obj_func=obj_func,
            algo=algorithm,
            velocity_clamp=(self.bounds[0], self.bounds[1])
        )
        
        # Optimize
        swarm.optimize()
        
        # Generate best airfoil
        best_airfoil = self.parameterization.generate_airfoil(swarm.best_pos)
        
        return {
            'best_params': swarm.best_pos,
            'best_cost': swarm.best_cost,
            'best_airfoil': best_airfoil,
            'runtime': swarm.runtime,
            'parameterization': self.parameterization.__class__.__name__
        }
