#!/usr/bin/env python3
"""
TDA-Guided PSO for Cancer Growth Modeling and Control

This example demonstrates how to use SwarmOpt with Topological Data Analysis
to model and control cancer growth dynamics.

Research Concept:
- Use persistent homology to characterize tumor morphology
- Optimize tumor growth simulator parameters via PSO to match observed topology
- Search for treatment policies that collapse invasive topological features

Dependencies:
    pip install swarmopt numpy scipy
    pip install giotto-tda  # For persistent homology
    # or
    pip install gudhi      # Alternative TDA library
"""

import numpy as np
from swarmopt import Swarm

# ============================================================================
# PART 1: TDA UTILITIES
# ============================================================================

class PersistenceDiagram:
    """
    Handles persistent homology computations and Wasserstein distance
    """
    
    def __init__(self, method='ripser'):
        """
        Initialize persistence diagram computer
        
        Parameters:
        -----------
        method : str
            'ripser' (giotto-tda) or 'gudhi' (GUDHI)
        """
        self.method = method
        
        if method == 'ripser':
            try:
                from gtda.homology import VietorisRipsPersistence
                self.computer = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
            except ImportError:
                print("⚠️  giotto-tda not installed. Install with: pip install giotto-tda")
                self.computer = None
        elif method == 'gudhi':
            try:
                import gudhi
                self.computer = gudhi
            except ImportError:
                print("⚠️  gudhi not installed. Install with: pip install gudhi")
                self.computer = None
    
    def compute_persistence(self, point_cloud):
        """
        Compute persistence diagram from point cloud
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Point cloud representing tumor cells (N x 3 for 3D)
        
        Returns:
        --------
        dict : Persistence diagrams for each dimension
        """
        if self.computer is None:
            # Mock implementation for demonstration
            return self._mock_persistence(point_cloud)
        
        if self.method == 'ripser':
            # Giotto-TDA implementation
            diagrams = self.computer.fit_transform([point_cloud])
            return self._parse_gtda_diagrams(diagrams[0])
        
        elif self.method == 'gudhi':
            # GUDHI implementation
            return self._compute_gudhi_persistence(point_cloud)
    
    def _mock_persistence(self, point_cloud):
        """Mock persistence for demonstration without TDA libraries"""
        n_points = len(point_cloud)
        
        # Generate mock persistence features
        # H0: Connected components (birth, death, dimension=0)
        h0 = np.array([[0.0, 0.5], [0.0, 0.3], [0.0, 0.2]])
        
        # H1: Loops/voids (birth, death, dimension=1)
        h1 = np.array([[0.1, 0.8], [0.2, 0.6]])
        
        # H2: Cavities (birth, death, dimension=2)
        h2 = np.array([[0.3, 0.7]])
        
        return {'H0': h0, 'H1': h1, 'H2': h2}
    
    def _parse_gtda_diagrams(self, diagrams):
        """Parse giotto-tda persistence diagrams"""
        result = {}
        for dim in [0, 1, 2]:
            dim_diagram = diagrams[diagrams[:, 2] == dim][:, :2]
            result[f'H{dim}'] = dim_diagram
        return result
    
    def _compute_gudhi_persistence(self, point_cloud):
        """Compute persistence using GUDHI"""
        import gudhi
        
        # Create Rips complex
        rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract diagrams by dimension
        result = {}
        for dim in [0, 1, 2]:
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            result[f'H{dim}'] = pairs
        
        return result
    
    def wasserstein_distance(self, diagram1, diagram2, dimension='H1', order=2):
        """
        Compute Wasserstein distance between persistence diagrams
        
        Parameters:
        -----------
        diagram1, diagram2 : dict
            Persistence diagrams
        dimension : str
            Homology dimension ('H0', 'H1', 'H2')
        order : int
            Order of Wasserstein distance (typically 1 or 2)
        
        Returns:
        --------
        float : Wasserstein distance
        """
        d1 = diagram1.get(dimension, np.array([]))
        d2 = diagram2.get(dimension, np.array([]))
        
        if len(d1) == 0 or len(d2) == 0:
            return float('inf')
        
        # Simplified Wasserstein distance (use scipy.stats.wasserstein_distance for exact)
        # This is a placeholder - use proper implementation
        from scipy.spatial.distance import cdist
        
        # Compute pairwise distances
        distances = cdist(d1, d2, metric='euclidean')
        
        # Simplified matching (use Hungarian algorithm for exact)
        return np.mean(np.min(distances, axis=1))


# ============================================================================
# PART 2: TUMOR GROWTH SIMULATOR
# ============================================================================

class TumorGrowthSimulator:
    """
    Simple tumor growth simulator based on reaction-diffusion dynamics
    """
    
    def __init__(self, grid_size=20):
        """
        Initialize tumor growth simulator
        
        Parameters:
        -----------
        grid_size : int
            Size of 3D grid
        """
        self.grid_size = grid_size
        self.grid = None
    
    def simulate(self, params, timesteps=100):
        """
        Simulate tumor growth with given parameters
        
        Parameters:
        -----------
        params : np.ndarray
            [proliferation_rate, diffusion_rate, death_rate, angiogenesis_factor]
        timesteps : int
            Number of simulation steps
        
        Returns:
        --------
        np.ndarray : Point cloud of tumor cells at final timestep
        """
        prolif_rate, diffusion_rate, death_rate, angio_factor = params
        
        # Initialize grid with small tumor seed
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        center = self.grid_size // 2
        self.grid[center-2:center+2, center-2:center+2, center-2:center+2] = 1.0
        
        # Run simulation
        for t in range(timesteps):
            self.grid = self._step(self.grid, prolif_rate, diffusion_rate, death_rate, angio_factor)
        
        # Extract point cloud from grid
        return self._grid_to_pointcloud(self.grid)
    
    def _step(self, grid, prolif, diffusion, death, angio):
        """Single simulation step"""
        # Simplified reaction-diffusion model
        new_grid = grid.copy()
        
        # Proliferation
        new_grid += prolif * grid * (1 - grid)
        
        # Death
        new_grid -= death * grid
        
        # Angiogenesis (nutrient availability)
        new_grid += angio * grid * (1 - grid)
        
        # Simple diffusion (without scipy)
        new_grid += diffusion * (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) - 2 * grid)
        
        # Clip values
        new_grid = np.clip(new_grid, 0, 1)
        
        return new_grid
    
    def _grid_to_pointcloud(self, grid, threshold=0.5):
        """Convert grid to point cloud of tumor cells"""
        tumor_cells = np.argwhere(grid > threshold)
        
        # Downsample if too many points
        if len(tumor_cells) > 1000:
            indices = np.random.choice(len(tumor_cells), 1000, replace=False)
            tumor_cells = tumor_cells[indices]
        
        return tumor_cells.astype(float)


# ============================================================================
# PART 3: TDA-GUIDED OPTIMIZATION
# ============================================================================

class TDAGuidedOptimization:
    """
    Use PSO to fit tumor growth parameters using TDA-based objective
    """
    
    def __init__(self, observed_diagram, simulator, persistence_computer):
        """
        Initialize TDA-guided optimization
        
        Parameters:
        -----------
        observed_diagram : dict
            Target persistence diagram from observed tumor
        simulator : TumorGrowthSimulator
            Tumor growth simulator
        persistence_computer : PersistenceDiagram
            Persistence diagram computer
        """
        self.observed_diagram = observed_diagram
        self.simulator = simulator
        self.persistence_computer = persistence_computer
        self.history = []
    
    def objective_function(self, params):
        """
        Objective: Minimize Wasserstein distance between simulated and observed topology
        
        Parameters:
        -----------
        params : np.ndarray
            Tumor growth parameters
        
        Returns:
        --------
        float : Total topological distance
        """
        # Ensure positive parameters
        params = np.abs(params)
        
        # Simulate tumor growth
        point_cloud = self.simulator.simulate(params, timesteps=50)
        
        # Compute persistence diagram
        simulated_diagram = self.persistence_computer.compute_persistence(point_cloud)
        
        # Compute Wasserstein distances for each homology dimension
        total_distance = 0.0
        weights = {'H0': 0.3, 'H1': 0.5, 'H2': 0.2}  # Weight invasive features (H1) more
        
        for dim in ['H0', 'H1', 'H2']:
            distance = self.persistence_computer.wasserstein_distance(
                self.observed_diagram, simulated_diagram, dimension=dim
            )
            total_distance += weights[dim] * distance
        
        # Store for analysis
        self.history.append({
            'params': params.copy(),
            'distance': total_distance,
            'diagram': simulated_diagram
        })
        
        return total_distance
    
    def optimize_parameters(self, bounds=[(0.01, 0.5), (0.01, 0.3), (0.0, 0.1), (0.0, 0.2)]):
        """
        Run PSO to find optimal tumor growth parameters
        
        Parameters:
        -----------
        bounds : list of tuples
            Parameter bounds [(min, max), ...]
        
        Returns:
        --------
        dict : Optimization results
        """
        print("🔬 Running TDA-Guided Parameter Fitting...")
        
        # Extract bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        # Create swarm
        swarm = Swarm(
            n_particles=20,
            dims=len(bounds),
            c1=2.0, c2=2.0, w=0.9,
            epochs=30,
            obj_func=self.objective_function,
            algo='global',
            velocity_clamp=(lower_bounds, upper_bounds)
        )
        
        # Run optimization
        swarm.optimize()
        
        print(f"✅ Optimization complete!")
        print(f"   Best distance: {swarm.best_cost:.6f}")
        print(f"   Best parameters: {swarm.best_pos}")
        
        return {
            'best_params': swarm.best_pos,
            'best_distance': swarm.best_cost,
            'runtime': swarm.runtime,
            'history': self.history
        }


# ============================================================================
# PART 4: TREATMENT CONTROL POLICY SEARCH
# ============================================================================

class TreatmentPolicySearch:
    """
    Search for treatment policies that collapse invasive topology
    """
    
    def __init__(self, simulator, persistence_computer, target_topology='minimal_H1'):
        """
        Initialize treatment policy search
        
        Parameters:
        -----------
        simulator : TumorGrowthSimulator
            Tumor growth simulator
        persistence_computer : PersistenceDiagram
            Persistence diagram computer
        target_topology : str
            Target topological signature ('minimal_H1', 'compact', etc.)
        """
        self.simulator = simulator
        self.persistence_computer = persistence_computer
        self.target_topology = target_topology
        self.policy_history = []
    
    def treatment_objective(self, policy_params):
        """
        Objective: Find treatment that minimizes invasive topology (H1 features)
        
        Parameters:
        -----------
        policy_params : np.ndarray
            Treatment policy [timing, intensity, duration, drug_combination]
        
        Returns:
        --------
        float : Invasive topology score (lower is better)
        """
        timing, intensity, duration, drug_combo = np.abs(policy_params)
        
        # Simulate tumor with treatment
        base_params = [0.2, 0.1, 0.05, 0.1]  # Base tumor parameters
        
        # Apply treatment effect
        treated_params = base_params.copy()
        treated_params[0] *= (1 - intensity * drug_combo)  # Reduce proliferation
        treated_params[2] += intensity * (1 - drug_combo)  # Increase death
        
        # Simulate
        point_cloud = self.simulator.simulate(treated_params, timesteps=int(50 * duration))
        
        # Compute persistence
        diagram = self.persistence_computer.compute_persistence(point_cloud)
        
        # Score based on invasive features
        if self.target_topology == 'minimal_H1':
            # Minimize H1 features (loops/voids indicate invasion)
            h1_features = diagram.get('H1', np.array([]))
            
            # Score: number of persistent H1 features + their persistence
            if len(h1_features) == 0:
                score = 0.0
            else:
                persistence = h1_features[:, 1] - h1_features[:, 0]
                score = len(h1_features) + np.sum(persistence)
        
        elif self.target_topology == 'compact':
            # Minimize all higher-dimensional features
            h1 = diagram.get('H1', np.array([]))
            h2 = diagram.get('H2', np.array([]))
            score = len(h1) + 2 * len(h2)  # Weight cavities more
        
        else:
            score = 0.0
        
        # Store for analysis
        self.policy_history.append({
            'policy': policy_params.copy(),
            'score': score,
            'diagram': diagram
        })
        
        return score
    
    def search_optimal_treatment(self):
        """
        Search for optimal treatment policy using PSO
        
        Returns:
        --------
        dict : Optimal treatment policy
        """
        print("🎯 Searching for Optimal Treatment Policy...")
        
        # Create swarm
        swarm = Swarm(
            n_particles=25,
            dims=4,  # [timing, intensity, duration, drug_combination]
            c1=2.0, c2=2.0, w=0.9,
            epochs=40,
            obj_func=self.treatment_objective,
            algo='global',
            velocity_clamp=(0.0, 1.0)
        )
        
        # Run optimization
        swarm.optimize()
        
        timing, intensity, duration, drug_combo = swarm.best_pos
        
        print(f"✅ Optimal treatment policy found!")
        print(f"   Timing: {timing:.3f}")
        print(f"   Intensity: {intensity:.3f}")
        print(f"   Duration: {duration:.3f}")
        print(f"   Drug combination: {drug_combo:.3f}")
        print(f"   Invasive topology score: {swarm.best_cost:.6f}")
        
        return {
            'timing': timing,
            'intensity': intensity,
            'duration': duration,
            'drug_combination': drug_combo,
            'score': swarm.best_cost,
            'runtime': swarm.runtime,
            'history': self.policy_history
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    """
    Demonstrate TDA-guided PSO for cancer modeling and control
    """
    print("=" * 70)
    print("🔬 TDA-Guided PSO for Cancer Growth Modeling and Control")
    print("=" * 70)
    
    # Initialize components
    print("\n📊 Initializing TDA components...")
    persistence_computer = PersistenceDiagram(method='ripser')
    simulator = TumorGrowthSimulator(grid_size=15)
    
    # Generate "observed" tumor data (in practice, this comes from imaging)
    print("📸 Generating observed tumor topology...")
    observed_params = [0.25, 0.15, 0.03, 0.12]
    observed_pointcloud = simulator.simulate(observed_params, timesteps=50)
    observed_diagram = persistence_computer.compute_persistence(observed_pointcloud)
    
    print(f"   H0 features: {len(observed_diagram.get('H0', []))}")
    print(f"   H1 features: {len(observed_diagram.get('H1', []))} (invasive topology)")
    print(f"   H2 features: {len(observed_diagram.get('H2', []))}")
    
    # Part 1: Parameter Fitting
    print("\n" + "=" * 70)
    print("PART 1: Parameter Fitting with TDA")
    print("=" * 70)
    
    optimizer = TDAGuidedOptimization(observed_diagram, simulator, persistence_computer)
    fitting_results = optimizer.optimize_parameters()
    
    print(f"\n📊 Fitting Results:")
    print(f"   True parameters: {observed_params}")
    print(f"   Fitted parameters: {fitting_results['best_params']}")
    print(f"   Final distance: {fitting_results['best_distance']:.6f}")
    
    # Part 2: Treatment Policy Search
    print("\n" + "=" * 70)
    print("PART 2: Treatment Policy Search")
    print("=" * 70)
    
    policy_search = TreatmentPolicySearch(simulator, persistence_computer, target_topology='minimal_H1')
    treatment_results = policy_search.search_optimal_treatment()
    
    print(f"\n🎯 Treatment Policy:")
    print(f"   Optimal timing: {treatment_results['timing']:.3f}")
    print(f"   Optimal intensity: {treatment_results['intensity']:.3f}")
    print(f"   Optimal duration: {treatment_results['duration']:.3f}")
    print(f"   Drug combination: {treatment_results['drug_combination']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ TDA-Guided Optimization Complete!")
    print("=" * 70)
    
    print("\n💡 Next Steps:")
    print("   1. Install TDA library: pip install giotto-tda or pip install gudhi")
    print("   2. Use real tumor imaging data for observed topology")
    print("   3. Implement more sophisticated tumor growth models")
    print("   4. Add temporal tracking of topology evolution")
    print("   5. Validate treatment policies in clinical simulations")

if __name__ == "__main__":
    main()
