"""
Swarm Diversity Measurement and Management

This module provides comprehensive diversity measurement tools for PSO swarms,
including various diversity metrics and diversity-based interventions.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform

def calculate_swarm_diversity(particles: List[np.ndarray], method: str = 'euclidean') -> float:
    """
    Calculate overall swarm diversity using various metrics
    
    Parameters:
    -----------
    particles : List[np.ndarray]
        List of particle positions
    method : str
        Diversity calculation method
        
    Returns:
    --------
    float : Diversity measure
    """
    if len(particles) < 2:
        return 0.0
    
    positions = np.array(particles)
    
    if method == 'euclidean':
        # Average pairwise Euclidean distance
        distances = pdist(positions, metric='euclidean')
        return np.mean(distances)
    
    elif method == 'manhattan':
        # Average pairwise Manhattan distance
        distances = pdist(positions, metric='cityblock')
        return np.mean(distances)
    
    elif method == 'variance':
        # Variance-based diversity
        return np.var(positions, axis=0).sum()
    
    elif method == 'radius':
        # Radius of the smallest sphere containing all particles
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        return np.max(distances)
    
    elif method == 'entropy':
        # Entropy-based diversity (discretized)
        # Discretize positions into bins
        n_bins = 10
        min_vals = np.min(positions, axis=0)
        max_vals = np.max(positions, axis=0)
        
        entropy = 0.0
        for dim in range(positions.shape[1]):
            bins = np.linspace(min_vals[dim], max_vals[dim], n_bins + 1)
            hist, _ = np.histogram(positions[:, dim], bins=bins)
            hist = hist[hist > 0]  # Remove zero counts
            if len(hist) > 0:
                probs = hist / np.sum(hist)
                entropy += -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy / positions.shape[1]
    
    else:
        raise ValueError(f"Unknown diversity method: {method}")

def calculate_dimension_diversity(particles: List[np.ndarray]) -> np.ndarray:
    """
    Calculate diversity for each dimension separately
    
    Parameters:
    -----------
    particles : List[np.ndarray]
        List of particle positions
        
    Returns:
    --------
    np.ndarray : Diversity for each dimension
    """
    if len(particles) < 2:
        return np.array([0.0] * len(particles[0]))
    
    positions = np.array(particles)
    return np.var(positions, axis=0)

def calculate_convergence_diversity(particles: List[np.ndarray], 
                                  convergence_threshold: float = 0.01) -> dict:
    """
    Calculate diversity metrics related to convergence
    
    Parameters:
    -----------
    particles : List[np.ndarray]
        List of particle positions
    convergence_threshold : float
        Threshold for considering particles converged
        
    Returns:
    --------
    dict : Convergence diversity metrics
    """
    if len(particles) < 2:
        return {
            'diversity': 0.0,
            'convergence_ratio': 1.0,
            'cluster_count': 1,
            'is_converged': True
        }
    
    positions = np.array(particles)
    
    # Calculate pairwise distances
    distances = pdist(positions, metric='euclidean')
    mean_distance = np.mean(distances)
    
    # Count converged particles (close to each other)
    converged_pairs = np.sum(distances < convergence_threshold)
    total_pairs = len(distances)
    convergence_ratio = converged_pairs / total_pairs if total_pairs > 0 else 0
    
    # Estimate number of clusters using distance threshold
    distance_matrix = squareform(distances)
    cluster_count = estimate_cluster_count(distance_matrix, convergence_threshold)
    
    return {
        'diversity': mean_distance,
        'convergence_ratio': convergence_ratio,
        'cluster_count': cluster_count,
        'is_converged': convergence_ratio > 0.8
    }

def estimate_cluster_count(distance_matrix: np.ndarray, 
                          threshold: float) -> int:
    """
    Estimate number of clusters in the swarm
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise distance matrix
    threshold : float
        Distance threshold for clustering
        
    Returns:
    --------
    int : Estimated number of clusters
    """
    n_particles = distance_matrix.shape[0]
    visited = np.zeros(n_particles, dtype=bool)
    cluster_count = 0
    
    for i in range(n_particles):
        if not visited[i]:
            cluster_count += 1
            # Mark all particles within threshold as visited
            close_particles = distance_matrix[i] < threshold
            visited[close_particles] = True
    
    return cluster_count

def calculate_velocity_diversity(particles: List, method: str = 'magnitude') -> float:
    """
    Calculate diversity based on particle velocities
    
    Parameters:
    -----------
    particles : List
        List of particle objects with velocity attributes
    method : str
        Velocity diversity calculation method
        
    Returns:
    --------
    float : Velocity diversity measure
    """
    if len(particles) < 2:
        return 0.0
    
    velocities = [p.velocity for p in particles if hasattr(p, 'velocity')]
    if len(velocities) < 2:
        return 0.0
    
    velocities = np.array(velocities)
    
    if method == 'magnitude':
        # Diversity in velocity magnitudes
        magnitudes = np.linalg.norm(velocities, axis=1)
        return np.std(magnitudes)
    
    elif method == 'direction':
        # Diversity in velocity directions
        # Normalize velocities to unit vectors
        unit_velocities = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-10)
        distances = pdist(unit_velocities, metric='cosine')
        return np.mean(distances)
    
    elif method == 'euclidean':
        # Euclidean distance between velocity vectors
        distances = pdist(velocities, metric='euclidean')
        return np.mean(distances)
    
    else:
        raise ValueError(f"Unknown velocity diversity method: {method}")

def calculate_fitness_diversity(particles: List) -> float:
    """
    Calculate diversity based on particle fitness values
    
    Parameters:
    -----------
    particles : List
        List of particle objects with best_cost attributes
        
    Returns:
    --------
    float : Fitness diversity measure
    """
    if len(particles) < 2:
        return 0.0
    
    costs = [p.best_cost for p in particles if hasattr(p, 'best_cost')]
    if len(costs) < 2:
        return 0.0
    
    costs = np.array(costs)
    return np.std(costs)

def detect_diversity_crisis(particles: List[np.ndarray], 
                           diversity_threshold: float = 0.1,
                           method: str = 'euclidean') -> bool:
    """
    Detect if swarm is experiencing a diversity crisis
    
    Parameters:
    -----------
    particles : List[np.ndarray]
        List of particle positions
    diversity_threshold : float
        Threshold below which diversity is considered too low
    method : str
        Diversity calculation method
        
    Returns:
    --------
    bool : True if diversity crisis detected
    """
    diversity = calculate_swarm_diversity(particles, method)
    return diversity < diversity_threshold

def calculate_swarm_statistics(particles: List, 
                              positions: Optional[List[np.ndarray]] = None) -> dict:
    """
    Calculate comprehensive swarm statistics
    
    Parameters:
    -----------
    particles : List
        List of particle objects
    positions : Optional[List[np.ndarray]]
        Optional pre-computed positions
        
    Returns:
    --------
    dict : Comprehensive swarm statistics
    """
    if positions is None:
        positions = [p.pos for p in particles if hasattr(p, 'pos')]
    
    if len(positions) < 2:
        return {
            'diversity': 0.0,
            'dimension_diversity': np.array([0.0]),
            'convergence_metrics': {'is_converged': True},
            'velocity_diversity': 0.0,
            'fitness_diversity': 0.0,
            'crisis_detected': True
        }
    
    # Basic diversity metrics
    diversity = calculate_swarm_diversity(positions)
    dimension_diversity = calculate_dimension_diversity(positions)
    convergence_metrics = calculate_convergence_diversity(positions)
    
    # Velocity diversity
    velocity_diversity = calculate_velocity_diversity(particles)
    
    # Fitness diversity
    fitness_diversity = calculate_fitness_diversity(particles)
    
    # Crisis detection
    crisis_detected = detect_diversity_crisis(positions)
    
    return {
        'diversity': diversity,
        'dimension_diversity': dimension_diversity,
        'convergence_metrics': convergence_metrics,
        'velocity_diversity': velocity_diversity,
        'fitness_diversity': fitness_diversity,
        'crisis_detected': crisis_detected,
        'n_particles': len(positions)
    }

def recommend_diversity_intervention(swarm_stats: dict) -> str:
    """
    Recommend diversity intervention based on swarm statistics
    
    Parameters:
    -----------
    swarm_stats : dict
        Swarm statistics from calculate_swarm_statistics
        
    Returns:
    --------
    str : Recommended intervention strategy
    """
    if swarm_stats['crisis_detected']:
        if swarm_stats['convergence_metrics']['is_converged']:
            return 'restart'  # Complete restart needed
        elif swarm_stats['diversity'] < 0.05:
            return 'escape_local_optima'  # Strong escape variation
        else:
            return 'diversity_preserving'  # Gentle diversity restoration
    
    elif swarm_stats['convergence_metrics']['convergence_ratio'] > 0.5:
        return 'opposition_based'  # Explore opposite regions
    
    elif swarm_stats['velocity_diversity'] < 0.01:
        return 'adaptive_strength'  # Increase variation strength
    
    else:
        return 'none'  # No intervention needed

# Diversity monitoring class
class DiversityMonitor:
    """
    Monitor swarm diversity over time and recommend interventions
    """
    
    def __init__(self, diversity_threshold: float = 0.1, 
                 convergence_threshold: float = 0.01):
        self.diversity_threshold = diversity_threshold
        self.convergence_threshold = convergence_threshold
        self.diversity_history = []
        self.intervention_history = []
    
    def update(self, particles: List) -> dict:
        """
        Update diversity monitoring and return recommendations
        
        Parameters:
        -----------
        particles : List
            List of particle objects
            
        Returns:
        --------
        dict : Monitoring results and recommendations
        """
        positions = [p.pos for p in particles if hasattr(p, 'pos')]
        stats = calculate_swarm_statistics(particles, positions)
        
        # Update history
        self.diversity_history.append(stats['diversity'])
        intervention = recommend_diversity_intervention(stats)
        self.intervention_history.append(intervention)
        
        return {
            'stats': stats,
            'recommended_intervention': intervention,
            'diversity_trend': self._calculate_trend(),
            'needs_intervention': intervention != 'none'
        }
    
    def _calculate_trend(self) -> str:
        """Calculate diversity trend over time"""
        if len(self.diversity_history) < 3:
            return 'insufficient_data'
        
        recent = self.diversity_history[-3:]
        if recent[-1] > recent[0] * 1.1:
            return 'increasing'
        elif recent[-1] < recent[0] * 0.9:
            return 'decreasing'
        else:
            return 'stable'
