import unittest
import numpy as np
from swarmopt.utils.variation import *

class TestVariation(unittest.TestCase):
    def setUp(self):
        self.particle_pos = np.array([1.0, 2.0, 3.0])
        self.bounds = (-5.0, 5.0)
        self.variation_rate = 0.5
        self.variation_strength = 0.1
        self.current_iter = 50
        self.max_iter = 100
        
    def test_gaussian_variation(self):
        result = gaussian_variation(self.particle_pos, self.variation_rate, self.variation_strength)
        self.assertEqual(result.shape, self.particle_pos.shape)
        # Check that at least some dimensions may be changed
        # Since it's random, we can't be sure
        
    def test_uniform_variation(self):
        variation_range = (-1.0, 1.0)
        result = uniform_variation(self.particle_pos, self.variation_rate, variation_range)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_polynomial_variation(self):
        result = polynomial_variation(self.particle_pos, self.variation_rate, 
                                     eta=20.0, bounds=self.bounds)
        self.assertEqual(result.shape, self.particle_pos.shape)
        # Check bounds are respected
        self.assertTrue(np.all(result >= self.bounds[0]))
        self.assertTrue(np.all(result <= self.bounds[1]))
        
    def test_cauchy_variation(self):
        result = cauchy_variation(self.particle_pos, self.variation_rate, scale=0.1)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_levy_variation(self):
        result = levy_variation(self.particle_pos, self.variation_rate, alpha=1.5, beta=1.0)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_adaptive_variation(self):
        result = adaptive_variation(self.particle_pos, self.current_iter, self.max_iter,
                                   self.variation_rate, self.variation_strength)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_chaotic_variation(self):
        result = chaotic_variation(self.particle_pos, self.variation_rate, chaos_param=3.9)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_differential_variation(self):
        population = [
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]
        result = differential_variation(self.particle_pos, population, 
                                       self.variation_rate, f=0.5)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_boundary_variation(self):
        result = boundary_variation(self.particle_pos, self.bounds, self.variation_rate)
        self.assertEqual(result.shape, self.particle_pos.shape)
        # Check bounds are respected
        self.assertTrue(np.all(result >= self.bounds[0]))
        self.assertTrue(np.all(result <= self.bounds[1]))
        
    def test_non_uniform_variation(self):
        result = non_uniform_variation(self.particle_pos, self.current_iter, self.max_iter,
                                      self.variation_rate, self.bounds, b=2.0)
        self.assertEqual(result.shape, self.particle_pos.shape)
        # Check bounds are respected
        self.assertTrue(np.all(result >= self.bounds[0]))
        self.assertTrue(np.all(result <= self.bounds[1]))
        
    def test_escape_local_optima_variation(self):
        result = escape_local_optima_variation(self.particle_pos, self.bounds, escape_strength=2.0)
        self.assertEqual(result.shape, self.particle_pos.shape)
        self.assertTrue(np.all(result >= self.bounds[0]))
        self.assertTrue(np.all(result <= self.bounds[1]))
        
    def test_diversity_preserving_variation(self):
        population = [
            np.array([0.0, 1.0, 2.0]),
            np.array([4.0, 5.0, 6.0])
        ]
        result = diversity_preserving_variation(self.particle_pos, population, 
                                               variation_rate=0.3)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_restart_variation(self):
        result = restart_variation(self.particle_pos, self.bounds)
        self.assertEqual(result.shape, self.particle_pos.shape)
        self.assertTrue(np.all(result >= self.bounds[0]))
        self.assertTrue(np.all(result <= self.bounds[1]))
        
    def test_adaptive_variation_strength(self):
        result = adaptive_variation_strength(self.particle_pos, self.current_iter, self.max_iter,
                                           base_strength=0.1, bounds=self.bounds)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_opposition_based_variation(self):
        result = opposition_based_variation(self.particle_pos, self.bounds, variation_rate=0.1)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_hybrid_variation(self):
        population = [
            np.array([0.0, 1.0, 2.0]),
            np.array([4.0, 5.0, 6.0])
        ]
        result = hybrid_variation(self.particle_pos, self.current_iter, self.max_iter,
                                 self.bounds, population)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_apply_variation(self):
        # Test applying a known strategy
        result = apply_variation(self.particle_pos, 'gaussian', 
                                variation_rate=0.1, variation_strength=0.1)
        self.assertEqual(result.shape, self.particle_pos.shape)
        
    def test_apply_variation_invalid_strategy(self):
        with self.assertRaises(ValueError):
            apply_variation(self.particle_pos, 'invalid_strategy')
            
    def test_detect_stalled_particles(self):
        # Create mock particles with stagnation_count attribute
        class MockParticle:
            def __init__(self, count):
                self.stagnation_count = count
                
        particles = [MockParticle(5), MockParticle(15), MockParticle(8)]
        stalled = detect_stalled_particles(particles, stagnation_threshold=10)
        self.assertEqual(stalled, [1])  # Only second particle has count >= 10
        
    def test_detect_converged_particles(self):
        class MockParticle:
            def __init__(self, velocity_mag):
                self.velocity = np.array([velocity_mag, 0.0, 0.0])
                
        particles = [MockParticle(1e-5), MockParticle(1e-7), MockParticle(1.0)]
        converged = detect_converged_particles(particles, convergence_threshold=1e-6)
        self.assertEqual(converged, [1])  # Only second particle has velocity magnitude < 1e-6

if __name__ == '__main__':
    unittest.main()
