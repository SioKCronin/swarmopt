import unittest
import numpy as np
from swarmopt.utils.velocity_clamping import *

class TestVelocityClamping(unittest.TestCase):
    def setUp(self):
        self.velocity = np.array([1.0, -2.0, 3.0, -4.0])
        self.velocity_bounds = 2.0
        self.current_iter = 50
        self.max_iter = 100
        self.z = 0.5
        
    def test_no_clamping(self):
        result = no_clamping(self.velocity, self.velocity_bounds)
        np.testing.assert_array_equal(result, self.velocity)
        
    def test_basic_clamping(self):
        result = basic_clamping(self.velocity, self.velocity_bounds)
        expected = np.clip(self.velocity, -self.velocity_bounds, self.velocity_bounds)
        np.testing.assert_array_equal(result, expected)
        
    def test_adaptive_clamping(self):
        result = adaptive_clamping(self.velocity, self.velocity_bounds, 
                                  self.current_iter, self.max_iter)
        adaptive_bounds = self.velocity_bounds * (1 - self.current_iter / self.max_iter)
        expected = np.clip(self.velocity, -adaptive_bounds, adaptive_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_exponential_clamping(self):
        result = exponential_clamping(self.velocity, self.velocity_bounds,
                                     self.current_iter, self.max_iter)
        exp_bounds = self.velocity_bounds * np.exp(-2 * self.current_iter / self.max_iter)
        expected = np.clip(self.velocity, -exp_bounds, exp_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_sigmoid_clamping(self):
        result = sigmoid_clamping(self.velocity, self.velocity_bounds,
                                 self.current_iter, self.max_iter)
        x = 2 * self.current_iter / self.max_iter - 1
        sigmoid_bounds = self.velocity_bounds / (1 + np.exp(x))
        expected = np.clip(self.velocity, -sigmoid_bounds, sigmoid_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_random_clamping(self):
        result = random_clamping(self.velocity, self.velocity_bounds)
        # Since it's random, just check bounds are reasonable
        self.assertTrue(np.all(result >= -self.velocity_bounds * 1.5))
        self.assertTrue(np.all(result <= self.velocity_bounds * 1.5))
        
    def test_chaotic_clamping(self):
        result = chaotic_clamping(self.velocity, self.velocity_bounds, self.z)
        # Check that z parameter is used and result is within reasonable bounds
        self.assertTrue(np.all(result >= -self.velocity_bounds))
        self.assertTrue(np.all(result <= self.velocity_bounds))
        
    def test_dimension_wise_clamping_single(self):
        result = dimension_wise_clamping(self.velocity, self.velocity_bounds)
        expected = np.clip(self.velocity, -self.velocity_bounds, self.velocity_bounds)
        np.testing.assert_array_equal(result, expected)
        
    def test_dimension_wise_clamping_array(self):
        bounds_array = np.array([1.0, 2.0, 3.0, 4.0])
        result = dimension_wise_clamping(self.velocity, bounds_array)
        expected = np.clip(self.velocity, -bounds_array, bounds_array)
        np.testing.assert_array_equal(result, expected)
        
    def test_soft_clamping(self):
        result = soft_clamping(self.velocity, self.velocity_bounds)
        expected = self.velocity_bounds * np.tanh(self.velocity / self.velocity_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_hybrid_clamping_first_half(self):
        current_iter = 25
        max_iter = 100
        result = hybrid_clamping(self.velocity, self.velocity_bounds,
                                current_iter, max_iter)
        adaptive_bounds = self.velocity_bounds * (1 - current_iter / max_iter)
        expected = np.clip(self.velocity, -adaptive_bounds, adaptive_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_hybrid_clamping_second_half(self):
        current_iter = 75
        max_iter = 100
        result = hybrid_clamping(self.velocity, self.velocity_bounds,
                                current_iter, max_iter)
        exp_bounds = self.velocity_bounds * np.exp(-2 * current_iter / max_iter)
        expected = np.clip(self.velocity, -exp_bounds, exp_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_convergence_based_clamping(self):
        best_cost = 0.5
        initial_cost = 1.0
        result = convergence_based_clamping(self.velocity, self.velocity_bounds,
                                           best_cost, initial_cost)
        convergence_ratio = (initial_cost - best_cost) / initial_cost
        adaptive_bounds = self.velocity_bounds * (1 - convergence_ratio)
        expected = np.clip(self.velocity, -adaptive_bounds, adaptive_bounds)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_convergence_based_clamping_zero_initial(self):
        best_cost = 0.5
        initial_cost = 0.0
        result = convergence_based_clamping(self.velocity, self.velocity_bounds,
                                           best_cost, initial_cost)
        expected = np.clip(self.velocity, -self.velocity_bounds, self.velocity_bounds)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
