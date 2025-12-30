"""Test suite for all benchmark functions
Tests all 30 functions from the benchmark collection
"""

import unittest
import numpy as np
import math
from context import functions
from swarmopt.functions import (
    FUNCTION_METADATA, get_function_metadata, get_optimal_position
)


class TestFunctionMetadata(unittest.TestCase):
    """Test that metadata is available for all functions"""
    
    def test_metadata_exists(self):
        """Test that metadata exists for all functions"""
        func_names = [
            'sphere', 'sum_squares', 'rotated_hyper_ellipsoid', 'zakharov',
            'dixon_price', 'powell', 'rosenbrock', 'ackley', 'griewank',
            'rastrigin', 'schwefel', 'levy', 'michalewicz', 'perm', 'trid',
            'weierstrass', 'de_jong_f5', 'beale', 'booth', 'matyas',
            'three_hump_camel', 'six_hump_camel', 'easom', 'goldstein_price',
            'branin', 'shubert', 'hartmann_3d', 'hartmann_6d', 'shekel'
        ]
        
        for func_name in func_names:
            meta = get_function_metadata(func_name)
            self.assertIsNotNone(meta, f"Metadata missing for {func_name}")
            self.assertIn('optimal_value', meta)
            self.assertIn('bounds', meta)
            self.assertIn('dimensions', meta)
            self.assertIn('type', meta)


class TestUnimodalFunctions(unittest.TestCase):
    """Test unimodal functions"""
    
    def test_sphere(self):
        """Test sphere function"""
        x = [0, 0, 0]
        self.assertEqual(functions.sphere(x), 0.0)
        
        x = [1, 2, 3]
        self.assertEqual(functions.sphere(x), 14.0)
        
        # Test at optimal position
        opt_pos = get_optimal_position('sphere', 3)
        opt_val = functions.sphere(opt_pos)
        self.assertAlmostEqual(opt_val, 0.0, places=10)
    
    def test_sum_squares(self):
        """Test sum squares function"""
        x = [0, 0, 0]
        self.assertEqual(functions.sum_squares(x), 0.0)
        
        x = [1, 2, 3]
        expected = 1*1 + 2*4 + 3*9  # 1 + 8 + 27 = 36
        self.assertEqual(functions.sum_squares(x), expected)
    
    def test_rotated_hyper_ellipsoid(self):
        """Test rotated hyper-ellipsoid function"""
        x = [0, 0, 0]
        self.assertEqual(functions.rotated_hyper_ellipsoid(x), 0.0)
        
        x = [1, 2, 3]
        # x[0]^2 + (x[0]^2 + x[1]^2) + (x[0]^2 + x[1]^2 + x[2]^2)
        expected = 1 + (1 + 4) + (1 + 4 + 9)  # 1 + 5 + 14 = 20
        self.assertEqual(functions.rotated_hyper_ellipsoid(x), expected)
    
    def test_zakharov(self):
        """Test zakharov function"""
        x = [0, 0, 0]
        self.assertEqual(functions.zakharov(x), 0.0)
        
        x = [1, 1]
        # sum1 = 2, sum2 = 0.5*1*1 + 0.5*2*1 = 1.5
        # result = 2 + 1.5^2 + 1.5^4 = 2 + 2.25 + 5.0625 = 9.3125
        result = functions.zakharov(x)
        self.assertAlmostEqual(result, 9.3125, places=5)
    
    def test_dixon_price(self):
        """Test dixon-price function"""
        # Test with optimal position for 2D
        x = get_optimal_position('dixon_price', 2)
        if x:
            result = functions.dixon_price(x)
            self.assertLess(result, 1e-6)  # Should be very close to 0
    
    def test_powell(self):
        """Test powell function"""
        x = [0, 0, 0, 0]
        self.assertEqual(functions.powell(x), 0.0)
        
        # Test dimension requirement
        with self.assertRaises(ValueError):
            functions.powell([1, 2, 3])


class TestMultimodalFunctions(unittest.TestCase):
    """Test multimodal functions"""
    
    def test_rosenbrock(self):
        """Test rosenbrock function"""
        x = [1, 1, 1]
        self.assertAlmostEqual(functions.rosenbrock(x), 0.0, places=10)
        
        x = [0, 0]
        result = functions.rosenbrock(x)
        self.assertGreater(result, 0)
    
    def test_ackley(self):
        """Test ackley function"""
        x = [0, 0, 0]
        result = functions.ackley(x)
        self.assertAlmostEqual(result, 0.0, places=10)
        
        x = [1, 1]
        result = functions.ackley(x)
        self.assertGreater(result, 0)
    
    def test_griewank(self):
        """Test griewank function"""
        x = [0, 0, 0]
        result = functions.griewank(x)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_rastrigin(self):
        """Test rastrigin function"""
        x = [0, 0, 0]
        result = functions.rastrigin(x)
        self.assertAlmostEqual(result, 0.0, places=10)
        
        # At origin should be 10*n
        x = [0, 0]
        result = functions.rastrigin(x)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_schwefel(self):
        """Test schwefel function"""
        # Test with approximate optimal position
        x = [420.9687] * 3
        result = functions.schwefel(x)
        self.assertLess(result, 1.0)  # Should be close to 0
    
    def test_levy(self):
        """Test levy function"""
        x = [1, 1, 1]
        result = functions.levy(x)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_michalewicz(self):
        """Test michalewicz function"""
        x = [2.2, 1.57]  # Known good values for 2D
        result = functions.michalewicz(x, m=10)
        self.assertIsInstance(result, (int, float))
        self.assertLess(result, 0)  # Should be negative
    
    def test_perm(self):
        """Test perm function"""
        x = get_optimal_position('perm', 3)
        if x:
            result = functions.perm(x)
            self.assertLess(result, 1e-6)
    
    def test_trid(self):
        """Test trid function"""
        x = get_optimal_position('trid', 3)
        if x:
            result = functions.trid(x)
            # For 3D, optimal value is -3(3+4)(3-1)/6 = -3*7*2/6 = -7
            self.assertAlmostEqual(result, -7.0, places=5)
    
    def test_weierstrass(self):
        """Test weierstrass function"""
        x = [0, 0, 0]
        result = functions.weierstrass(x)
        self.assertAlmostEqual(result, 0.0, places=5)


class TestLowDimensionalFunctions(unittest.TestCase):
    """Test low-dimensional functions (2D-6D)"""
    
    def test_beale(self):
        """Test beale function (2D)"""
        x = [3, 0.5]
        result = functions.beale(x)
        self.assertAlmostEqual(result, 0.0, places=5)
        
        with self.assertRaises(ValueError):
            functions.beale([1, 2, 3])
    
    def test_booth(self):
        """Test booth function (2D)"""
        x = [1, 3]
        result = functions.booth(x)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_matyas(self):
        """Test matyas function (2D)"""
        x = [0, 0]
        result = functions.matyas(x)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_three_hump_camel(self):
        """Test three-hump camel function (2D)"""
        x = [0, 0]
        result = functions.three_hump_camel(x)
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_six_hump_camel(self):
        """Test six-hump camel function (2D)"""
        # Test one of the optimal positions
        x = [0.0898, -0.7126]
        result = functions.six_hump_camel(x)
        self.assertAlmostEqual(result, -1.0316, places=3)
    
    def test_easom(self):
        """Test easom function (2D)"""
        x = [math.pi, math.pi]
        result = functions.easom(x)
        self.assertAlmostEqual(result, -1.0, places=5)
    
    def test_goldstein_price(self):
        """Test goldstein-price function (2D)"""
        x = [0, -1]
        result = functions.goldstein_price(x)
        self.assertAlmostEqual(result, 3.0, places=1)
    
    def test_branin(self):
        """Test branin function (2D)"""
        # Test one of the optimal positions
        x = [-math.pi, 12.275]
        result = functions.branin(x)
        self.assertAlmostEqual(result, 0.397887, places=3)
    
    def test_shubert(self):
        """Test shubert function (2D)"""
        # Known to have multiple global minima
        result = functions.shubert([0, 0])
        self.assertIsInstance(result, (int, float))
        self.assertLess(result, 0)
    
    def test_hartmann_3d(self):
        """Test hartmann 3D function"""
        x = [0.114614, 0.555649, 0.852547]
        result = functions.hartmann_3d(x)
        self.assertAlmostEqual(result, -3.86278, places=3)
        
        with self.assertRaises(ValueError):
            functions.hartmann_3d([1, 2])
    
    def test_hartmann_6d(self):
        """Test hartmann 6D function"""
        x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
        result = functions.hartmann_6d(x)
        self.assertAlmostEqual(result, -3.32237, places=3)
        
        with self.assertRaises(ValueError):
            functions.hartmann_6d([1, 2, 3])
    
    def test_de_jong_f5(self):
        """Test de jong F5 function (2D)"""
        x = [-32, -32]
        result = functions.de_jong_f5(x)
        self.assertAlmostEqual(result, 0.998, places=2)
        
        with self.assertRaises(ValueError):
            functions.de_jong_f5([1, 2, 3])
    
    def test_shekel(self):
        """Test shekel function (4D)"""
        x = [4, 4, 4, 4]
        result = functions.shekel(x)
        self.assertIsInstance(result, (int, float))
        self.assertLess(result, 0)  # Should be negative
        
        with self.assertRaises(ValueError):
            functions.shekel([1, 2, 3])


class TestFunctionProperties(unittest.TestCase):
    """Test general properties of functions"""
    
    def test_all_functions_return_numeric(self):
        """Test that all functions return numeric values"""
        test_point = [1.0, 2.0, 3.0]
        test_point_2d = [1.0, 2.0]
        
        # Test functions that work with any dimension
        funcs_any_dim = [
            functions.sphere, functions.sum_squares,
            functions.rotated_hyper_ellipsoid, functions.zakharov,
            functions.dixon_price, functions.rosenbrock, functions.ackley,
            functions.griewank, functions.rastrigin, functions.schwefel,
            functions.levy, functions.michalewicz, functions.perm,
            functions.trid, functions.weierstrass
        ]
        
        for func in funcs_any_dim:
            if func == functions.powell:
                result = func([0, 0, 0, 0])
            else:
                result = func(test_point)
            self.assertIsInstance(result, (int, float, np.number))
        
        # Test 2D functions
        funcs_2d = [
            functions.beale, functions.booth, functions.matyas,
            functions.three_hump_camel, functions.six_hump_camel,
            functions.easom, functions.goldstein_price, functions.branin,
            functions.shubert, functions.de_jong_f5
        ]
        
        for func in funcs_2d:
            result = func(test_point_2d)
            self.assertIsInstance(result, (int, float, np.number))
        
        # Test fixed dimension functions
        result = functions.hartmann_3d([0.5, 0.5, 0.5])
        self.assertIsInstance(result, (int, float, np.number))
        
        result = functions.hartmann_6d([0.5] * 6)
        self.assertIsInstance(result, (int, float, np.number))
        
        result = functions.shekel([5, 5, 5, 5])
        self.assertIsInstance(result, (int, float, np.number))
    
    def test_functions_handle_zero_vector(self):
        """Test that functions handle zero vector appropriately"""
        zero_vec = [0.0] * 5
        
        # Functions where zero should give optimal value
        optimal_at_zero = [
            functions.sphere, functions.sum_squares,
            functions.rotated_hyper_ellipsoid, functions.zakharov,
            functions.ackley, functions.griewank, functions.rastrigin,
            functions.weierstrass
        ]
        
        for func in optimal_at_zero:
            if func == functions.powell:
                result = func([0, 0, 0, 0])
            else:
                result = func(zero_vec)
            self.assertIsInstance(result, (int, float, np.number))
            # Most should be at or near optimal
            if func in [functions.sphere, functions.sum_squares,
                       functions.rotated_hyper_ellipsoid, functions.zakharov]:
                self.assertEqual(result, 0.0)


class TestFunctionBounds(unittest.TestCase):
    """Test that functions work within their specified bounds"""
    
    def test_functions_within_bounds(self):
        """Test functions with points within typical bounds"""
        # Test with points within reasonable bounds
        test_points = {
            'sphere': [0.5, -0.3, 1.2],
            'ackley': [1.0, -1.0, 0.5],
            'rastrigin': [0.5, -0.5, 1.0],
            'rosenbrock': [1.0, 1.0, 1.0],
        }
        
        for func_name, point in test_points.items():
            func = getattr(functions, func_name)
            result = func(point)
            self.assertIsInstance(result, (int, float, np.number))
            self.assertFalse(np.isnan(result))
            self.assertFalse(np.isinf(result))


if __name__ == "__main__":
    unittest.main()

