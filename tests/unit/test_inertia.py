import unittest

from context import inertia

class TestInertia(unittest.TestCase):

    def test_constant_inertia_weight(self):
        self.assertEqual(inertia.constant_inertia_weight(2), 2)
        self.assertRaises(TypeError, inertia.constant_inertia_weight)

    def test_random_inertia_weight(self):
        self.assertEqual(inertia.random_inertia_weight(2), 0.9780171359446247)

    def test_chaotic_inertia_weight(self):
        result = inertia.chaotic_inertia_weight(0.3, 10, 1)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.1)
        self.assertLessEqual(result, 1.0)

    def test_linear_inertia_weight_start(self):
        # At iteration 0 the result should equal w_start
        result = inertia.linear_inertia_weight(0.9, 0.4, 100, 0)
        self.assertAlmostEqual(result, 0.9)

    def test_linear_inertia_weight_end(self):
        # At max_iter the result should equal w_end
        result = inertia.linear_inertia_weight(0.9, 0.4, 100, 100)
        self.assertAlmostEqual(result, 0.4)

    def test_linear_inertia_weight_monotone(self):
        # Weight should decrease across iterations
        values = [inertia.linear_inertia_weight(0.9, 0.4, 100, t) for t in range(0, 101, 10)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_exponential_inertia_weight_bounds(self):
        # At t=0 result should be close to w_start; at t=max_iter close to w_end
        w_start, w_end = 0.9, 0.4
        result_start = inertia.exponential_inertia_weight(w_start, w_end, 100, 0)
        result_end = inertia.exponential_inertia_weight(w_start, w_end, 100, 100)
        self.assertAlmostEqual(result_start, w_start, places=5)
        self.assertAlmostEqual(result_end, w_end, places=1)

    def test_exponential_inertia_weight_monotone(self):
        values = [inertia.exponential_inertia_weight(0.9, 0.4, 100, t) for t in range(0, 101, 10)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_sigmoid_inertia_weight_bounds(self):
        w_start, w_end = 0.9, 0.4
        for t in range(0, 101, 25):
            result = inertia.sigmoid_inertia_weight(w_start, w_end, 100, t)
            self.assertGreaterEqual(result, w_end - 1e-9)
            self.assertLessEqual(result, w_start + 1e-9)

    def test_sigmoid_inertia_weight_monotone(self):
        values = [inertia.sigmoid_inertia_weight(0.9, 0.4, 100, t) for t in range(0, 101, 10)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_adaptive_inertia_weight_no_convergence(self):
        # When convergence_ratio is 0 (initial == current cost) factor = 1 → equals linear
        linear = inertia.linear_inertia_weight(0.9, 0.4, 100, 50)
        adaptive = inertia.adaptive_inertia_weight(0.9, 0.4, 100, 50, None, 100.0, 100.0)
        self.assertAlmostEqual(adaptive, linear)

    def test_adaptive_inertia_weight_zero_initial(self):
        # When initial_cost == 0 the function falls back to the linear component
        linear = inertia.linear_inertia_weight(0.9, 0.4, 100, 50)
        result = inertia.adaptive_inertia_weight(0.9, 0.4, 100, 50, None, 0, 50.0)
        self.assertAlmostEqual(result, linear)

    def test_adaptive_inertia_weight_full_convergence(self):
        # When cost has dropped to zero the adaptive factor drives weight toward 0
        result = inertia.adaptive_inertia_weight(0.9, 0.4, 100, 50, None, 100.0, 0.0)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_chaotic_random_inertia_weight_range(self):
        import random
        random.seed(42)
        for z in [0.1, 0.3, 0.5, 0.7]:
            result = inertia.chaotic_random_inertia_weight(z)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_chaotic_random_inertia_weight_seeded(self):
        # Same seed should produce the same result
        r1 = inertia.chaotic_random_inertia_weight(0.3, s=7)
        r2 = inertia.chaotic_random_inertia_weight(0.3, s=7)
        self.assertAlmostEqual(r1, r2)

if __name__ == '__main__':
    unittest.main()
