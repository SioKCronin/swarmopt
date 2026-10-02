import unittest

from context import inertia

class TestInertia(unittest.TestCase):

    def test_constant_inertia_weight(self):
        self.assertEqual(inertia.constant_inertia_weight(2), 2)
        self.assertRaises(TypeError, inertia.constant_inertia_weight)

    def test_random_inertia_weight(self):
        # Same seed, same value; always within [0.5, 1.0]
        self.assertEqual(inertia.random_inertia_weight(2), inertia.random_inertia_weight(2))
        for s in range(20):
            w = inertia.random_inertia_weight(s)
            self.assertGreaterEqual(w, 0.5)
            self.assertLessEqual(w, 1.0)

    def test_random_inertia_does_not_touch_global_state(self):
        import random
        random.seed(123)
        expected = random.random()
        random.seed(123)
        inertia.random_inertia_weight(2)
        self.assertEqual(random.random(), expected)

    def test_chaotic_inertia_weight(self):
        # Test chaotic inertia weight with new signature
        result = inertia.chaotic_inertia_weight(0.3, 10, 1)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.1)
        self.assertLessEqual(result, 1.0)

if __name__ == '__main__':
    unittest.main()
