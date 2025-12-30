import unittest

from context import inertia

class TestInertia(unittest.TestCase):

    def test_constant_inertia_weight(self):
        self.assertEqual(inertia.constant_inertia_weight(2), 2)
        self.assertRaises(TypeError, inertia.constant_inertia_weight)

    def test_random_inertia_weight(self):
        self.assertEqual(inertia.random_inertia_weight(2), 0.9780171359446247)

    def test_chaotic_inertia_weight(self):
        # Test chaotic inertia weight with new signature
        result = inertia.chaotic_inertia_weight(0.3, 10, 1)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.1)
        self.assertLessEqual(result, 1.0)

if __name__ == '__main__':
    unittest.main()
