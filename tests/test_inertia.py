import unittest

from context import inertia

class TestInertia(unittest.TestCase):

    def test_constant_inertia_weight(self):
        self.assertEqual(inertia.constant_inertia_weight(2), 2)
        self.assertRaises(TypeError, inertia.constant_inertia_weight)

    def test_random_inertia_weight(self):
        self.assertEqual(inertia.random_inertia_weight(2), 0.9780171359446247)

    def test_chaotic_inertia_weight(self):
        self.assertEqual(inertia.chaotic_inertia_weight(0.3, 0.7, 2, 10, 1), -5.96)

if __name__ == '__main__':
    unittest.main()
