import unittest

from inertia import constant_inertia_weight, random_inertia_weight,\
                    chaotic_inertia_weight

class TestInertia(unittest.TestCase):

    def test_constant_inertia_weight(self):
        self.assertEqual(constant_inertia_weight(2), 2)
        self.assertRaises(TypeError, constant_inertia_weight)

    def test_random_inertia_weight(self):
        self.assertEqual(random_inertia_weight(2), 0.9780171359446247)

    def test_chaotic_inertia_weight(self):
        self.assertEqual(chaotic_inertia_weight(0.3, 0.7, 2, 10, 1), -5.96)

if __name__ == '__main__':
    unittest.main()
