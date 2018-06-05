import unittest
from context import distance

class TestDistance(unittest.TestCase):
    def test_euclidian(self):
        self.assertEqual(distance.euclideanDistance([0, 0], [2, 0]), 2.0)
        self.assertEqual(distance.euclideanDistance([0, 0], [0, 2]), 2.0)
        self.assertEqual(distance.euclideanDistance([1, 1], [4, 5]), 5.0)
        self.assertEqual(distance.euclideanDistance([1], [4]), 3.0)

