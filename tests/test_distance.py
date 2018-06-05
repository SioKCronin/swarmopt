import unittest
from context import distance

class TestDistance(unittest.TestCase):
    def test_euclidian(self):
        self.assertEqual(distance.euclideanDistance([1, 2], [2, 3]), 1.0)

