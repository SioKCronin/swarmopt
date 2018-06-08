"""Distance functions"""
import math
from functools import reduce

def euclideanDistance(point1, point2):
    return math.sqrt(
        reduce(
            (lambda memo, pair: memo + (pair[1] - pair[0]) ** 2), zip(point1, point2), 0
        )
    )



