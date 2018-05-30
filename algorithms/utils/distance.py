# Distance utilities

import math

def euclideanDistance(point1, point2):
    distance = 0
    for x in range(len(point1)-1):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(target, swarm, k):
    distances = []
    def takeSecond(elem):
        return elem[1]
    for x in range(len(swarm)-1):
        distances.append((swarm[x][0], euclideanDistance(swarm[x][0], target)))
    sorted_distances = sorted(distances,key=takeSecond)
    neighbors = []
    for x in range(k):
        neighbors.append(sorted_distances[x][0])
    return neighbors
