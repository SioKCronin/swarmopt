import os
import sys

# Ensure project root is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from swarmopt.swarm import Swarm, Particle
from swarmopt import functions
from swarmopt.utils import distance, inertia
