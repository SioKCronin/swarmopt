# Global Best Particle Swarm Optimization (GBEST_PSO)

Particle Swarm Optimization (PSO) is a stochastic population-based
optimization method. There are two key methods that each handle
information exchange differently, local best and global best, and both
are implemented here. Both methods begin by initializing a swarm of
"particles" (randomly assigned vectors) that will canvas the solutions
landscape across a number of iterations, wich velocities updated between
iterations based on the updating procedure defined by the method.

## Global Best PSO

In Global Best, velocities are updated by comparing personal historical bests
(the positions visited by the particle that yield the most optimal
results) with historical global bests. The weighting of each is
determined by scalar multiples (cognitive factor and social factor)
that are defined during intitialization.

### Mean cost of 20 runs

|Function|  Mean cost |
|--- |---|
|  Sphere | 4.6114e-003|
|  Rosenbrock | 1.0707e+000 |
|  Ackley | 1.3570e-001 |
|  Griewank | 2.0263e-003 |
|  Rastrigin | 9.4401e-001 |
|  Weierstrass | 4.5357e-001 |
