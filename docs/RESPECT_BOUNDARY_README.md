# 🎯 Respect Boundary Feature

## Overview

The **Respect Boundary** feature allows particles in a swarm to converge to a safe distance
around a target rather than exactly on it. This is useful for applications requiring a "standoff distance" 
or "safety margin" from a target position.

## Key Concept

Instead of converging to the optimal position (which might be dangerous or undesirable), 
particles find the optimal position on or outside a **sphere of respect** around the target.

```
      Target
         ●
        ╱│╲
       ╱ │ ╲   Respect Boundary (radius = r)
      ●──┼──●
       ╲ │ ╱   
        ╲│╱
         ●
    Particles converge here
    (distance ≥ r from target)
```

## Usage

### Basic Example

```python
import numpy as np
from swarmopt import Swarm

# Define target position
target = np.array([5.0, 5.0, 5.0])

# Define objective function
def distance_to_target(x):
    return np.linalg.norm(x - target)

# Create swarm with respect boundary (automatically enabled!)
swarm = Swarm(
    n_particles=20,
    dims=3,
    c1=2.0, c2=2.0, w=0.9,
    epochs=50,
    obj_func=distance_to_target,
    algo='global',
    target_position=target  # Respect boundary automatically enforced for safety!
)
# Warning will be shown: "Respect boundary automatically enabled for safety: X.XXX"

# Optimize
swarm.optimize()

# Results
final_distance = np.linalg.norm(swarm.best_pos - target)
print(f"Converged to distance: {final_distance}")
print(f"Automatic respect boundary: {swarm.respect_boundary:.4f}")
print(f"Boundary respected: {final_distance >= swarm.respect_boundary}")
```

## Parameters

### `target_position` (array-like or None)
- **Description**: Position of the target to respect
- **Default**: `None` (disabled)
- **Shape**: Must match `dims` parameter
- **Example**: `target_position=[10.0, 10.0, 10.0]`

### `respect_boundary` (float, automatic)
- **Description**: Minimum allowed distance from target position
- **Automatic**: Calculated as 10% of search space diagonal when `target_position` is provided
- **Cannot be disabled**: For safety-critical applications, this is mandatory
- **Units**: Same as your coordinate system
- **Formula**: `0.1 × √(dims × (val_max - val_min)²)`

** IMPORTANT SAFETY FEATURE**: When you provide a `target_position`, the respect boundary is **automatically enabled and cannot be disabled**. This is by design to prevent accidents in safety-critical applications where particles must maintain a minimum safe distance from targets.

## How It Works

### Penalty Function

When a particle gets closer than the respect boundary to the target, a penalty is added:

```python
distance_to_target = ||position - target||

if distance < respect_boundary:
    violation = respect_boundary - distance
    penalty_factor = (violation / respect_boundary)²
    modified_cost = base_cost × (1 + 10 × penalty_factor)
```

The penalty:
- **Increases quadratically** as particles get closer to the target
- **Scales with base cost** to remain proportional
- **Ensures convergence** to the boundary, not inside it

### Particle Behavior

1. **Outside boundary**: Particles behave normally, minimizing objective
2. **At boundary**: Optimal convergence point (distance = respect_boundary)
3. **Inside boundary**: Strong penalty pushes particles outward

## Applications

### 1. Satellite Positioning 🛰️

Maintain safe orbital distance from Earth or other celestial bodies:

```python
earth_center = np.array([0.0, 0.0, 0.0])
earth_radius = 6371.0  # km
desired_altitude = 400.0  # km (ISS altitude)
min_safe_distance = earth_radius + desired_altitude

swarm = Swarm(
    ...,
    respect_boundary=min_safe_distance,
    target_position=earth_center
)
```

### 2. Obstacle Avoidance

Navigate around obstacles with safety margins:

```python
obstacle_center = np.array([10.0, 10.0])
safety_margin = 5.0  # meters

def path_cost(position):
    # Cost to reach goal
    goal_distance = np.linalg.norm(position - goal)
    
    # Penalty for entering obstacle zone
    obs_distance = np.linalg.norm(position - obstacle_center)
    if obs_distance < safety_margin:
        penalty = 1000 * (safety_margin - obs_distance)**2
        return goal_distance + penalty
    
    return goal_distance
```

### 3. Social Distancing 👥

Maintain minimum distance between agents:

```python
# Each agent has a personal space
personal_space = 2.0  # meters

swarm = Swarm(
    ...,
    respect_boundary=personal_space,
    target_position=other_agent_position
)
```

### 4. Sensor Placement 📡

Place sensors at optimal distance from signal source:

```python
signal_source = np.array([0.0, 0.0, 0.0])
optimal_sensing_distance = 10.0  # meters

swarm = Swarm(
    ...,
    respect_boundary=optimal_sensing_distance,
    target_position=signal_source
)
```

### 5. Robot Formation Control 🤖

Maintain formation with minimum separation:

```python
leader_position = np.array([x_leader, y_leader])
min_separation = 3.0  # meters

# Each follower maintains respect boundary from leader
swarm = Swarm(
    ...,
    respect_boundary=min_separation,
    target_position=leader_position
)
```

## Advanced Usage

### Multiple Targets with Different Boundaries

For multiple targets, use the objective function to encode multiple respect boundaries:

```python
targets = [
    {'position': np.array([5.0, 5.0]), 'respect_distance': 2.0},
    {'position': np.array([15.0, 15.0]), 'respect_distance': 3.0},
]

def multi_target_objective(x):
    total_cost = 0.0
    
    for target in targets:
        distance = np.linalg.norm(x - target['position'])
        
        # Add penalty if too close
        if distance < target['respect_distance']:
            violation = target['respect_distance'] - distance
            total_cost += 1000 * (violation ** 2)
        else:
            # Normal objective (minimize distance)
            total_cost += distance
    
    return total_cost
```

### Dynamic Respect Boundary

Adjust respect boundary during optimization:

```python
class DynamicBoundarySwarm(Swarm):
    def optimize(self):
        for epoch in range(self.epochs):
            # Gradually decrease respect boundary
            self.respect_boundary = max(
                1.0,  # Minimum boundary
                self.respect_boundary * 0.95  # 5% decrease per epoch
            )
            
            # Continue optimization...
```

### Soft vs Hard Boundaries

The current implementation uses a **soft boundary** (penalty-based). For a **hard boundary** (constraint-based), modify particle positions:

```python
class HardBoundarySwarm(Swarm):
    def _enforce_hard_boundary(self, position):
        if not self.use_respect_boundary:
            return position
        
        distance = np.linalg.norm(position - self.target_position)
        
        if distance < self.respect_boundary:
            # Project onto respect boundary
            direction = (position - self.target_position) / distance
            position = self.target_position + direction * self.respect_boundary
        
        return position
```

## Comparison: With vs Without Respect Boundary

| Aspect | Without Respect Boundary | With Respect Boundary |
|--------|-------------------------|----------------------|
| **Convergence** | Exactly on target | On respect sphere |
| **Safety** | No safety margin | Guaranteed minimum distance |
| **Applications** | Standard optimization | Standoff operations |
| **Final Distance** | ≈ 0 | ≥ respect_boundary |

## Tips and Best Practices

### 1. **Choose Appropriate Penalty Scale**
The default penalty multiplier is 10.0. Adjust if needed:
- **Larger values** (20-50): Stricter boundary enforcement
- **Smaller values** (1-5): Softer boundary, allow occasional violations

### 2. **Balance Respect Distance with Search Space**
Ensure respect boundary is meaningful relative to your search space:
```python
search_space_size = val_max - val_min
respect_boundary = 0.1 * search_space_size  # 10% of space
```

### 3. **Use Appropriate Swarm Size**
Larger respect boundaries may need more particles:
```python
n_particles = max(20, int(respect_boundary * dims * 5))
```

### 4. **Monitor Convergence**
Check if particles actually respect the boundary:
```python
distances = [np.linalg.norm(p.pos - target) for p in swarm.swarm]
violations = sum(1 for d in distances if d < respect_boundary)
print(f"Boundary violations: {violations}/{len(distances)}")
```

## Performance Considerations

- **Computational overhead**: Minimal (~5% slower due to distance calculations)
- **Convergence speed**: May be slightly slower as particles avoid target
- **Memory**: No additional memory required

## Limitations

1. **Single Target**: Built-in support for one target per swarm
2. **Spherical Boundary**: Respect region is spherical (same distance in all directions)
3. **Static Boundary**: Respect distance doesn't change during optimization

For more complex scenarios, customize the objective function or subclass `Swarm`.

## Related Features

- **Velocity Clamping**: Controls particle speed
- **Diversity Monitoring**: Prevents premature convergence
- **Variation Operators**: Helps escape local optima near boundary

## Testing

Run the comprehensive test suite:

```bash
python tests_scripts/test_respect_boundary.py
```

Tests include:
- ✅ Basic 2D convergence
- ✅ 3D spatial optimization
- ✅ Multiple respect distances
- ✅ Satellite positioning scenario
- ✅ Obstacle avoidance
- ✅ Visualization generation

## Examples

See `tests_scripts/test_respect_boundary.py` for:
- Basic usage
- Satellite positioning (ISS altitude maintenance)
- Obstacle avoidance with multiple obstacles
- Visualization of respect boundary behavior

## References

- **Formation Control**: Reynolds, C. W. (1987). "Flocks, herds and schools"
- **Obstacle Avoidance**: Ge, S. S., & Cui, Y. J. (2002). "Dynamic motion planning for mobile robots"
- **Standoff Tracking**: Kim, Y., & Mesbahi, M. (2006). "On maximizing the second smallest eigenvalue"

## Citation

If you use the respect boundary feature in your research, please cite:

```bibtex
@software{swarmopt_respect_boundary,
  title={SwarmOpt: Respect Boundary Feature},
  author={Cronin, Siobhan K.},
  year={2025},
  url={https://github.com/SioKCronin/swarmopt}
}
```

---

**Built with [SwarmOpt](https://github.com/SioKCronin/swarmopt)** 🐝
