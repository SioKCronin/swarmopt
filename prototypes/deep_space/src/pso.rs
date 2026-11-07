//! Bounded global-best particle swarm optimization.
//!
//! Mirrors the update rule of `swarmopt.Swarm(algo='global',
//! inertia_func='linear')` — `v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`
//! with per-dimension uniform `r1, r2` and inertia decaying linearly from
//! `w_start` to `w_end` — but with flight-oriented constraints:
//!
//! * swarm size `P` and dimension `D` are compile-time constants, so all state
//!   is stack-allocated and its size is known at build time;
//! * the iteration count is capped by [`MAX_ITERATIONS`], so the number of
//!   objective evaluations is exactly `P · (iterations + 1)`;
//! * velocities are clamped to `±v_max` and positions to `[lower, upper]`
//!   (the Python library does not clamp positions);
//! * a non-finite objective value is treated as `+inf` and can never become
//!   the global best.

use crate::fmath::{clamp, sanitize_cost};
use crate::rng::SplitMix64;

/// Hard ceiling on iterations regardless of configuration.
pub const MAX_ITERATIONS: u32 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsoConfig {
    pub w_start: f32,
    pub w_end: f32,
    pub c1: f32,
    pub c2: f32,
    pub lower: f32,
    pub upper: f32,
    pub v_max: f32,
    pub iterations: u32,
}

impl PsoConfig {
    /// Defaults matching SwarmOpt's README quick start (`c1 = c2 = 2.0`,
    /// linear inertia 0.9 → 0.4) and its velocity bound of 20% of the range.
    pub fn standard(lower: f32, upper: f32, iterations: u32) -> Self {
        Self {
            w_start: 0.9,
            w_end: 0.4,
            c1: 2.0,
            c2: 2.0,
            lower,
            upper,
            v_max: 0.2 * (upper - lower),
            iterations,
        }
    }

    pub fn validate(&self) -> Result<(), PsoError> {
        let all_finite = [
            self.w_start,
            self.w_end,
            self.c1,
            self.c2,
            self.lower,
            self.upper,
            self.v_max,
        ]
        .iter()
        .all(|x| x.is_finite());
        if !all_finite {
            return Err(PsoError::NonFiniteParameter);
        }
        if self.lower >= self.upper {
            return Err(PsoError::EmptyBounds);
        }
        if self.v_max <= 0.0 || self.c1 < 0.0 || self.c2 < 0.0 {
            return Err(PsoError::InvalidParameter);
        }
        if self.iterations > MAX_ITERATIONS {
            return Err(PsoError::IterationBudgetExceeded);
        }
        Ok(())
    }

    fn inertia(&self, iter: u32) -> f32 {
        if self.iterations == 0 {
            return self.w_start;
        }
        let t = iter as f32 / self.iterations as f32;
        self.w_start + (self.w_end - self.w_start) * t
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PsoError {
    NonFiniteParameter,
    EmptyBounds,
    InvalidParameter,
    IterationBudgetExceeded,
    EmptySwarm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsoResult<const D: usize> {
    pub best_position: [f32; D],
    pub best_cost: f32,
    pub evaluations: u32,
}

/// Minimize `f` over `[lower, upper]^D` with a `P`-particle swarm.
pub fn minimize<const P: usize, const D: usize, F>(
    cfg: &PsoConfig,
    seed: u64,
    mut f: F,
) -> Result<PsoResult<D>, PsoError>
where
    F: FnMut(&[f32; D]) -> f32,
{
    cfg.validate()?;
    if P == 0 || D == 0 {
        return Err(PsoError::EmptySwarm);
    }

    let mut rng = SplitMix64::new(seed);
    let mut pos = [[0.0f32; D]; P];
    let mut vel = [[0.0f32; D]; P];
    let mut pbest = [[0.0f32; D]; P];
    let mut pbest_cost = [f32::INFINITY; P];
    let mut gbest = [0.0f32; D];
    let mut gbest_cost = f32::INFINITY;
    let mut evaluations: u32 = 0;

    for i in 0..P {
        for d in 0..D {
            pos[i][d] = rng.range(cfg.lower, cfg.upper);
            vel[i][d] = rng.range(-cfg.v_max, cfg.v_max);
        }
        let c = sanitize_cost(f(&pos[i]));
        evaluations += 1;
        pbest[i] = pos[i];
        pbest_cost[i] = c;
        if c < gbest_cost {
            gbest_cost = c;
            gbest = pos[i];
        }
    }
    // If every initial evaluation was non-finite, gbest stays at a valid
    // in-bounds point rather than the zero vector.
    if !gbest_cost.is_finite() {
        gbest = pos[0];
    }

    for iter in 0..cfg.iterations {
        let w = cfg.inertia(iter);
        for i in 0..P {
            for d in 0..D {
                let r1 = rng.next_f32();
                let r2 = rng.next_f32();
                let v = w * vel[i][d]
                    + cfg.c1 * r1 * (pbest[i][d] - pos[i][d])
                    + cfg.c2 * r2 * (gbest[d] - pos[i][d]);
                vel[i][d] = clamp(v, -cfg.v_max, cfg.v_max);
                let x = pos[i][d] + vel[i][d];
                let xc = clamp(x, cfg.lower, cfg.upper);
                if xc != x {
                    // Absorbing wall: stop at the boundary.
                    vel[i][d] = 0.0;
                }
                pos[i][d] = xc;
            }
            let c = sanitize_cost(f(&pos[i]));
            evaluations += 1;
            if c < pbest_cost[i] {
                pbest_cost[i] = c;
                pbest[i] = pos[i];
                if c < gbest_cost {
                    gbest_cost = c;
                    gbest = pos[i];
                }
            }
        }
    }

    Ok(PsoResult {
        best_position: gbest,
        best_cost: gbest_cost,
        evaluations,
    })
}

/// Benchmark objectives shared with the Python comparison script.
pub mod functions {
    pub fn sphere<const D: usize>(x: &[f32; D]) -> f32 {
        x.iter().map(|v| v * v).sum()
    }

    pub fn rosenbrock<const D: usize>(x: &[f32; D]) -> f32 {
        let mut s = 0.0;
        for i in 0..D.saturating_sub(1) {
            let a = x[i] * x[i] - x[i + 1];
            let b = x[i] - 1.0;
            s += 100.0 * a * a + b * b;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_sphere() {
        let cfg = PsoConfig::standard(-5.12, 5.12, 200);
        let r = minimize::<30, 5, _>(&cfg, 42, functions::sphere).unwrap();
        assert!(r.best_cost < 1e-3, "best_cost = {}", r.best_cost);
    }

    #[test]
    fn evaluation_count_is_exact() {
        let cfg = PsoConfig::standard(-1.0, 1.0, 17);
        let r = minimize::<9, 3, _>(&cfg, 0, functions::sphere).unwrap();
        assert_eq!(r.evaluations, 9 * (17 + 1));
    }

    #[test]
    fn positions_stay_in_bounds() {
        let cfg = PsoConfig::standard(2.0, 3.0, 50);
        let mut ok = true;
        let _ = minimize::<10, 4, _>(&cfg, 3, |x| {
            ok &= x.iter().all(|v| (2.0..=3.0).contains(v));
            functions::sphere(x)
        })
        .unwrap();
        assert!(ok);
        // Optimum of sphere on [2,3]^4 is the corner (2,2,2,2).
    }

    #[test]
    fn nan_objective_never_wins() {
        let cfg = PsoConfig::standard(-1.0, 1.0, 20);
        let r = minimize::<10, 2, _>(&cfg, 5, |x| {
            if x[0] > 0.0 {
                f32::NAN
            } else {
                functions::sphere(x)
            }
        })
        .unwrap();
        assert!(r.best_cost.is_finite());
        assert!(r.best_position[0] <= 0.0);
    }

    #[test]
    fn rejects_bad_config() {
        let mut cfg = PsoConfig::standard(-1.0, 1.0, 10);
        cfg.iterations = MAX_ITERATIONS + 1;
        assert_eq!(
            minimize::<4, 2, _>(&cfg, 0, functions::sphere).unwrap_err(),
            PsoError::IterationBudgetExceeded
        );
        let cfg = PsoConfig::standard(1.0, 1.0, 10);
        assert_eq!(cfg.validate(), Err(PsoError::EmptyBounds));
        let mut cfg = PsoConfig::standard(-1.0, 1.0, 10);
        cfg.c1 = f32::NAN;
        assert_eq!(cfg.validate(), Err(PsoError::NonFiniteParameter));
    }

    #[test]
    fn deterministic_replay() {
        let cfg = PsoConfig::standard(-5.0, 10.0, 100);
        let a = minimize::<20, 4, _>(&cfg, 99, functions::rosenbrock).unwrap();
        let b = minimize::<20, 4, _>(&cfg, 99, functions::rosenbrock).unwrap();
        assert_eq!(a, b);
    }
}
