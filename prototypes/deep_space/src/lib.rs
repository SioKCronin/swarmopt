//! Deep-space autonomous swarm runtime — research prototype.
//!
//! This crate is the "flight research core" described in
//! `docs/deep_space_mission.md`. It is `no_std`, allocation-free and
//! dependency-free. All state lives in fixed-capacity arrays, every loop has a
//! compile-time or configuration-time bound, and all randomness comes from an
//! explicitly seeded generator so any run can be replayed bit-for-bit.
//!
//! The design separates **proposal** from **approval**: optimizers
//! ([`baseline`], [`planner`] built on [`pso`]) propose actions, and a separate
//! [`safety`] gate approves or rejects them against hard constraints (energy
//! reserve, valid targets, finite values). Nothing an optimizer returns is
//! executed without passing the gate.
//!
//! This is not flight software. Memory safety from Rust does not by itself
//! give real-time determinism, radiation tolerance or flight qualification.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod baseline;
pub mod envelope;
pub mod fmath;
pub mod planner;
pub mod pso;
pub mod rng;
pub mod safety;
pub mod sim;
pub mod state;
