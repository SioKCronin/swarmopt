//! Desktop driver for the deep-space prototype (Stage 1: deterministic
//! simulation). The library itself is `no_std`; this binary uses `std` only
//! for argument parsing, timing and printing.
//!
//! ```text
//! deep-space-sim                      # mission comparison table
//! deep-space-sim mission [SEED]       # same, with a chosen seed
//! deep-space-sim bench FUNC DIMS SEED_FROM SEED_TO [PARTICLES] [EPOCHS]
//!                                     # one best cost per line, for
//!                                     # compare_python.py
//! ```

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use deep_space::pso::{self, functions, PsoConfig};
use deep_space::sim::{self, Faults, SimConfig, Strategy};

fn mission(seed: u64) {
    println!(
        "deep-space swarm prototype — seed {seed}, {} B of fixed simulation state\n",
        sim::state_bytes()
    );
    println!(
        "{:<9} {:<8} {:>8} {:>5} {:>5} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7} {:>8} {:>9}",
        "faults",
        "strategy",
        "science",
        "obs",
        "dup",
        "energy",
        "sent",
        "accept",
        "decErr",
        "replay",
        "safety",
        "minMarg",
        "us/decide"
    );
    for (label, faults) in [("nominal", Faults::NONE), ("harsh", Faults::harsh())] {
        for strategy in [Strategy::Baseline, Strategy::Pso] {
            let mut cfg = SimConfig::nominal(seed, strategy);
            cfg.faults = faults;
            let start = Instant::now();
            let m = sim::run(&cfg);
            let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(m.decisions.max(1));
            println!(
                "{:<9} {:<8} {:>8.1} {:>5} {:>5} {:>7.1} {:>6} {:>6} {:>6} {:>6} {:>7} {:>8.2} {:>9.1}",
                label,
                format!("{strategy:?}"),
                m.unique_science,
                m.observations,
                m.duplicate_observations,
                m.energy_used,
                m.messages_sent,
                m.messages_accepted,
                m.rejected_decode,
                m.rejected_replay,
                m.safety_rejections,
                m.min_reserve_margin,
                us,
            );
        }
    }
    println!("\nminMarg = lowest (energy - reserve) after any action; must be >= 0.");
}

type Objective<const D: usize> = fn(&[f32; D]) -> f32;

fn bench_one<const D: usize>(func: &str, seed: u64, epochs: u32) -> Option<f32> {
    let (lo, hi, f): (f32, f32, Objective<D>) = match func {
        "sphere" => (-5.12, 5.12, functions::sphere::<D>),
        "rosenbrock" => (-5.0, 10.0, functions::rosenbrock::<D>),
        _ => return None,
    };
    let cfg = PsoConfig::standard(lo, hi, epochs);
    pso::minimize::<30, D, _>(&cfg, seed, f)
        .ok()
        .map(|r| r.best_cost)
}

fn bench(args: &[String]) -> Result<(), String> {
    let get = |i: usize| args.get(i).ok_or_else(|| "missing argument".to_string());
    let func = get(0)?.as_str();
    let dims: usize = get(1)?.parse().map_err(|_| "bad DIMS")?;
    let from: u64 = get(2)?.parse().map_err(|_| "bad SEED_FROM")?;
    let to: u64 = get(3)?.parse().map_err(|_| "bad SEED_TO")?;
    let particles: usize = args
        .get(4)
        .map_or(Ok(30), |s| s.parse())
        .map_err(|_| "bad PARTICLES")?;
    let epochs: u32 = args
        .get(5)
        .map_or(Ok(100), |s| s.parse())
        .map_err(|_| "bad EPOCHS")?;
    if particles != 30 {
        return Err("this prototype compiles the bench swarm at 30 particles".into());
    }
    for seed in from..to {
        let r = match dims {
            2 => bench_one::<2>(func, seed, epochs),
            5 => bench_one::<5>(func, seed, epochs),
            10 => bench_one::<10>(func, seed, epochs),
            _ => return Err("DIMS must be 2, 5 or 10 (compile-time sizes)".into()),
        };
        match r {
            Some(c) => println!("{c:e}"),
            None => return Err(format!("unknown function {func}")),
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None => {
            mission(2026);
            Ok(())
        }
        Some("mission") => args
            .get(1)
            .map_or(Ok(2026), |s| s.parse::<u64>())
            .map(mission)
            .map_err(|_| "bad SEED".to_string()),
        Some("bench") => bench(&args[1..]),
        Some(other) => Err(format!("unknown command {other}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
