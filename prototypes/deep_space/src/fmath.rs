//! Tiny float helpers usable without `std` or `libm`.

/// Absolute value without relying on `std`.
#[inline]
pub fn abs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Clamp `x` into `[lo, hi]`. Non-finite input maps to `lo` so that a NaN can
/// never propagate into position or velocity state.
#[inline]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if !x.is_finite() {
        return lo;
    }
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Replace a non-finite cost with `+inf` so it always loses comparisons.
#[inline]
pub fn sanitize_cost(c: f32) -> f32 {
    if c.is_nan() {
        f32::INFINITY
    } else {
        c
    }
}
