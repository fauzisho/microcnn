/// SIMD micro-kernels with NEON acceleration and scalar fallbacks.

// ── FP32 AXPY: c[c_off..] += a_val * b[b_off..] ──

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn axpy_f32(c: &mut [f32], c_off: usize, b: &[f32], b_off: usize, a_val: f32, len: usize) {
    use core::arch::aarch64::*;
    let mut j = 0usize;
    unsafe {
        let a_vec = vdupq_n_f32(a_val);
        while j + 4 <= len {
            let b_vec = vld1q_f32(b.as_ptr().add(b_off + j));
            let c_vec = vld1q_f32(c.as_ptr().add(c_off + j));
            let r = vfmaq_f32(c_vec, a_vec, b_vec);
            vst1q_f32(c.as_mut_ptr().add(c_off + j), r);
            j += 4;
        }
    }
    // scalar tail
    while j < len {
        c[c_off + j] += a_val * b[b_off + j];
        j += 1;
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn axpy_f32(c: &mut [f32], c_off: usize, b: &[f32], b_off: usize, a_val: f32, len: usize) {
    for j in 0..len {
        c[c_off + j] += a_val * b[b_off + j];
    }
}

// ── INT8 dot product: sum(a[a_off..] * b[b_off..]) ──
//
// Uses vmull_s8 (8xi8 -> 8xi16) + vpadalq_s16 (pairwise add-accumulate i16 -> i32)
// since vdotq_s32 requires nightly `stdarch_neon_dotprod`.

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn dot_i8(a: &[i8], a_off: usize, b: &[i8], b_off: usize, len: usize) -> i32 {
    use core::arch::aarch64::*;
    let mut j = 0usize;
    unsafe {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);

        // Process 16 i8 elements per iteration (two vmull of 8 each)
        while j + 16 <= len {
            let va = vld1q_s8(a.as_ptr().add(a_off + j));
            let vb = vld1q_s8(b.as_ptr().add(b_off + j));
            // Low halves: 8xi8 * 8xi8 -> 8xi16
            let prod_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            // High halves
            let prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            // Pairwise add i16 pairs into i32 accumulators
            acc0 = vpadalq_s16(acc0, prod_lo);
            acc1 = vpadalq_s16(acc1, prod_hi);
            j += 16;
        }

        // Process 8 elements
        if j + 8 <= len {
            let va = vld1_s8(a.as_ptr().add(a_off + j));
            let vb = vld1_s8(b.as_ptr().add(b_off + j));
            let prod = vmull_s8(va, vb);
            acc0 = vpadalq_s16(acc0, prod);
            j += 8;
        }

        acc0 = vaddq_s32(acc0, acc1);
        let mut sum = vaddvq_s32(acc0);

        // scalar tail
        while j < len {
            sum += a[a_off + j] as i32 * b[b_off + j] as i32;
            j += 1;
        }
        sum
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn dot_i8(a: &[i8], a_off: usize, b: &[i8], b_off: usize, len: usize) -> i32 {
    let mut sum: i32 = 0;
    for j in 0..len {
        sum += a[a_off + j] as i32 * b[b_off + j] as i32;
    }
    sum
}

// ── INT8 row sum: sum(a[a_off..a_off+len]) ──
//
// Uses vpaddlq_s8 (16xi8 -> 8xi16) + vpadalq_s16 (8xi16 -> 4xi32) for stable Rust.

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn sum_i8(a: &[i8], a_off: usize, len: usize) -> i32 {
    use core::arch::aarch64::*;
    let mut j = 0usize;
    unsafe {
        let mut acc = vdupq_n_s32(0);
        while j + 16 <= len {
            let va = vld1q_s8(a.as_ptr().add(a_off + j));
            // 16xi8 -> 8xi16 (pairwise add)
            let wide = vpaddlq_s8(va);
            // 8xi16 -> 4xi32 (pairwise add-accumulate)
            acc = vpadalq_s16(acc, wide);
            j += 16;
        }
        let mut sum = vaddvq_s32(acc);

        while j < len {
            sum += a[a_off + j] as i32;
            j += 1;
        }
        sum
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn sum_i8(a: &[i8], a_off: usize, len: usize) -> i32 {
    let mut sum: i32 = 0;
    for j in 0..len {
        sum += a[a_off + j] as i32;
    }
    sum
}
