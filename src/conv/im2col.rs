/// Im2col + tiled GEMM convolution.
///
/// Unfolds input patches into a column matrix, then performs a tiled matrix multiply
/// for cache-friendly access patterns.

use super::simd;

const TILE: usize = 32;

/// Unfold input patches into a column matrix.
///
/// Output shape: rows = `in_channels * kernel_size * kernel_size`, cols = `out_h * out_w`
fn im2col(
    input: &[f32],
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_size: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let col_cols = out_h * out_w;
    for ic in 0..in_channels {
        for kh in 0..kernel_size {
            for kw in 0..kernel_size {
                let row = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                let row_off = row * col_cols;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = oh * stride + kh;
                        let iw = ow * stride + kw;
                        col[row_off + oh * out_w + ow] =
                            input[ic * in_h * in_w + ih * in_w + iw];
                    }
                }
            }
        }
    }
}

/// Tiled matrix multiply: C = A * B + bias (broadcast per row of C).
///
/// A: m x k, B: k x n, C: m x n
fn gemm_tiled(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
) {
    // Initialize with bias
    for i in 0..m {
        let bias_val = bias[i];
        let row_off = i * n;
        for j in 0..n {
            c[row_off + j] = bias_val;
        }
    }

    // Tiled multiply
    let mut ii = 0;
    while ii < m {
        let i_end = (ii + TILE).min(m);
        let mut pp = 0;
        while pp < k {
            let p_end = (pp + TILE).min(k);
            let mut jj = 0;
            while jj < n {
                let j_end = (jj + TILE).min(n);
                for i in ii..i_end {
                    let c_row = i * n;
                    let a_row = i * k;
                    for p in pp..p_end {
                        let a_val = a[a_row + p];
                        let b_row = p * n;
                        simd::axpy_f32(c, c_row + jj, b, b_row + jj, a_val, j_end - jj);
                    }
                }
                jj += TILE;
            }
            pp += TILE;
        }
        ii += TILE;
    }
}

/// Im2col + GEMM convolution.
///
/// Works with any kernel size and stride.
pub fn conv2d_im2col(
    input: &[f32],
    batch: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    weights: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    output: &mut [f32],
) {
    let col_rows = in_channels * kernel_size * kernel_size;
    let col_cols = out_h * out_w;
    let mut col = vec![0.0f32; col_rows * col_cols];

    let in_spatial = in_channels * in_h * in_w;
    let out_spatial = out_channels * out_h * out_w;

    for n in 0..batch {
        let in_off = n * in_spatial;
        let out_off = n * out_spatial;

        im2col(
            &input[in_off..],
            in_channels,
            in_h,
            in_w,
            kernel_size,
            stride,
            out_h,
            out_w,
            &mut col,
        );

        // weights is (out_channels x col_rows), col is (col_rows x col_cols)
        // output slice is (out_channels x col_cols)
        gemm_tiled(
            weights,
            &col,
            bias,
            out_channels,
            col_cols,
            col_rows,
            &mut output[out_off..out_off + out_spatial],
        );
    }
}

// ── INT8 Im2col + GEMM ──

/// Unfold INT8 input patches into a column matrix (transposed: cols x rows).
///
/// Output layout: `col_t[col_idx * k + row]` where col_idx = oh*out_w+ow, k = in_channels*kH*kW.
/// This transpose gives contiguous k-access for both weight rows and input columns.
fn im2col_i8(
    input: &[i8],
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    kernel_size: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    col_t: &mut [i8],
) {
    let k = in_channels * kernel_size * kernel_size;
    for oh in 0..out_h {
        for ow in 0..out_w {
            let col_idx = oh * out_w + ow;
            let dst_off = col_idx * k;
            for ic in 0..in_channels {
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let row = ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        let ih = oh * stride + kh;
                        let iw = ow * stride + kw;
                        col_t[dst_off + row] = input[ic * in_h * in_w + ih * in_w + iw];
                    }
                }
            }
        }
    }
}

/// INT8 GEMM: for each (oc, spatial), compute dot(weight_row, col_t_row) with zero-point
/// correction, dequantize, add bias, requantize.
///
/// weight: out_channels x k (row-major, contiguous along k)
/// col_t:  n_cols x k (row-major, contiguous along k)
fn gemm_i8(
    weight: &[i8],
    col_t: &[i8],
    bias: &[f32],
    out_channels: usize,
    n_cols: usize,
    k: usize,
    input_zp: i32,
    combined_scale: f32,
    output_scale: f32,
    output_zp: i32,
    output: &mut [i8],
) {
    let inv_out_scale = 1.0 / output_scale;

    // Precompute sum of weights per output channel for zero-point correction:
    // sum((in - zp) * w) = sum(in * w) - zp * sum(w)
    let mut weight_sums = vec![0i32; out_channels];
    for oc in 0..out_channels {
        weight_sums[oc] = simd::sum_i8(weight, oc * k, k);
    }

    for oc in 0..out_channels {
        let w_off = oc * k;
        let w_sum = weight_sums[oc];
        let bias_val = bias[oc];
        let out_row = oc * n_cols;

        for col in 0..n_cols {
            let c_off = col * k;
            // dot(input, weight) without zero-point subtraction
            let raw_dot = simd::dot_i8(weight, w_off, col_t, c_off, k);
            // Apply zero-point correction: sum((in-zp)*w) = sum(in*w) - zp*sum(w)
            let corrected = raw_dot - input_zp * w_sum;
            // Dequantize to f32, add bias, requantize
            let result = corrected as f32 * combined_scale + bias_val;
            let quantized = (result * inv_out_scale).round() as i32 + output_zp;
            output[out_row + col] = quantized.clamp(-128, 127) as i8;
        }
    }
}

/// INT8 Im2col + GEMM convolution entry point.
///
/// Input/weights are flat `&[i8]` in NCHW layout. Output is flat `&mut [i8]` in NCHW.
pub fn conv2d_im2col_i8(
    input: &[i8],
    batch: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    weights: &[i8],
    bias: &[f32],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    input_zp: i32,
    combined_scale: f32,
    output_scale: f32,
    output_zp: i32,
    output: &mut [i8],
) {
    let k = in_channels * kernel_size * kernel_size;
    let n_cols = out_h * out_w;
    let mut col_t = vec![0i8; n_cols * k];

    let in_spatial = in_channels * in_h * in_w;
    let out_spatial = out_channels * out_h * out_w;

    for n in 0..batch {
        let in_off = n * in_spatial;
        let out_off = n * out_spatial;

        im2col_i8(
            &input[in_off..],
            in_channels,
            in_h,
            in_w,
            kernel_size,
            stride,
            out_h,
            out_w,
            &mut col_t,
        );

        gemm_i8(
            weights,
            &col_t,
            bias,
            out_channels,
            n_cols,
            k,
            input_zp,
            combined_scale,
            output_scale,
            output_zp,
            &mut output[out_off..out_off + out_spatial],
        );
    }
}
