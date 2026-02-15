/// Im2col + tiled GEMM convolution.
///
/// Unfolds input patches into a column matrix, then performs a tiled matrix multiply
/// for cache-friendly access patterns.

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
                        for j in jj..j_end {
                            c[c_row + j] += a_val * b[b_row + j];
                        }
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
