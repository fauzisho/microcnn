/// Winograd F(2x2, 3x3) convolution.
///
/// Only supports 3x3 kernels with stride=1. For other configurations,
/// the dispatcher falls back to Im2col.

use alloc::vec;

/// Transform a 3x3 filter tile using G * g * G^T.
/// G is 4x3, g is 3x3, result U is 4x4.
#[inline]
fn transform_filter(g: &[f32; 9]) -> [f32; 16] {
    // G matrix for F(2x2, 3x3):
    // [ 1     0     0   ]
    // [ 1/2   1/2   1/2 ]
    // [ 1/2  -1/2   1/2 ]
    // [ 0     0     1   ]
    //
    // Compute G * g (4x3 * 3x3 = 4x3 intermediate)
    let mut tmp = [0.0f32; 12]; // 4x3
    for j in 0..3 {
        let g0 = g[0 * 3 + j];
        let g1 = g[1 * 3 + j];
        let g2 = g[2 * 3 + j];
        tmp[0 * 3 + j] = g0;
        tmp[1 * 3 + j] = (g0 + g1 + g2) * 0.5;
        tmp[2 * 3 + j] = (g0 - g1 + g2) * 0.5;
        tmp[3 * 3 + j] = g2;
    }
    // Compute (G * g) * G^T (4x3 * 3x4 = 4x4)
    let mut u = [0.0f32; 16];
    for i in 0..4 {
        let t0 = tmp[i * 3 + 0];
        let t1 = tmp[i * 3 + 1];
        let t2 = tmp[i * 3 + 2];
        u[i * 4 + 0] = t0;
        u[i * 4 + 1] = (t0 + t1 + t2) * 0.5;
        u[i * 4 + 2] = (t0 - t1 + t2) * 0.5;
        u[i * 4 + 3] = t2;
    }
    u
}

/// Transform a 4x4 input tile using B^T * d * B.
/// B^T is 4x4, d is 4x4, result V is 4x4.
#[inline]
fn transform_input(d: &[f32; 16]) -> [f32; 16] {
    // B^T matrix for F(2x2, 3x3):
    // [ 1   0  -1   0 ]
    // [ 0   1   1   0 ]
    // [ 0  -1   1   0 ]
    // [ 0   1   0  -1 ]
    //
    // Compute B^T * d (4x4 * 4x4 = 4x4 intermediate)
    let mut tmp = [0.0f32; 16];
    for j in 0..4 {
        let d0 = d[0 * 4 + j];
        let d1 = d[1 * 4 + j];
        let d2 = d[2 * 4 + j];
        let d3 = d[3 * 4 + j];
        tmp[0 * 4 + j] = d0 - d2;
        tmp[1 * 4 + j] = d1 + d2;
        tmp[2 * 4 + j] = -d1 + d2;
        tmp[3 * 4 + j] = d1 - d3;
    }
    // Compute (B^T * d) * B (4x4 * 4x4 = 4x4)
    let mut v = [0.0f32; 16];
    for i in 0..4 {
        let t0 = tmp[i * 4 + 0];
        let t1 = tmp[i * 4 + 1];
        let t2 = tmp[i * 4 + 2];
        let t3 = tmp[i * 4 + 3];
        v[i * 4 + 0] = t0 - t2;
        v[i * 4 + 1] = t1 + t2;
        v[i * 4 + 2] = -t1 + t2;
        v[i * 4 + 3] = t1 - t3;
    }
    v
}

/// Inverse transform: A^T * m * A, producing 2x2 output from 4x4.
/// A^T is 2x4, m is 4x4, result is 2x2.
#[inline]
fn transform_output(m: &[f32; 16]) -> [f32; 4] {
    // A^T matrix for F(2x2, 3x3):
    // [ 1  1  1  0 ]
    // [ 0  1 -1 -1 ]
    //
    // Compute A^T * m (2x4 * 4x4 = 2x4 intermediate)
    let mut tmp = [0.0f32; 8]; // 2x4
    for j in 0..4 {
        let m0 = m[0 * 4 + j];
        let m1 = m[1 * 4 + j];
        let m2 = m[2 * 4 + j];
        let m3 = m[3 * 4 + j];
        tmp[0 * 4 + j] = m0 + m1 + m2;
        tmp[1 * 4 + j] = m1 - m2 - m3;
    }
    // Compute (A^T * m) * A (2x4 * 4x2 = 2x2)
    let mut out = [0.0f32; 4];
    for i in 0..2 {
        let t0 = tmp[i * 4 + 0];
        let t1 = tmp[i * 4 + 1];
        let t2 = tmp[i * 4 + 2];
        let t3 = tmp[i * 4 + 3];
        out[i * 2 + 0] = t0 + t1 + t2;
        out[i * 2 + 1] = t1 - t2 - t3;
    }
    out
}

/// Winograd F(2x2, 3x3) convolution.
///
/// Only for 3x3 kernels with stride=1. Input is already padded.
pub fn conv2d_winograd(
    input: &[f32],
    batch: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    weights: &[f32],
    bias: &[f32],
    out_channels: usize,
    out_h: usize,
    out_w: usize,
    output: &mut [f32],
    relu: bool,
) {
    // Pre-transform all filters: U[oc][ic] = G * g * G^T
    let mut u_all = vec![[0.0f32; 16]; out_channels * in_channels];
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            let mut g = [0.0f32; 9];
            let w_base = oc * in_channels * 9 + ic * 9;
            g.copy_from_slice(&weights[w_base..w_base + 9]);
            u_all[oc * in_channels + ic] = transform_filter(&g);
        }
    }

    // Number of 2x2 output tiles
    let tiles_h = (out_h + 1) / 2;
    let tiles_w = (out_w + 1) / 2;

    let in_spatial = in_channels * in_h * in_w;
    let out_spatial = out_channels * out_h * out_w;

    for n in 0..batch {
        let in_off = n * in_spatial;
        let out_off = n * out_spatial;

        for oc in 0..out_channels {
            let bias_val = bias[oc];

            for th in 0..tiles_h {
                for tw in 0..tiles_w {
                    // Accumulate element-wise products across input channels
                    let mut acc = [0.0f32; 16];

                    for ic in 0..in_channels {
                        // Extract 4x4 input tile
                        let mut d = [0.0f32; 16];
                        let base_h = th * 2;
                        let base_w = tw * 2;
                        for dh in 0..4 {
                            for dw in 0..4 {
                                let ih = base_h + dh;
                                let iw = base_w + dw;
                                if ih < in_h && iw < in_w {
                                    d[dh * 4 + dw] = input[in_off + ic * in_h * in_w + ih * in_w + iw];
                                }
                            }
                        }

                        let v = transform_input(&d);
                        let u = &u_all[oc * in_channels + ic];

                        // Element-wise multiply and accumulate
                        for i in 0..16 {
                            acc[i] += u[i] * v[i];
                        }
                    }

                    // Inverse transform to get 2x2 output tile
                    let out_tile = transform_output(&acc);

                    // Write output, handling edge tiles
                    let oh_base = th * 2;
                    let ow_base = tw * 2;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let oh = oh_base + dy;
                            let ow = ow_base + dx;
                            if oh < out_h && ow < out_w {
                                let val = out_tile[dy * 2 + dx] + bias_val;
                                output[out_off + oc * out_h * out_w + oh * out_w + ow] =
                                    if relu { val.max(0.0) } else { val };
                            }
                        }
                    }
                }
            }
        }
    }
}
