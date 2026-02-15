/// Tests for SIMD-accelerated INT8 Im2col+GEMM convolution.

use microcnn::conv::conv2d_im2col_i8;

/// Reference scalar INT8 convolution (6-nested-loop) for correctness comparison.
fn conv2d_scalar_i8(
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
    let inv_out_scale = 1.0 / output_scale;
    let in_spatial = in_channels * in_h * in_w;
    let out_spatial = out_channels * out_h * out_w;

    for n in 0..batch {
        let in_off = n * in_spatial;
        let out_off = n * out_spatial;
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum: i32 = 0;
                    for ic in 0..in_channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let inp = input[in_off + ic * in_h * in_w + ih * in_w + iw] as i32;
                                let wt = weights[oc * in_channels * kernel_size * kernel_size
                                    + ic * kernel_size * kernel_size
                                    + kh * kernel_size + kw] as i32;
                                sum += (inp - input_zp) * wt;
                            }
                        }
                    }
                    let result = sum as f32 * combined_scale + bias[oc];
                    let quantized = (result * inv_out_scale).round() as i32 + output_zp;
                    output[out_off + oc * out_h * out_w + oh * out_w + ow] =
                        quantized.clamp(-128, 127) as i8;
                }
            }
        }
    }
}

#[test]
fn test_i8_im2col_vs_scalar_3x3() {
    // 1 batch, 1 input channel, 5x5 input, 1 output channel, 3x3 kernel, stride 1
    let batch = 1;
    let in_c = 1;
    let in_h = 5;
    let in_w = 5;
    let out_c = 2;
    let ks = 3;
    let stride = 1;
    let out_h = (in_h - ks) / stride + 1;
    let out_w = (in_w - ks) / stride + 1;

    let input: Vec<i8> = (0..batch * in_c * in_h * in_w)
        .map(|i| ((i % 200) as i8).wrapping_sub(50))
        .collect();
    let weights: Vec<i8> = (0..out_c * in_c * ks * ks)
        .map(|i| ((i * 7 % 100) as i8).wrapping_sub(50))
        .collect();
    let bias = vec![0.5f32, -0.3];
    let input_zp = 0;
    let combined_scale = 0.01;
    let output_scale = 0.05;
    let output_zp = 0;

    let mut out_scalar = vec![0i8; batch * out_c * out_h * out_w];
    let mut out_im2col = vec![0i8; batch * out_c * out_h * out_w];

    conv2d_scalar_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_scalar,
    );
    conv2d_im2col_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_im2col,
    );

    assert_eq!(out_scalar, out_im2col, "3x3 kernel: im2col != scalar");
}

#[test]
fn test_i8_im2col_vs_scalar_5x5_non_aligned_k() {
    // k = 1*5*5 = 25, not a multiple of 16 — tests SIMD tail handling
    let batch = 2;
    let in_c = 1;
    let in_h = 8;
    let in_w = 8;
    let out_c = 3;
    let ks = 5;
    let stride = 1;
    let out_h = (in_h - ks) / stride + 1;
    let out_w = (in_w - ks) / stride + 1;

    let input: Vec<i8> = (0..batch * in_c * in_h * in_w)
        .map(|i| ((i * 3 % 256) as i8))
        .collect();
    let weights: Vec<i8> = (0..out_c * in_c * ks * ks)
        .map(|i| ((i * 11 % 200) as i8).wrapping_sub(100))
        .collect();
    let bias = vec![1.0f32, 0.0, -1.0];
    let input_zp = 0;
    let combined_scale = 0.005;
    let output_scale = 0.02;
    let output_zp = 0;

    let mut out_scalar = vec![0i8; batch * out_c * out_h * out_w];
    let mut out_im2col = vec![0i8; batch * out_c * out_h * out_w];

    conv2d_scalar_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_scalar,
    );
    conv2d_im2col_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_im2col,
    );

    assert_eq!(out_scalar, out_im2col, "5x5 kernel (k=25): im2col != scalar");
}

#[test]
fn test_i8_im2col_nonzero_zero_point() {
    // Non-zero input zero_point tests the zp correction logic
    let batch = 1;
    let in_c = 3;
    let in_h = 6;
    let in_w = 6;
    let out_c = 4;
    let ks = 3;
    let stride = 1;
    let out_h = (in_h - ks) / stride + 1;
    let out_w = (in_w - ks) / stride + 1;

    let input: Vec<i8> = (0..batch * in_c * in_h * in_w)
        .map(|i| ((i % 180) as i8).wrapping_sub(20))
        .collect();
    let weights: Vec<i8> = (0..out_c * in_c * ks * ks)
        .map(|i| ((i * 5 % 150) as i8).wrapping_sub(75))
        .collect();
    let bias = vec![0.1, -0.2, 0.3, -0.4];
    let input_zp = 10; // non-zero zero point
    let combined_scale = 0.002;
    let output_scale = 0.01;
    let output_zp = -5;

    let mut out_scalar = vec![0i8; batch * out_c * out_h * out_w];
    let mut out_im2col = vec![0i8; batch * out_c * out_h * out_w];

    conv2d_scalar_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_scalar,
    );
    conv2d_im2col_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_im2col,
    );

    assert_eq!(out_scalar, out_im2col, "non-zero zp: im2col != scalar");
}

#[test]
fn test_i8_im2col_multi_channel() {
    // Multi-channel with stride=2
    let batch = 1;
    let in_c = 6;
    let in_h = 14;
    let in_w = 14;
    let out_c = 16;
    let ks = 5;
    let stride = 2;
    let out_h = (in_h - ks) / stride + 1;
    let out_w = (in_w - ks) / stride + 1;

    let input: Vec<i8> = (0..batch * in_c * in_h * in_w)
        .map(|i| ((i * 13 % 256) as i8))
        .collect();
    let weights: Vec<i8> = (0..out_c * in_c * ks * ks)
        .map(|i| ((i * 7 % 256) as i8))
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|i| i as f32 * 0.1 - 0.5).collect();
    let input_zp = -3;
    let combined_scale = 0.001;
    let output_scale = 0.015;
    let output_zp = 2;

    let mut out_scalar = vec![0i8; batch * out_c * out_h * out_w];
    let mut out_im2col = vec![0i8; batch * out_c * out_h * out_w];

    conv2d_scalar_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_scalar,
    );
    conv2d_im2col_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut out_im2col,
    );

    assert_eq!(out_scalar, out_im2col, "multi-channel stride=2: im2col != scalar");
}
