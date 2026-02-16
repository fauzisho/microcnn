/// Tests that Conv2d+ReLU fusion produces identical output to sequential Conv2d → ReLU.

use microcnn::conv::{conv2d, conv2d_im2col_i8, ConvAlgorithm};

fn fill_deterministic(data: &mut [f32]) {
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.1 + 0.05).sin();
    }
}

/// FP32: fused vs sequential must be bit-identical for each algorithm.
#[test]
fn test_fp32_fusion_correctness() {
    let (batch, in_c, in_h, in_w) = (1, 3, 8, 8);
    let (out_c, kernel, stride) = (4, 3, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;
    let out_size = batch * out_c * out_h * out_w;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.1, -0.2, 0.3, -0.1];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    for algo in [ConvAlgorithm::Naive, ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        // Sequential: conv2d then manual ReLU
        let mut sequential = vec![0.0f32; out_size];
        conv2d(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w, &mut sequential, false);
        for v in sequential.iter_mut() {
            *v = v.max(0.0);
        }

        // Fused: conv2d with relu=true
        let mut fused = vec![0.0f32; out_size];
        conv2d(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w, &mut fused, true);

        assert_eq!(sequential, fused, "{:?}: fused != sequential", algo);
    }
}

/// INT8: fused vs sequential must produce identical output.
#[test]
fn test_i8_fusion_correctness() {
    let batch = 1;
    let in_c = 3;
    let in_h = 6;
    let in_w = 6;
    let out_c = 4;
    let ks = 3;
    let stride = 1;
    let out_h = (in_h - ks) / stride + 1;
    let out_w = (in_w - ks) / stride + 1;
    let out_size = batch * out_c * out_h * out_w;

    let input: Vec<i8> = (0..batch * in_c * in_h * in_w)
        .map(|i| ((i % 180) as i8).wrapping_sub(20))
        .collect();
    let weights: Vec<i8> = (0..out_c * in_c * ks * ks)
        .map(|i| ((i * 5 % 150) as i8).wrapping_sub(75))
        .collect();
    let bias = vec![0.1, -0.2, 0.3, -0.4];
    let input_zp = 10;
    let combined_scale = 0.002;
    let output_scale = 0.01;
    let output_zp = -5;

    // Sequential: conv then manual ReLU (clamp to output_zp)
    let mut sequential = vec![0i8; out_size];
    conv2d_im2col_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut sequential, false,
    );
    let zp_i8 = output_zp.clamp(-128, 127) as i8;
    for v in sequential.iter_mut() {
        *v = (*v).max(zp_i8);
    }

    // Fused
    let mut fused = vec![0i8; out_size];
    conv2d_im2col_i8(
        &input, batch, in_c, in_h, in_w, &weights, &bias,
        out_c, ks, stride, out_h, out_w,
        input_zp, combined_scale, output_scale, output_zp, &mut fused, true,
    );

    assert_eq!(sequential, fused, "INT8: fused != sequential");
}
