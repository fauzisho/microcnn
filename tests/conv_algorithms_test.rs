use microcnn::conv::{ConvAlgorithm, conv2d, conv2d_naive};
use microcnn::network::Conv2dLayer;
use microcnn::tensor::Tensor;
use microcnn::network::Layer;

/// Fill a flat vec with deterministic values based on index.
fn fill_deterministic(data: &mut [f32]) {
    for (i, v) in data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.1 + 0.05).sin();
    }
}

/// Compare two output slices with a given tolerance.
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", label, a.len(), b.len());
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (va - vb).abs() < tol,
            "{}: mismatch at index {}: {} vs {} (diff={})",
            label, i, va, vb, (va - vb).abs()
        );
    }
}

/// Helper: run naive and return output for comparison.
fn run_naive(
    input: &[f32], batch: usize, in_c: usize, in_h: usize, in_w: usize,
    weights: &[f32], bias: &[f32], out_c: usize, kernel: usize, stride: usize,
    out_h: usize, out_w: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_c * out_h * out_w];
    conv2d_naive(input, batch, in_c, in_h, in_w, weights, bias, out_c, kernel, stride, out_h, out_w, &mut output, false);
    output
}

/// Helper: run a given algorithm and return output.
fn run_algorithm(
    algo: ConvAlgorithm,
    input: &[f32], batch: usize, in_c: usize, in_h: usize, in_w: usize,
    weights: &[f32], bias: &[f32], out_c: usize, kernel: usize, stride: usize,
    out_h: usize, out_w: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_c * out_h * out_w];
    conv2d(algo, input, batch, in_c, in_h, in_w, weights, bias, out_c, kernel, stride, out_h, out_w, &mut output, false);
    output
}

// Test 1: Small 1-channel input (1x1x5x5, 3x3 kernel)
#[test]
fn test_small_single_channel() {
    let (batch, in_c, in_h, in_w) = (1, 1, 5, 5);
    let (out_c, kernel, stride) = (1, 3, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.5f32; out_c];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} small_single_channel", algo));
    }
}

// Test 2: Multi-channel (1x3x8x8, 4 output channels, 3x3 kernel)
#[test]
fn test_multi_channel() {
    let (batch, in_c, in_h, in_w) = (1, 3, 8, 8);
    let (out_c, kernel, stride) = (4, 3, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.1, -0.2, 0.3, -0.1];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} multi_channel", algo));
    }
}

// Test 3: With padding (input pre-padded, simulating pad=1 on a 4x4 input → 6x6 padded)
#[test]
fn test_with_padding() {
    let (batch, in_c, orig_h, orig_w) = (1, 1, 4, 4);
    let pad = 1;
    let (in_h, in_w) = (orig_h + 2 * pad, orig_w + 2 * pad); // 6x6
    let (out_c, kernel, stride) = (2, 3, 1);
    let out_h = in_h - kernel + 1; // 4
    let out_w = in_w - kernel + 1; // 4

    // Create padded input
    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    for h in 0..orig_h {
        for w in 0..orig_w {
            input[(h + pad) * in_w + (w + pad)] = ((h * orig_w + w) as f32) * 0.1;
        }
    }

    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.0; out_c];
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} with_padding", algo));
    }
}

// Test 4: Batch dimension (2x1x5x5)
#[test]
fn test_batch() {
    let (batch, in_c, in_h, in_w) = (2, 1, 5, 5);
    let (out_c, kernel, stride) = (2, 3, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.1, -0.1];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} batch", algo));
    }
}

// Test 5: Stride > 1 (1x1x8x8, stride=2)
#[test]
fn test_stride() {
    let (batch, in_c, in_h, in_w) = (1, 1, 8, 8);
    let (out_c, kernel, stride) = (1, 3, 2);
    let out_h = (in_h - kernel) / stride + 1;
    let out_w = (in_w - kernel) / stride + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.0; out_c];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    // Im2col and FFT support stride; Winograd falls back to Im2col for stride != 1
    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} stride", algo));
    }
}

// Test 6: 5x5 kernel (Im2col and FFT only, Winograd falls back)
#[test]
fn test_5x5_kernel() {
    let (batch, in_c, in_h, in_w) = (1, 1, 10, 10);
    let (out_c, kernel, stride) = (2, 5, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.2, -0.3];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} 5x5_kernel", algo));
    }
}

// Test 7: Odd spatial dims (1x1x7x7, tests Winograd edge tiles)
#[test]
fn test_odd_dims() {
    let (batch, in_c, in_h, in_w) = (1, 1, 7, 7);
    let (out_c, kernel, stride) = (1, 3, 1);
    let out_h = in_h - kernel + 1; // 5 (odd)
    let out_w = in_w - kernel + 1; // 5 (odd)

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.0; out_c];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} odd_dims", algo));
    }
}

// Test 8: Non-square input (1x1x6x10)
#[test]
fn test_non_square() {
    let (batch, in_c, in_h, in_w) = (1, 1, 6, 10);
    let (out_c, kernel, stride) = (2, 3, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.1, 0.2];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    for algo in [ConvAlgorithm::Im2col, ConvAlgorithm::Winograd, ConvAlgorithm::Fft] {
        let result = run_algorithm(algo, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
        assert_approx_eq(&naive, &result, 1e-4, &format!("{:?} non_square", algo));
    }
}

// Test 9: Winograd fallback test (5x5 kernel → should produce same result as Im2col)
#[test]
fn test_winograd_fallback() {
    let (batch, in_c, in_h, in_w) = (1, 1, 10, 10);
    let (out_c, kernel, stride) = (1, 5, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.0; out_c];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let im2col_result = run_algorithm(ConvAlgorithm::Im2col, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
    let winograd_result = run_algorithm(ConvAlgorithm::Winograd, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    // Winograd should fall back to Im2col for 5x5 kernels, so results should be identical
    assert_approx_eq(&im2col_result, &winograd_result, 1e-6, "winograd_fallback");
}

// Test 10: FFT precision test with larger input (1x1x32x32, tolerance 1e-3)
#[test]
fn test_fft_precision_large() {
    let (batch, in_c, in_h, in_w) = (1, 1, 32, 32);
    let (out_c, kernel, stride) = (1, 3, 1);
    let out_h = in_h - kernel + 1;
    let out_w = in_w - kernel + 1;

    let mut input = vec![0.0f32; batch * in_c * in_h * in_w];
    let mut weights = vec![0.0f32; out_c * in_c * kernel * kernel];
    let bias = vec![0.0; out_c];
    fill_deterministic(&mut input);
    fill_deterministic(&mut weights);

    let naive = run_naive(&input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);
    let fft_result = run_algorithm(ConvAlgorithm::Fft, &input, batch, in_c, in_h, in_w, &weights, &bias, out_c, kernel, stride, out_h, out_w);

    assert_approx_eq(&naive, &fft_result, 1e-3, "fft_precision_large");
}

// Test 11: Integration test via Conv2dLayer::with_algorithm()
#[test]
fn test_conv2d_layer_integration() {
    let (in_c, out_c, kernel, stride, pad) = (1, 2, 3, 1, 1);
    let (in_h, in_w) = (5, 5);

    // Create input tensor
    let input = Tensor::new(1, in_c, in_h, in_w);
    for h in 0..in_h {
        for w in 0..in_w {
            input.set(0, 0, h, w, ((h * in_w + w) as f32) * 0.1);
        }
    }

    // Run with Naive
    let mut naive_layer = Conv2dLayer::new(in_c, out_c, kernel, stride, pad);
    // Set some weights
    for oc in 0..out_c {
        for ic in 0..in_c {
            for kh in 0..kernel {
                for kw in 0..kernel {
                    naive_layer.weights_ref().set(oc, ic, kh, kw, ((oc * 9 + ic * 9 + kh * 3 + kw) as f32) * 0.1);
                }
            }
        }
    }
    naive_layer.bias_ref().set(0, 0, 0, 0, 0.1);
    naive_layer.bias_ref().set(1, 0, 0, 0, -0.1);
    naive_layer.set_input(input.clone());
    naive_layer.fwd();

    // Run with Im2col
    let mut im2col_layer = Conv2dLayer::with_algorithm(in_c, out_c, kernel, stride, pad, ConvAlgorithm::Im2col);
    // Copy same weights
    for oc in 0..out_c {
        for ic in 0..in_c {
            for kh in 0..kernel {
                for kw in 0..kernel {
                    im2col_layer.weights_ref().set(oc, ic, kh, kw, ((oc * 9 + ic * 9 + kh * 3 + kw) as f32) * 0.1);
                }
            }
        }
    }
    im2col_layer.bias_ref().set(0, 0, 0, 0, 0.1);
    im2col_layer.bias_ref().set(1, 0, 0, 0, -0.1);
    im2col_layer.set_input(input.clone());
    im2col_layer.fwd();

    // Compare outputs
    let out_h = (in_h + 2 * pad - kernel) / stride + 1;
    let out_w = (in_w + 2 * pad - kernel) / stride + 1;
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let naive_val = naive_layer.output().get(0, oc, oh, ow);
                let im2col_val = im2col_layer.output().get(0, oc, oh, ow);
                assert!(
                    (naive_val - im2col_val).abs() < 1e-4,
                    "Integration mismatch at ({},{},{}): {} vs {}",
                    oc, oh, ow, naive_val, im2col_val
                );
            }
        }
    }
}
