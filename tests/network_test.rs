use microcnn::tensor::Tensor;
use microcnn::network::*;

#[test]
fn relu_zeros_negatives() {
    let input = Tensor::new(1, 1, 1, 4);
    input.set(0, 0, 0, 0, -1.0);
    input.set(0, 0, 0, 1, 0.0);
    input.set(0, 0, 0, 2, 0.5);
    input.set(0, 0, 0, 3, -0.3);

    let mut layer = ReLuLayer::new();
    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    assert_eq!(out.get(0, 0, 0, 0), 0.0);
    assert_eq!(out.get(0, 0, 0, 1), 0.0);
    assert_eq!(out.get(0, 0, 0, 2), 0.5);
    assert_eq!(out.get(0, 0, 0, 3), 0.0);
}

#[test]
fn softmax_sums_to_one() {
    let input = Tensor::new2(1, 3);
    input.set(0, 0, 0, 0, 1.0);
    input.set(0, 1, 0, 0, 2.0);
    input.set(0, 2, 0, 0, 3.0);

    let mut layer = SoftMaxLayer::new();
    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    let sum: f32 = (0..3).map(|c| out.get(0, c, 0, 0)).sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {}", sum);

    // Outputs should be in increasing order
    assert!(out.get(0, 0, 0, 0) < out.get(0, 1, 0, 0));
    assert!(out.get(0, 1, 0, 0) < out.get(0, 2, 0, 0));
}

#[test]
fn linear_basic() {
    // 2 inputs, 1 output: y = w0*x0 + w1*x1 + bias
    let mut layer = LinearLayer::new(2, 1);
    // Manually set weights and bias
    // weights is Tensor::new2(1, 2) = (1, 2, 1, 1)
    // We need to write via the Layer trait's read_weights_bias, but we can use
    // a simple approach: create input and check dimensions

    // For a basic test, just verify it runs without panicking
    let input = Tensor::new2(1, 2);
    input.set(0, 0, 0, 0, 1.0);
    input.set(0, 1, 0, 0, 1.0);

    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    // With zero weights and zero bias, output should be 0
    assert_eq!(out.get(0, 0, 0, 0), 0.0);
}

#[test]
fn conv2d_output_dimensions() {
    // 1 input channel, 2 output channels, 3x3 kernel, stride 1, no padding
    // Input: 1x1x5x5 -> Output should be 1x2x3x3
    let mut layer = Conv2dLayer::new(1, 2, 3, 1, 0);
    let input = Tensor::new(1, 1, 5, 5);
    input.fill(1.0);

    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    assert_eq!(out.n, 1);
    assert_eq!(out.c, 2);
    assert_eq!(out.h, 3);
    assert_eq!(out.w, 3);
}

#[test]
fn conv2d_with_padding() {
    // With padding=1, 3x3 kernel, stride 1: output same as input spatial dims
    let mut layer = Conv2dLayer::new(1, 1, 3, 1, 1);
    let input = Tensor::new(1, 1, 4, 4);
    input.fill(1.0);

    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    assert_eq!(out.h, 4);
    assert_eq!(out.w, 4);
}

#[test]
fn flatten_reshapes_correctly() {
    let input = Tensor::new(1, 2, 3, 4);
    input.fill(1.0);
    input.set(0, 1, 2, 3, 42.0);

    let mut layer = FlattenLayer::new();
    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    assert_eq!(out.n, 1);
    assert_eq!(out.c, 24); // 2*3*4
    assert_eq!(out.h, 1);
    assert_eq!(out.w, 1);
}

#[test]
fn maxpool2d_basic() {
    // 2x2 pooling, stride 2 on a 4x4 input
    let mut layer = MaxPool2dLayer::new(2, 2, 0);
    let input = Tensor::new(1, 1, 4, 4);
    // Fill with sequential values
    for h in 0..4 {
        for w in 0..4 {
            input.set(0, 0, h, w, (h * 4 + w) as f32);
        }
    }

    layer.set_input(input);
    layer.fwd();

    let out = layer.output();
    assert_eq!(out.h, 2);
    assert_eq!(out.w, 2);
    // Top-left 2x2 block: max of 0,1,4,5 = 5
    assert_eq!(out.get(0, 0, 0, 0), 5.0);
    // Top-right: max of 2,3,6,7 = 7
    assert_eq!(out.get(0, 0, 0, 1), 7.0);
}

#[test]
fn neural_network_predict_empty_returns_empty() {
    let mut net = NeuralNetwork::new(false);
    net.add(Box::new(ReLuLayer::new()));
    let result = net.predict(Tensor::empty());
    assert!(result.is_empty());
}
