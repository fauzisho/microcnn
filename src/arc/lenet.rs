use crate::conv::ConvAlgorithm;
use crate::network::*;
use crate::quantization::{QuantParams, quantize_tensor_symmetric, quantize_tensor_symmetric_i4};

/// Build a FP32 LeNet-5 network for 28x28 grayscale images (10-class output).
pub fn lenet(debug: bool) -> NeuralNetwork {
    let mut net = NeuralNetwork::new(debug);
    net.add(Box::new(Conv2dLayer::new(1, 6, 5, 1, 2)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::new(6, 16, 5, 1, 0)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::new(16, 120, 5, 1, 0)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(LinearLayer::new(120, 84)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(LinearLayer::new(84, 10)));
    net.add(Box::new(SoftMaxLayer::new()));
    net
}

/// Build a FP32 LeNet-5 using a specific convolution algorithm.
pub fn lenet_with_algorithm(debug: bool, algorithm: ConvAlgorithm) -> NeuralNetwork {
    let mut net = NeuralNetwork::new(debug);
    net.add(Box::new(Conv2dLayer::with_algorithm(1, 6, 5, 1, 2, algorithm)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::with_algorithm(6, 16, 5, 1, 0, algorithm)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::with_algorithm(16, 120, 5, 1, 0, algorithm)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(LinearLayer::new(120, 84)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(LinearLayer::new(84, 10)));
    net.add(Box::new(SoftMaxLayer::new()));
    net
}

/// Build an INT8 quantized LeNet-5 from a calibrated FP32 network.
pub fn lenet_quantized(
    fp32_net: &NeuralNetwork,
    input_params: QuantParams,
    layer_params: &[QuantParams],
) -> QuantizedNeuralNetwork {
    let layers = fp32_net.layers();
    let mut qnet = QuantizedNeuralNetwork::new(input_params);

    // Layer 0: Conv2d(1, 6, 5, 1, 2)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[0].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ::new(
        1, 6, 5, 1, 2, w_i8, w_params, layers[0].bias_ref().clone(), layer_params[0].clone(),
    )));

    // Layer 1: ReLu
    qnet.add(Box::new(ReLuLayerQ::new()));

    // Layer 2: MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ::new(2, 2, 0)));

    // Layer 3: Conv2d(6, 16, 5, 1, 0)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[3].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ::new(
        6, 16, 5, 1, 0, w_i8, w_params, layers[3].bias_ref().clone(), layer_params[3].clone(),
    )));

    // Layer 4: ReLu
    qnet.add(Box::new(ReLuLayerQ::new()));

    // Layer 5: MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ::new(2, 2, 0)));

    // Layer 6: Conv2d(16, 120, 5, 1, 0)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[6].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ::new(
        16, 120, 5, 1, 0, w_i8, w_params, layers[6].bias_ref().clone(), layer_params[6].clone(),
    )));

    // Layer 7: ReLu
    qnet.add(Box::new(ReLuLayerQ::new()));

    // Layer 8: Linear(120, 84)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[8].weights_ref());
    qnet.add(Box::new(LinearLayerQ::new(
        120, 84, w_i8, w_params, layers[8].bias_ref().clone(), layer_params[8].clone(),
    )));

    // Layer 9: ReLu
    qnet.add(Box::new(ReLuLayerQ::new()));

    // Layer 10: Linear(84, 10)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[10].weights_ref());
    qnet.add(Box::new(LinearLayerQ::new(
        84, 10, w_i8, w_params, layers[10].bias_ref().clone(), layer_params[10].clone(),
    )));

    // SoftMax handled internally by QuantizedNeuralNetwork

    qnet
}

/// Build a FP32 LeNet-5 with Conv2d+ReLU fusion (3 fewer layers).
pub fn lenet_fused(debug: bool) -> NeuralNetwork {
    let mut net = NeuralNetwork::new(debug);
    net.add(Box::new(Conv2dLayer::with_relu(1, 6, 5, 1, 2)));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::with_relu(6, 16, 5, 1, 0)));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::with_relu(16, 120, 5, 1, 0)));
    net.add(Box::new(LinearLayer::new(120, 84)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(LinearLayer::new(84, 10)));
    net.add(Box::new(SoftMaxLayer::new()));
    net
}

/// Build a FP32 LeNet-5 with Conv2d+ReLU fusion using a specific convolution algorithm.
pub fn lenet_fused_with_algorithm(debug: bool, algorithm: ConvAlgorithm) -> NeuralNetwork {
    let mut net = NeuralNetwork::new(debug);
    net.add(Box::new(Conv2dLayer::with_algorithm_relu(1, 6, 5, 1, 2, algorithm)));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::with_algorithm_relu(6, 16, 5, 1, 0, algorithm)));
    net.add(Box::new(MaxPool2dLayer::new(2, 2, 0)));
    net.add(Box::new(Conv2dLayer::with_algorithm_relu(16, 120, 5, 1, 0, algorithm)));
    net.add(Box::new(LinearLayer::new(120, 84)));
    net.add(Box::new(ReLuLayer::new()));
    net.add(Box::new(LinearLayer::new(84, 10)));
    net.add(Box::new(SoftMaxLayer::new()));
    net
}

/// Build an INT8 quantized LeNet-5 with Conv2d+ReLU fusion from a calibrated FP32 network.
///
/// Uses the non-fused FP32 network for weight extraction (layer indices 0, 3, 6, 8, 10).
pub fn lenet_quantized_fused(
    fp32_net: &NeuralNetwork,
    input_params: QuantParams,
    layer_params: &[QuantParams],
) -> QuantizedNeuralNetwork {
    let layers = fp32_net.layers();
    let mut qnet = QuantizedNeuralNetwork::new(input_params);

    // Conv2d+ReLu(1, 6, 5, 1, 2) — fused
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[0].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ::new_fused(
        1, 6, 5, 1, 2, w_i8, w_params, layers[0].bias_ref().clone(), layer_params[0].clone(), true,
    )));

    // MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ::new(2, 2, 0)));

    // Conv2d+ReLu(6, 16, 5, 1, 0) — fused
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[3].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ::new_fused(
        6, 16, 5, 1, 0, w_i8, w_params, layers[3].bias_ref().clone(), layer_params[3].clone(), true,
    )));

    // MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ::new(2, 2, 0)));

    // Conv2d+ReLu(16, 120, 5, 1, 0) — fused
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[6].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ::new_fused(
        16, 120, 5, 1, 0, w_i8, w_params, layers[6].bias_ref().clone(), layer_params[6].clone(), true,
    )));

    // Linear(120, 84)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[8].weights_ref());
    qnet.add(Box::new(LinearLayerQ::new(
        120, 84, w_i8, w_params, layers[8].bias_ref().clone(), layer_params[8].clone(),
    )));

    // ReLu
    qnet.add(Box::new(ReLuLayerQ::new()));

    // Linear(84, 10)
    let (w_i8, w_params) = quantize_tensor_symmetric(layers[10].weights_ref());
    qnet.add(Box::new(LinearLayerQ::new(
        84, 10, w_i8, w_params, layers[10].bias_ref().clone(), layer_params[10].clone(),
    )));

    qnet
}

/// Build an INT4 quantized LeNet-5 from a calibrated FP32 network.
pub fn lenet_quantized_i4(
    fp32_net: &NeuralNetwork,
    input_params: QuantParams,
    layer_params: &[QuantParams],
) -> QuantizedNeuralNetworkI4 {
    let layers = fp32_net.layers();
    let mut qnet = QuantizedNeuralNetworkI4::new(input_params);

    // Layer 0: Conv2d(1, 6, 5, 1, 2)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[0].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ4::new(
        1, 6, 5, 1, 2, w_i4, w_params, layers[0].bias_ref().clone(), layer_params[0].clone(),
    )));

    // Layer 1: ReLu
    qnet.add(Box::new(ReLuLayerQ4::new()));

    // Layer 2: MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ4::new(2, 2, 0)));

    // Layer 3: Conv2d(6, 16, 5, 1, 0)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[3].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ4::new(
        6, 16, 5, 1, 0, w_i4, w_params, layers[3].bias_ref().clone(), layer_params[3].clone(),
    )));

    // Layer 4: ReLu
    qnet.add(Box::new(ReLuLayerQ4::new()));

    // Layer 5: MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ4::new(2, 2, 0)));

    // Layer 6: Conv2d(16, 120, 5, 1, 0)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[6].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ4::new(
        16, 120, 5, 1, 0, w_i4, w_params, layers[6].bias_ref().clone(), layer_params[6].clone(),
    )));

    // Layer 7: ReLu
    qnet.add(Box::new(ReLuLayerQ4::new()));

    // Layer 8: Linear(120, 84)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[8].weights_ref());
    qnet.add(Box::new(LinearLayerQ4::new(
        120, 84, w_i4, w_params, layers[8].bias_ref().clone(), layer_params[8].clone(),
    )));

    // Layer 9: ReLu
    qnet.add(Box::new(ReLuLayerQ4::new()));

    // Layer 10: Linear(84, 10)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[10].weights_ref());
    qnet.add(Box::new(LinearLayerQ4::new(
        84, 10, w_i4, w_params, layers[10].bias_ref().clone(), layer_params[10].clone(),
    )));

    // SoftMax handled internally by QuantizedNeuralNetworkI4

    qnet
}

/// Build an INT4 quantized LeNet-5 with Conv2d+ReLU fusion from a calibrated FP32 network.
///
/// Uses the non-fused FP32 network for weight extraction (layer indices 0, 3, 6, 8, 10).
pub fn lenet_quantized_i4_fused(
    fp32_net: &NeuralNetwork,
    input_params: QuantParams,
    layer_params: &[QuantParams],
) -> QuantizedNeuralNetworkI4 {
    let layers = fp32_net.layers();
    let mut qnet = QuantizedNeuralNetworkI4::new(input_params);

    // Conv2d+ReLu(1, 6, 5, 1, 2) — fused
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[0].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ4::new_fused(
        1, 6, 5, 1, 2, w_i4, w_params, layers[0].bias_ref().clone(), layer_params[0].clone(), true,
    )));

    // MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ4::new(2, 2, 0)));

    // Conv2d+ReLu(6, 16, 5, 1, 0) — fused
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[3].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ4::new_fused(
        6, 16, 5, 1, 0, w_i4, w_params, layers[3].bias_ref().clone(), layer_params[3].clone(), true,
    )));

    // MaxPool2d(2, 2, 0)
    qnet.add(Box::new(MaxPool2dLayerQ4::new(2, 2, 0)));

    // Conv2d+ReLu(16, 120, 5, 1, 0) — fused
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[6].weights_ref());
    qnet.add(Box::new(Conv2dLayerQ4::new_fused(
        16, 120, 5, 1, 0, w_i4, w_params, layers[6].bias_ref().clone(), layer_params[6].clone(), true,
    )));

    // Linear(120, 84)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[8].weights_ref());
    qnet.add(Box::new(LinearLayerQ4::new(
        120, 84, w_i4, w_params, layers[8].bias_ref().clone(), layer_params[8].clone(),
    )));

    // ReLu
    qnet.add(Box::new(ReLuLayerQ4::new()));

    // Linear(84, 10)
    let (w_i4, w_params) = quantize_tensor_symmetric_i4(layers[10].weights_ref());
    qnet.add(Box::new(LinearLayerQ4::new(
        84, 10, w_i4, w_params, layers[10].bias_ref().clone(), layer_params[10].clone(),
    )));

    qnet
}
