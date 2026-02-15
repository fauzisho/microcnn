use std::time::{Duration, Instant};

use crate::tensor::Tensor;
use crate::tensor_i8::TensorI8;
use crate::quantization::{QuantParams, quantize_tensor_asymmetric, dequantize_tensor};
use super::{LayerType, Layer, SoftMaxLayer};

fn pad_tensor_i8(input: &TensorI8, pad: usize, c: i8) -> TensorI8 {
    if pad == 0 {
        return input.clone();
    }
    let mut padded = TensorI8::new(input.n, input.c, input.h + 2 * pad, input.w + 2 * pad);
    padded.fill(c);
    for n in 0..input.n {
        for ch in 0..input.c {
            for h in 0..input.h {
                for w in 0..input.w {
                    padded.set(n, ch, h + pad, w + pad, input.get(n, ch, h, w));
                }
            }
        }
    }
    padded
}

/// Trait for INT8 quantized layers.
///
/// Each layer performs forward inference in INT8 arithmetic, taking quantized
/// input and returning quantized output along with quantization parameters.
pub trait QuantizedLayer {
    fn layer_type(&self) -> LayerType;
    fn fwd_i8(&self, input: &TensorI8, input_params: &QuantParams) -> (TensorI8, QuantParams);
    fn weight_memory_bytes(&self) -> usize;
}

/// INT8 quantized 2D convolution layer.
pub struct Conv2dLayerQ {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    weights_i8: TensorI8,
    weight_params: QuantParams,
    bias: Tensor,
    output_params: QuantParams,
}

impl Conv2dLayerQ {
    pub fn new(
        in_channels: usize, out_channels: usize, kernel_size: usize,
        stride: usize, pad: usize,
        weights_i8: TensorI8, weight_params: QuantParams,
        bias: Tensor, output_params: QuantParams,
    ) -> Self {
        Conv2dLayerQ {
            in_channels, out_channels, kernel_size, stride, pad,
            weights_i8, weight_params, bias, output_params,
        }
    }
}

impl QuantizedLayer for Conv2dLayerQ {
    fn layer_type(&self) -> LayerType { LayerType::Conv2d }

    fn weight_memory_bytes(&self) -> usize {
        self.weights_i8.memory_bytes() + self.out_channels * 4
    }

    fn fwd_i8(&self, input: &TensorI8, input_params: &QuantParams) -> (TensorI8, QuantParams) {
        let output_height = (input.h + 2 * self.pad - self.kernel_size) / self.stride + 1;
        let output_width = (input.w + 2 * self.pad - self.kernel_size) / self.stride + 1;

        let padded = pad_tensor_i8(input, self.pad, input_params.zero_point.clamp(-128, 127) as i8);
        let mut output = TensorI8::new(input.n, self.out_channels, output_height, output_width);

        let combined_scale = input_params.scale * self.weight_params.scale;
        let input_zp = input_params.zero_point;

        for n in 0..input.n {
            for oc in 0..self.out_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum: i32 = 0;
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let inp = padded.get(n, ic, oh * self.stride + kh, ow * self.stride + kw) as i32;
                                    let wt = self.weights_i8.get(oc, ic, kh, kw) as i32;
                                    sum += (inp - input_zp) * wt;
                                }
                            }
                        }
                        let result = sum as f32 * combined_scale + self.bias.get(oc, 0, 0, 0);
                        let quantized = (result / self.output_params.scale).round() as i32 + self.output_params.zero_point;
                        output.set(n, oc, oh, ow, quantized.clamp(-128, 127) as i8);
                    }
                }
            }
        }
        (output, self.output_params.clone())
    }
}

/// INT8 quantized fully-connected (linear) layer.
pub struct LinearLayerQ {
    in_features: usize,
    out_features: usize,
    weights_i8: TensorI8,
    weight_params: QuantParams,
    bias: Tensor,
    output_params: QuantParams,
}

impl LinearLayerQ {
    pub fn new(
        in_features: usize, out_features: usize,
        weights_i8: TensorI8, weight_params: QuantParams,
        bias: Tensor, output_params: QuantParams,
    ) -> Self {
        LinearLayerQ {
            in_features, out_features,
            weights_i8, weight_params, bias, output_params,
        }
    }
}

impl QuantizedLayer for LinearLayerQ {
    fn layer_type(&self) -> LayerType { LayerType::Linear }

    fn weight_memory_bytes(&self) -> usize {
        self.weights_i8.memory_bytes() + self.out_features * 4
    }

    fn fwd_i8(&self, input: &TensorI8, input_params: &QuantParams) -> (TensorI8, QuantParams) {
        let mut output = TensorI8::new2(input.n, self.out_features);
        let combined_scale = input_params.scale * self.weight_params.scale;
        let input_zp = input_params.zero_point;

        for n in 0..input.n {
            for o in 0..self.out_features {
                let mut sum: i32 = 0;
                for i in 0..self.in_features {
                    let inp = input.get(n, i, 0, 0) as i32;
                    let wt = self.weights_i8.get(o, i, 0, 0) as i32;
                    sum += (inp - input_zp) * wt;
                }
                let result = sum as f32 * combined_scale + self.bias.get(o, 0, 0, 0);
                let quantized = (result / self.output_params.scale).round() as i32 + self.output_params.zero_point;
                output.set(n, o, 0, 0, quantized.clamp(-128, 127) as i8);
            }
        }
        (output, self.output_params.clone())
    }
}

/// INT8 quantized ReLU activation layer.
pub struct ReLuLayerQ;

impl ReLuLayerQ {
    pub fn new() -> Self { ReLuLayerQ }
}

impl QuantizedLayer for ReLuLayerQ {
    fn layer_type(&self) -> LayerType { LayerType::ReLu }
    fn weight_memory_bytes(&self) -> usize { 0 }

    fn fwd_i8(&self, input: &TensorI8, input_params: &QuantParams) -> (TensorI8, QuantParams) {
        let mut output = TensorI8::new(input.n, input.c, input.h, input.w);
        let zp = input_params.zero_point.clamp(-128, 127) as i8;
        for n in 0..input.n {
            for c in 0..input.c {
                for h in 0..input.h {
                    for w in 0..input.w {
                        output.set(n, c, h, w, input.get(n, c, h, w).max(zp));
                    }
                }
            }
        }
        (output, input_params.clone())
    }
}

/// INT8 quantized 2D max pooling layer.
pub struct MaxPool2dLayerQ {
    kernel_size: usize,
    stride: usize,
    pad: usize,
}

impl MaxPool2dLayerQ {
    pub fn new(kernel_size: usize, stride: usize, pad: usize) -> Self {
        MaxPool2dLayerQ { kernel_size, stride, pad }
    }
}

impl QuantizedLayer for MaxPool2dLayerQ {
    fn layer_type(&self) -> LayerType { LayerType::MaxPool2d }
    fn weight_memory_bytes(&self) -> usize { 0 }

    fn fwd_i8(&self, input: &TensorI8, input_params: &QuantParams) -> (TensorI8, QuantParams) {
        let output_height = (input.h + 2 * self.pad - self.kernel_size) / self.stride + 1;
        let output_width = (input.w + 2 * self.pad - self.kernel_size) / self.stride + 1;

        let padded = pad_tensor_i8(input, self.pad, i8::MIN);
        let mut output = TensorI8::new(input.n, input.c, output_height, output_width);

        for n in 0..input.n {
            for c in 0..input.c {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut max_val = i8::MIN;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                max_val = max_val.max(padded.get(n, c, h * self.stride + kh, w * self.stride + kw));
                            }
                        }
                        output.set(n, c, h, w, max_val);
                    }
                }
            }
        }
        (output, input_params.clone())
    }
}

/// INT8 quantized flatten layer.
pub struct FlattenLayerQ;

impl FlattenLayerQ {
    pub fn new() -> Self { FlattenLayerQ }
}

impl QuantizedLayer for FlattenLayerQ {
    fn layer_type(&self) -> LayerType { LayerType::Flatten }
    fn weight_memory_bytes(&self) -> usize { 0 }

    fn fwd_i8(&self, input: &TensorI8, input_params: &QuantParams) -> (TensorI8, QuantParams) {
        let mut output = TensorI8::new2(input.n, input.c * input.h * input.w);
        for n in 0..input.n {
            for c in 0..input.c {
                for h in 0..input.h {
                    for w in 0..input.w {
                        let idx = input.h * input.w * c + input.w * h + w;
                        output.set(n, idx, 0, 0, input.get(n, c, h, w));
                    }
                }
            }
        }
        (output, input_params.clone())
    }
}

/// An INT8 quantized neural network.
///
/// Quantizes FP32 input using the provided input parameters, runs all layers
/// in INT8 arithmetic, then dequantizes and applies softmax in FP32.
pub struct QuantizedNeuralNetwork {
    layers: Vec<Box<dyn QuantizedLayer>>,
    input_params: QuantParams,
}

impl QuantizedNeuralNetwork {
    pub fn new(input_params: QuantParams) -> Self {
        QuantizedNeuralNetwork {
            layers: Vec::new(),
            input_params,
        }
    }

    pub fn add(&mut self, layer: Box<dyn QuantizedLayer>) {
        self.layers.push(layer);
    }

    pub fn predict(&self, input: &Tensor) -> Tensor {
        let input_i8 = quantize_tensor_asymmetric(input, &self.input_params);
        let mut cur = input_i8;
        let mut cur_params = self.input_params.clone();

        for layer in &self.layers {
            let (out, out_params) = layer.fwd_i8(&cur, &cur_params);
            cur = out;
            cur_params = out_params;
        }

        let fp32 = dequantize_tensor(&cur, &cur_params);
        let mut softmax = SoftMaxLayer::new();
        softmax.set_input(fp32);
        softmax.fwd();
        softmax.output().clone()
    }

    pub fn predict_timed(&self, input: &Tensor) -> (Vec<Tensor>, Vec<(LayerType, Duration)>) {
        let mut timings = Vec::new();
        let mut intermediates = Vec::new();

        let input_i8 = quantize_tensor_asymmetric(input, &self.input_params);
        let mut cur = input_i8;
        let mut cur_params = self.input_params.clone();

        for layer in &self.layers {
            let start = Instant::now();
            let (out, out_params) = layer.fwd_i8(&cur, &cur_params);
            timings.push((layer.layer_type(), start.elapsed()));
            intermediates.push(dequantize_tensor(&out, &out_params));
            cur = out;
            cur_params = out_params;
        }

        let fp32 = dequantize_tensor(&cur, &cur_params);
        let mut softmax = SoftMaxLayer::new();
        softmax.set_input(fp32);
        let start = Instant::now();
        softmax.fwd();
        timings.push((LayerType::SoftMax, start.elapsed()));
        intermediates.push(softmax.output().clone());

        (intermediates, timings)
    }

    pub fn layer_weight_memory(&self) -> Vec<usize> {
        let mut mem: Vec<usize> = self.layers.iter().map(|l| l.weight_memory_bytes()).collect();
        mem.push(0); // SoftMax
        mem
    }
}
