#[cfg(feature = "std")]
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::tensor::Tensor;
use crate::tensor_i4::TensorI4;
use crate::quantization::{QuantParams, quantize_tensor_asymmetric_i4, dequantize_tensor_i4};
use super::{LayerType, Layer, SoftMaxLayer};

fn pad_tensor_i4(input: &TensorI4, pad: usize, c: i8) -> TensorI4 {
    if pad == 0 {
        return input.clone();
    }
    let mut padded = TensorI4::new(input.n, input.c, input.h + 2 * pad, input.w + 2 * pad);
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

/// Trait for INT4 quantized layers.
///
/// Each layer performs forward inference in INT4 arithmetic, taking quantized
/// input and returning quantized output along with quantization parameters.
pub trait QuantizedLayerI4 {
    fn layer_type(&self) -> LayerType;
    fn fwd_i4(&self, input: &TensorI4, input_params: &QuantParams) -> (TensorI4, QuantParams);
    fn weight_memory_bytes(&self) -> usize;
}

/// INT4 quantized 2D convolution layer.
pub struct Conv2dLayerQ4 {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    relu: bool,
    weights_i4: TensorI4,
    weight_params: QuantParams,
    bias: Tensor,
    output_params: QuantParams,
}

impl Conv2dLayerQ4 {
    pub fn new(
        in_channels: usize, out_channels: usize, kernel_size: usize,
        stride: usize, pad: usize,
        weights_i4: TensorI4, weight_params: QuantParams,
        bias: Tensor, output_params: QuantParams,
    ) -> Self {
        Conv2dLayerQ4 {
            in_channels, out_channels, kernel_size, stride, pad,
            relu: false,
            weights_i4, weight_params, bias, output_params,
        }
    }

    pub fn new_fused(
        in_channels: usize, out_channels: usize, kernel_size: usize,
        stride: usize, pad: usize,
        weights_i4: TensorI4, weight_params: QuantParams,
        bias: Tensor, output_params: QuantParams,
        relu: bool,
    ) -> Self {
        Conv2dLayerQ4 {
            in_channels, out_channels, kernel_size, stride, pad,
            relu,
            weights_i4, weight_params, bias, output_params,
        }
    }
}

impl QuantizedLayerI4 for Conv2dLayerQ4 {
    fn layer_type(&self) -> LayerType { if self.relu { LayerType::Conv2dReLu } else { LayerType::Conv2d } }

    fn weight_memory_bytes(&self) -> usize {
        self.weights_i4.memory_bytes() + self.out_channels * 4
    }

    fn fwd_i4(&self, input: &TensorI4, input_params: &QuantParams) -> (TensorI4, QuantParams) {
        let output_height = (input.h + 2 * self.pad - self.kernel_size) / self.stride + 1;
        let output_width = (input.w + 2 * self.pad - self.kernel_size) / self.stride + 1;

        let padded = pad_tensor_i4(input, self.pad, input_params.zero_point.clamp(-8, 7) as i8);
        let mut output = TensorI4::new(input.n, self.out_channels, output_height, output_width);

        let combined_scale = input_params.scale * self.weight_params.scale;
        let input_zp = input_params.zero_point;
        let lo = if self.relu { self.output_params.zero_point.clamp(-8, 7) } else { -8 };

        for n in 0..input.n {
            for oc in 0..self.out_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum: i32 = 0;
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    let inp = padded.get(n, ic, oh * self.stride + kh, ow * self.stride + kw) as i32;
                                    let wt = self.weights_i4.get(oc, ic, kh, kw) as i32;
                                    sum += (inp - input_zp) * wt;
                                }
                            }
                        }
                        let result = sum as f32 * combined_scale + self.bias.get(oc, 0, 0, 0);
                        let quantized = libm::roundf(result / self.output_params.scale) as i32 + self.output_params.zero_point;
                        output.set(n, oc, oh, ow, quantized.clamp(lo, 7) as i8);
                    }
                }
            }
        }
        (output, self.output_params.clone())
    }
}

/// INT4 quantized fully-connected (linear) layer.
pub struct LinearLayerQ4 {
    in_features: usize,
    out_features: usize,
    weights_i4: TensorI4,
    weight_params: QuantParams,
    bias: Tensor,
    output_params: QuantParams,
}

impl LinearLayerQ4 {
    pub fn new(
        in_features: usize, out_features: usize,
        weights_i4: TensorI4, weight_params: QuantParams,
        bias: Tensor, output_params: QuantParams,
    ) -> Self {
        LinearLayerQ4 {
            in_features, out_features,
            weights_i4, weight_params, bias, output_params,
        }
    }
}

impl QuantizedLayerI4 for LinearLayerQ4 {
    fn layer_type(&self) -> LayerType { LayerType::Linear }

    fn weight_memory_bytes(&self) -> usize {
        self.weights_i4.memory_bytes() + self.out_features * 4
    }

    fn fwd_i4(&self, input: &TensorI4, input_params: &QuantParams) -> (TensorI4, QuantParams) {
        let mut output = TensorI4::new2(input.n, self.out_features);
        let combined_scale = input_params.scale * self.weight_params.scale;
        let input_zp = input_params.zero_point;

        for n in 0..input.n {
            for o in 0..self.out_features {
                let mut sum: i32 = 0;
                for i in 0..self.in_features {
                    let inp = input.get(n, i, 0, 0) as i32;
                    let wt = self.weights_i4.get(o, i, 0, 0) as i32;
                    sum += (inp - input_zp) * wt;
                }
                let result = sum as f32 * combined_scale + self.bias.get(o, 0, 0, 0);
                let quantized = libm::roundf(result / self.output_params.scale) as i32 + self.output_params.zero_point;
                output.set(n, o, 0, 0, quantized.clamp(-8, 7) as i8);
            }
        }
        (output, self.output_params.clone())
    }
}

/// INT4 quantized ReLU activation layer.
pub struct ReLuLayerQ4;

impl ReLuLayerQ4 {
    pub fn new() -> Self { ReLuLayerQ4 }
}

impl QuantizedLayerI4 for ReLuLayerQ4 {
    fn layer_type(&self) -> LayerType { LayerType::ReLu }
    fn weight_memory_bytes(&self) -> usize { 0 }

    fn fwd_i4(&self, input: &TensorI4, input_params: &QuantParams) -> (TensorI4, QuantParams) {
        let mut output = TensorI4::new(input.n, input.c, input.h, input.w);
        let zp = input_params.zero_point.clamp(-8, 7) as i8;
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

/// INT4 quantized 2D max pooling layer.
pub struct MaxPool2dLayerQ4 {
    kernel_size: usize,
    stride: usize,
    pad: usize,
}

impl MaxPool2dLayerQ4 {
    pub fn new(kernel_size: usize, stride: usize, pad: usize) -> Self {
        MaxPool2dLayerQ4 { kernel_size, stride, pad }
    }
}

impl QuantizedLayerI4 for MaxPool2dLayerQ4 {
    fn layer_type(&self) -> LayerType { LayerType::MaxPool2d }
    fn weight_memory_bytes(&self) -> usize { 0 }

    fn fwd_i4(&self, input: &TensorI4, input_params: &QuantParams) -> (TensorI4, QuantParams) {
        let output_height = (input.h + 2 * self.pad - self.kernel_size) / self.stride + 1;
        let output_width = (input.w + 2 * self.pad - self.kernel_size) / self.stride + 1;

        let padded = pad_tensor_i4(input, self.pad, -8);
        let mut output = TensorI4::new(input.n, input.c, output_height, output_width);

        for n in 0..input.n {
            for c in 0..input.c {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut max_val: i8 = -8;
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

/// INT4 quantized flatten layer.
pub struct FlattenLayerQ4;

impl FlattenLayerQ4 {
    pub fn new() -> Self { FlattenLayerQ4 }
}

impl QuantizedLayerI4 for FlattenLayerQ4 {
    fn layer_type(&self) -> LayerType { LayerType::Flatten }
    fn weight_memory_bytes(&self) -> usize { 0 }

    fn fwd_i4(&self, input: &TensorI4, input_params: &QuantParams) -> (TensorI4, QuantParams) {
        let mut output = TensorI4::new2(input.n, input.c * input.h * input.w);
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

/// An INT4 quantized neural network.
///
/// Quantizes FP32 input using the provided input parameters, runs all layers
/// in INT4 arithmetic, then dequantizes and applies softmax in FP32.
pub struct QuantizedNeuralNetworkI4 {
    layers: Vec<Box<dyn QuantizedLayerI4>>,
    input_params: QuantParams,
}

impl QuantizedNeuralNetworkI4 {
    pub fn new(input_params: QuantParams) -> Self {
        QuantizedNeuralNetworkI4 {
            layers: Vec::new(),
            input_params,
        }
    }

    pub fn add(&mut self, layer: Box<dyn QuantizedLayerI4>) {
        self.layers.push(layer);
    }

    pub fn predict(&self, input: &Tensor) -> Tensor {
        let input_i4 = quantize_tensor_asymmetric_i4(input, &self.input_params);
        let mut cur = input_i4;
        let mut cur_params = self.input_params.clone();

        for layer in &self.layers {
            let (out, out_params) = layer.fwd_i4(&cur, &cur_params);
            cur = out;
            cur_params = out_params;
        }

        let fp32 = dequantize_tensor_i4(&cur, &cur_params);
        let mut softmax = SoftMaxLayer::new();
        softmax.set_input(fp32);
        softmax.fwd();
        softmax.output().clone()
    }

    #[cfg(feature = "std")]
    pub fn predict_timed(&self, input: &Tensor) -> (Vec<Tensor>, Vec<(LayerType, Duration)>) {
        let mut timings = Vec::new();
        let mut intermediates = Vec::new();

        let input_i4 = quantize_tensor_asymmetric_i4(input, &self.input_params);
        let mut cur = input_i4;
        let mut cur_params = self.input_params.clone();

        for layer in &self.layers {
            let start = Instant::now();
            let (out, out_params) = layer.fwd_i4(&cur, &cur_params);
            timings.push((layer.layer_type(), start.elapsed()));
            intermediates.push(dequantize_tensor_i4(&out, &out_params));
            cur = out;
            cur_params = out_params;
        }

        let fp32 = dequantize_tensor_i4(&cur, &cur_params);
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
