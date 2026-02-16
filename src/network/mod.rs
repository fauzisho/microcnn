/// FP32 layer implementations (Conv2d, Linear, MaxPool2d, ReLu, SoftMax, Flatten).
mod layers;
/// INT8 quantized layer implementations.
mod layers_i8;
/// INT4 quantized layer implementations.
mod layers_i4;

pub use layers::*;
pub use layers_i8::*;
pub use layers_i4::*;

use core::fmt;
use alloc::boxed::Box;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

use crate::tensor::Tensor;

/// Identifies the type of a neural network layer.
#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    Conv2d,
    Conv2dReLu,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten,
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerType::Conv2d => write!(f, "Conv2d"),
            LayerType::Conv2dReLu => write!(f, "Conv2d+ReLu"),
            LayerType::Linear => write!(f, "Linear"),
            LayerType::MaxPool2d => write!(f, "MaxPool2d"),
            LayerType::ReLu => write!(f, "ReLu"),
            LayerType::SoftMax => write!(f, "SoftMax"),
            LayerType::Flatten => write!(f, "Flatten"),
        }
    }
}

/// Pads a tensor with a constant value on all spatial dimensions.
pub(crate) fn pad_tensor(input: &Tensor, pad: usize, c: f32) -> Tensor {
    if pad == 0 {
        return input.clone();
    }
    let padded = Tensor::new(input.n, input.c, input.h + 2 * pad, input.w + 2 * pad);
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

/// Trait for FP32 neural network layers.
///
/// Each layer can read weights from a byte slice, accept input, perform a forward pass,
/// and expose its output.
pub trait Layer {
    fn layer_type(&self) -> LayerType;
    fn fwd(&mut self);
    fn read_weights_bias(&mut self, data: &[u8], offset: &mut usize);
    fn set_input(&mut self, input: Tensor);
    fn output(&self) -> &Tensor;

    fn print(&self) {
        #[cfg(feature = "std")]
        {
            std::println!("{}", self.layer_type());
            if !self.input_ref().is_empty() {
                std::println!("  input: {}", self.input_ref());
            }
            if !self.weights_ref().is_empty() {
                std::println!("  weights: {}", self.weights_ref());
            }
            if !self.bias_ref().is_empty() {
                std::println!("  bias: {}", self.bias_ref());
            }
            if !self.output().is_empty() {
                std::println!("  output: {}", self.output());
            }
        }
    }

    fn input_ref(&self) -> &Tensor;
    fn weights_ref(&self) -> &Tensor;
    fn bias_ref(&self) -> &Tensor;
}

macro_rules! impl_layer_common {
    () => {
        fn set_input(&mut self, input: Tensor) {
            self.input = input;
        }

        fn output(&self) -> &Tensor {
            &self.output
        }

        fn input_ref(&self) -> &Tensor {
            &self.input
        }

        fn weights_ref(&self) -> &Tensor {
            &self.weights
        }

        fn bias_ref(&self) -> &Tensor {
            &self.bias
        }
    };
}

pub(crate) use impl_layer_common;

/// Read `count` little-endian f32 values from `data` starting at `*offset`,
/// writing them into `tensor` in NCHW order. Advances `*offset`.
pub(crate) fn read_floats(data: &[u8], offset: &mut usize, tensor: &Tensor, count: usize) {
    for i in 0..count {
        let pos = *offset + i * 4;
        let bytes = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
        let val = f32::from_le_bytes(bytes);
        let n_size = tensor.c * tensor.h * tensor.w;
        let c_size = tensor.h * tensor.w;
        let h_size = tensor.w;
        let tn = i / n_size;
        let tc = (i % n_size) / c_size;
        let th = (i % c_size) / h_size;
        let tw = i % h_size;
        tensor.set(tn, tc, th, tw, val);
    }
    *offset += count * 4;
}

/// A FP32 neural network composed of sequential layers.
///
/// Supports loading weights from a binary file or byte slice and running forward inference.
pub struct NeuralNetwork {
    debug: bool,
    layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new(debug: bool) -> Self {
        NeuralNetwork {
            debug,
            layers: Vec::new(),
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    #[cfg(feature = "std")]
    pub fn load(&mut self, path: &str) {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => {
                if self.debug {
                    std::eprintln!("Error: Failed to open weights file: {}", path);
                }
                return;
            }
        };

        let mut buf = Vec::new();
        if let Err(_) = file.read_to_end(&mut buf) {
            if self.debug {
                std::eprintln!("Error: Failed to read weights file: {}", path);
            }
            return;
        }

        self.load_from_bytes(&buf);
    }

    /// Load weights from a byte slice. Works in both `std` and `no_std` environments.
    pub fn load_from_bytes(&mut self, data: &[u8]) {
        let mut offset = 0usize;
        for layer in self.layers.iter_mut() {
            layer.read_weights_bias(data, &mut offset);
            #[cfg(feature = "std")]
            if self.debug {
                std::println!("Loaded weights for layer: {}", layer.layer_type());
            }
        }
    }

    pub fn predict(&mut self, input: Tensor) -> Tensor {
        if input.is_empty() {
            #[cfg(feature = "std")]
            if self.debug {
                std::eprintln!("Error: Predict received empty input tensor.");
            }
            return Tensor::empty();
        }

        let mut cur = input;
        for layer in self.layers.iter_mut() {
            layer.set_input(cur);
            layer.fwd();
            if layer.output().is_empty() {
                #[cfg(feature = "std")]
                if self.debug {
                    std::eprintln!("Error: Layer {} produced empty output.", layer.layer_type());
                }
                return Tensor::empty();
            }
            if self.debug {
                layer.print();
            }
            cur = layer.output().clone();
        }
        cur
    }

    pub fn predict_with_intermediates(&mut self, input: Tensor) -> Vec<Tensor> {
        let mut intermediates = Vec::new();
        let mut cur = input;
        for layer in self.layers.iter_mut() {
            layer.set_input(cur);
            layer.fwd();
            cur = layer.output().clone();
            intermediates.push(cur.clone());
        }
        intermediates
    }

    #[cfg(feature = "std")]
    pub fn predict_timed(&mut self, input: Tensor) -> (Vec<Tensor>, Vec<(LayerType, Duration)>) {
        let mut timings = Vec::new();
        let mut intermediates = Vec::new();
        let mut cur = input;
        for layer in self.layers.iter_mut() {
            layer.set_input(cur);
            let start = Instant::now();
            layer.fwd();
            timings.push((layer.layer_type(), start.elapsed()));
            cur = layer.output().clone();
            intermediates.push(cur.clone());
        }
        (intermediates, timings)
    }

    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_weight_memory(&self) -> Vec<usize> {
        self.layers.iter().map(|l| {
            let w = l.weights_ref();
            let b = l.bias_ref();
            (w.n * w.c * w.h * w.w + b.n * b.c * b.h * b.w) * 4
        }).collect()
    }
}
