//! A minimal CNN framework in Rust with INT8 and INT4 quantization support.
//!
//! This crate provides building blocks for constructing and running convolutional neural
//! networks in FP32, INT8, and INT4 precision. It includes a reference LeNet-5 implementation
//! for MNIST digit classification.
//!
//! # Example
//!
//! ```no_run
//! use microcnn::lenet::lenet;
//!
//! let mut net = lenet(false);
//! net.load("data/lenet.raw");
//! ```

/// FP32, INT8, and INT4 tensors.
pub mod tensor;
/// Re-export tensor types at legacy paths.
pub use tensor::TensorI8;
pub use tensor::TensorI4;
/// Alias modules so `crate::tensor_i8::TensorI8` etc. still resolve.
pub mod tensor_i8 {
    pub use crate::tensor::TensorI8;
}
pub mod tensor_i4 {
    pub use crate::tensor::TensorI4;
}

/// Quantization utilities and model architecture (LeNet).
pub mod arc;
/// Re-export at legacy paths.
pub mod quantization {
    pub use crate::arc::QuantParams;
    pub use crate::arc::Calibrator;
    pub use crate::arc::quantize_tensor_symmetric;
    pub use crate::arc::quantize_tensor_asymmetric;
    pub use crate::arc::dequantize_tensor;
    pub use crate::arc::quantize_tensor_symmetric_i4;
    pub use crate::arc::quantize_tensor_asymmetric_i4;
    pub use crate::arc::dequantize_tensor_i4;
}
pub mod lenet {
    pub use crate::arc::lenet;
    pub use crate::arc::lenet_with_algorithm;
    pub use crate::arc::lenet_quantized;
    pub use crate::arc::lenet_quantized_i4;
    pub use crate::arc::lenet_fused;
    pub use crate::arc::lenet_fused_with_algorithm;
    pub use crate::arc::lenet_quantized_fused;
    pub use crate::arc::lenet_quantized_i4_fused;
}

/// Neural network layers and network types (FP32, INT8, INT4).
pub mod network;

/// MNIST dataset loaders.
pub mod loader;
pub mod mnist {
    pub use crate::loader::MNIST;
    pub use crate::loader::MNISTLabels;
}

/// Convolution algorithm implementations (Naive, Im2col, Winograd, FFT).
pub mod conv;

/// Benchmarking utilities for comparing FP32/INT8/INT4 performance.
pub mod metrics;
pub mod benchmark {
    pub use crate::metrics::*;
}
