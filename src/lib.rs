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

/// FP32 tensor with shared-memory support.
pub mod tensor;
/// INT8 tensor.
pub mod tensor_i8;
/// INT4 packed tensor (2 values per byte).
pub mod tensor_i4;
/// Quantization utilities: symmetric/asymmetric quantization, calibration.
pub mod quantization;
/// Neural network layers and network types (FP32, INT8, INT4).
pub mod network;
/// LeNet-5 model builders.
pub mod lenet;
/// MNIST dataset loaders.
pub mod mnist;
/// Benchmarking utilities for comparing FP32/INT8/INT4 performance.
pub mod benchmark;
