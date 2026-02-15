use std::fs::File;

use crate::tensor::Tensor;
use super::{LayerType, Layer, impl_layer_common, read_floats, pad_tensor};

/// 2D convolution layer (FP32).
pub struct Conv2dLayer {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    pad: usize,
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
}

impl Conv2dLayer {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, pad: usize) -> Self {
        Conv2dLayer {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad,
            input: Tensor::empty(),
            weights: Tensor::new(out_channels, in_channels, kernel_size, kernel_size),
            bias: Tensor::new1(out_channels),
            output: Tensor::empty(),
        }
    }
}

impl Layer for Conv2dLayer {
    fn layer_type(&self) -> LayerType { LayerType::Conv2d }
    impl_layer_common!();

    fn read_weights_bias(&mut self, file: &mut File) {
        let w_count = self.weights.n * self.weights.c * self.weights.h * self.weights.w;
        read_floats(file, &self.weights, w_count);
        read_floats(file, &self.bias, self.bias.n);
    }

    fn fwd(&mut self) {
        let output_height = (self.input.h + 2 * self.pad - self.kernel_size) / self.stride + 1;
        let output_width = (self.input.w + 2 * self.pad - self.kernel_size) / self.stride + 1;

        if output_height == 0 || output_width == 0 {
            eprintln!("Error: Invalid dimensions in Conv2d.");
            return;
        }

        self.output = Tensor::new(self.input.n, self.out_channels, output_height, output_width);
        let padded = pad_tensor(&self.input, self.pad, 0.0);

        for n in 0..self.input.n {
            for oc in 0..self.out_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = 0.0f32;
                        for ic in 0..self.in_channels {
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    sum += padded.get(n, ic, oh * self.stride + kh, ow * self.stride + kw)
                                        * self.weights.get(oc, ic, kh, kw);
                                }
                            }
                        }
                        self.output.set(n, oc, oh, ow, sum + self.bias.get(oc, 0, 0, 0));
                    }
                }
            }
        }
    }
}

/// Fully-connected (linear) layer (FP32).
pub struct LinearLayer {
    in_features: usize,
    out_features: usize,
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LinearLayer {
            in_features,
            out_features,
            input: Tensor::empty(),
            weights: Tensor::new2(out_features, in_features),
            bias: Tensor::new1(out_features),
            output: Tensor::empty(),
        }
    }
}

impl Layer for LinearLayer {
    fn layer_type(&self) -> LayerType { LayerType::Linear }
    impl_layer_common!();

    fn read_weights_bias(&mut self, file: &mut File) {
        read_floats(file, &self.weights, self.out_features * self.in_features);
        read_floats(file, &self.bias, self.out_features);
    }

    fn fwd(&mut self) {
        if self.input.is_empty() || self.input.c != self.in_features {
            eprintln!("Error: Linear layer received input with invalid dimensions.");
            return;
        }

        self.output = Tensor::new2(self.input.n, self.out_features);
        for n in 0..self.input.n {
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += self.input.get(n, i, 0, 0) * self.weights.get(o, i, 0, 0);
                }
                self.output.set(n, o, 0, 0, sum + self.bias.get(o, 0, 0, 0));
            }
        }
    }
}

/// 2D max pooling layer (FP32).
pub struct MaxPool2dLayer {
    kernel_size: usize,
    stride: usize,
    pad: usize,
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
}

impl MaxPool2dLayer {
    pub fn new(kernel_size: usize, stride: usize, pad: usize) -> Self {
        MaxPool2dLayer {
            kernel_size,
            stride,
            pad,
            input: Tensor::empty(),
            weights: Tensor::empty(),
            bias: Tensor::empty(),
            output: Tensor::empty(),
        }
    }
}

impl Layer for MaxPool2dLayer {
    fn layer_type(&self) -> LayerType { LayerType::MaxPool2d }
    impl_layer_common!();

    fn read_weights_bias(&mut self, _file: &mut File) {}

    fn fwd(&mut self) {
        let output_height = (self.input.h + 2 * self.pad - self.kernel_size) / self.stride + 1;
        let output_width = (self.input.w + 2 * self.pad - self.kernel_size) / self.stride + 1;

        if output_height == 0 || output_width == 0 {
            eprintln!("Error: Invalid dimensions in MaxPool2d.");
            return;
        }

        self.output = Tensor::new(self.input.n, self.input.c, output_height, output_width);
        let padded = pad_tensor(&self.input, self.pad, f32::MIN);

        for n in 0..self.input.n {
            for c in 0..self.input.c {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut max_val = f32::MIN;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                max_val = max_val.max(padded.get(n, c, h * self.stride + kh, w * self.stride + kw));
                            }
                        }
                        self.output.set(n, c, h, w, max_val);
                    }
                }
            }
        }
    }
}

/// ReLU activation layer (FP32).
pub struct ReLuLayer {
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
}

impl ReLuLayer {
    pub fn new() -> Self {
        ReLuLayer {
            input: Tensor::empty(),
            weights: Tensor::empty(),
            bias: Tensor::empty(),
            output: Tensor::empty(),
        }
    }
}

impl Layer for ReLuLayer {
    fn layer_type(&self) -> LayerType { LayerType::ReLu }
    impl_layer_common!();

    fn read_weights_bias(&mut self, _file: &mut File) {}

    fn fwd(&mut self) {
        if self.input.is_empty() || self.input.h == 0 || self.input.w == 0 {
            eprintln!("Error: ReLu received empty or invalid input tensor.");
            return;
        }

        self.output = Tensor::new(self.input.n, self.input.c, self.input.h, self.input.w);
        for n in 0..self.input.n {
            for c in 0..self.input.c {
                for h in 0..self.input.h {
                    for w in 0..self.input.w {
                        self.output.set(n, c, h, w, self.input.get(n, c, h, w).max(0.0));
                    }
                }
            }
        }
    }
}

/// Softmax activation layer (FP32).
pub struct SoftMaxLayer {
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
}

impl SoftMaxLayer {
    pub fn new() -> Self {
        SoftMaxLayer {
            input: Tensor::empty(),
            weights: Tensor::empty(),
            bias: Tensor::empty(),
            output: Tensor::empty(),
        }
    }
}

impl Layer for SoftMaxLayer {
    fn layer_type(&self) -> LayerType { LayerType::SoftMax }
    impl_layer_common!();

    fn read_weights_bias(&mut self, _file: &mut File) {}

    fn fwd(&mut self) {
        if self.input.is_empty() || self.input.c == 0 {
            eprintln!("Error: SoftMax received empty or invalid input tensor.");
            return;
        }

        self.output = Tensor::new2(self.input.n, self.input.c);
        for n in 0..self.input.n {
            let mut max_val = f32::MIN;
            for c in 0..self.input.c {
                max_val = max_val.max(self.input.get(n, c, 0, 0));
            }

            let mut sum = 0.0f32;
            for c in 0..self.input.c {
                let val = (self.input.get(n, c, 0, 0) - max_val).exp();
                self.output.set(n, c, 0, 0, val);
                sum += val;
            }
            for c in 0..self.input.c {
                let val = self.output.get(n, c, 0, 0) / sum;
                self.output.set(n, c, 0, 0, val);
            }
        }
    }
}

/// Flatten layer that reshapes 4D tensors into 2D (FP32).
pub struct FlattenLayer {
    input: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
}

impl FlattenLayer {
    pub fn new() -> Self {
        FlattenLayer {
            input: Tensor::empty(),
            weights: Tensor::empty(),
            bias: Tensor::empty(),
            output: Tensor::empty(),
        }
    }
}

impl Layer for FlattenLayer {
    fn layer_type(&self) -> LayerType { LayerType::Flatten }
    impl_layer_common!();

    fn read_weights_bias(&mut self, _file: &mut File) {}

    fn fwd(&mut self) {
        if self.output.is_empty() {
            self.output = Tensor::new2(self.input.n, self.input.c * self.input.h * self.input.w);
        }

        for n in 0..self.input.n {
            for c in 0..self.input.c {
                for h in 0..self.input.h {
                    for w in 0..self.input.w {
                        let idx = self.input.h * self.input.w * c + self.input.w * h + w;
                        self.output.set(n, idx, 0, 0, self.input.get(n, c, h, w));
                    }
                }
            }
        }
    }
}
