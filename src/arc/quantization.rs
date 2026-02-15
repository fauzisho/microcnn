use crate::tensor::Tensor;
use crate::tensor_i8::TensorI8;
use crate::tensor_i4::TensorI4;

/// Parameters for affine quantization: `real_value = scale * (quantized_value - zero_point)`.
#[derive(Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantParams {
    pub fn symmetric(max_abs: f32) -> Self {
        let scale = if max_abs < 1e-10 { 1.0 } else { max_abs / 127.0 };
        QuantParams { scale, zero_point: 0 }
    }

    pub fn asymmetric(min_val: f32, max_val: f32) -> Self {
        let range = max_val - min_val;
        let scale = if range < 1e-10 { 1.0 } else { range / 255.0 };
        let zero_point = ((-128.0f32 - min_val / scale).round() as i32).clamp(-128, 127);
        QuantParams { scale, zero_point }
    }

    pub fn symmetric_i4(max_abs: f32) -> Self {
        let scale = if max_abs < 1e-10 { 1.0 } else { max_abs / 7.0 };
        QuantParams { scale, zero_point: 0 }
    }

    pub fn asymmetric_i4(min_val: f32, max_val: f32) -> Self {
        let range = max_val - min_val;
        let scale = if range < 1e-10 { 1.0 } else { range / 15.0 };
        let zero_point = ((-8.0f32 - min_val / scale).round() as i32).clamp(-8, 7);
        QuantParams { scale, zero_point }
    }
}

/// Quantize a FP32 tensor to INT8 using symmetric quantization (zero_point = 0).
pub fn quantize_tensor_symmetric(tensor: &Tensor) -> (TensorI8, QuantParams) {
    let mut max_abs: f32 = 0.0;
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    max_abs = max_abs.max(tensor.get(n, c, h, w).abs());
                }
            }
        }
    }
    let params = QuantParams::symmetric(max_abs);
    let mut out = TensorI8::new(tensor.n, tensor.c, tensor.h, tensor.w);
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    let val = (tensor.get(n, c, h, w) / params.scale).round() as i32;
                    out.set(n, c, h, w, val.clamp(-128, 127) as i8);
                }
            }
        }
    }
    (out, params)
}

/// Quantize a FP32 tensor to INT8 using pre-computed asymmetric parameters.
pub fn quantize_tensor_asymmetric(tensor: &Tensor, params: &QuantParams) -> TensorI8 {
    let mut out = TensorI8::new(tensor.n, tensor.c, tensor.h, tensor.w);
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    let val = (tensor.get(n, c, h, w) / params.scale).round() as i32 + params.zero_point;
                    out.set(n, c, h, w, val.clamp(-128, 127) as i8);
                }
            }
        }
    }
    out
}

/// Dequantize an INT8 tensor back to FP32.
pub fn dequantize_tensor(tensor: &TensorI8, params: &QuantParams) -> Tensor {
    let out = Tensor::new(tensor.n, tensor.c, tensor.h, tensor.w);
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    let val = (tensor.get(n, c, h, w) as i32 - params.zero_point) as f32 * params.scale;
                    out.set(n, c, h, w, val);
                }
            }
        }
    }
    out
}

/// Quantize a FP32 tensor to INT4 using symmetric quantization (zero_point = 0).
pub fn quantize_tensor_symmetric_i4(tensor: &Tensor) -> (TensorI4, QuantParams) {
    let mut max_abs: f32 = 0.0;
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    max_abs = max_abs.max(tensor.get(n, c, h, w).abs());
                }
            }
        }
    }
    let params = QuantParams::symmetric_i4(max_abs);
    let mut out = TensorI4::new(tensor.n, tensor.c, tensor.h, tensor.w);
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    let val = (tensor.get(n, c, h, w) / params.scale).round() as i32;
                    out.set(n, c, h, w, val.clamp(-8, 7) as i8);
                }
            }
        }
    }
    (out, params)
}

/// Quantize a FP32 tensor to INT4 using pre-computed asymmetric parameters.
pub fn quantize_tensor_asymmetric_i4(tensor: &Tensor, params: &QuantParams) -> TensorI4 {
    let mut out = TensorI4::new(tensor.n, tensor.c, tensor.h, tensor.w);
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    let val = (tensor.get(n, c, h, w) / params.scale).round() as i32 + params.zero_point;
                    out.set(n, c, h, w, val.clamp(-8, 7) as i8);
                }
            }
        }
    }
    out
}

/// Dequantize an INT4 tensor back to FP32.
pub fn dequantize_tensor_i4(tensor: &TensorI4, params: &QuantParams) -> Tensor {
    let out = Tensor::new(tensor.n, tensor.c, tensor.h, tensor.w);
    for n in 0..tensor.n {
        for c in 0..tensor.c {
            for h in 0..tensor.h {
                for w in 0..tensor.w {
                    let val = (tensor.get(n, c, h, w) as i32 - params.zero_point) as f32 * params.scale;
                    out.set(n, c, h, w, val);
                }
            }
        }
    }
    out
}

/// Collects min/max statistics over calibration samples to compute quantization parameters.
///
/// Run representative inputs through the FP32 network, observing each layer's output
/// range, then call [`Calibrator::layer_params`] or [`Calibrator::layer_params_i4`]
/// to obtain per-layer quantization parameters.
pub struct Calibrator {
    input_min: f32,
    input_max: f32,
    layer_mins: Vec<f32>,
    layer_maxs: Vec<f32>,
    num_samples: usize,
}

impl Calibrator {
    pub fn new(num_layers: usize) -> Self {
        Calibrator {
            input_min: f32::MAX,
            input_max: f32::MIN,
            layer_mins: vec![f32::MAX; num_layers],
            layer_maxs: vec![f32::MIN; num_layers],
            num_samples: 0,
        }
    }

    pub fn observe_input(&mut self, tensor: &Tensor) {
        for n in 0..tensor.n {
            for c in 0..tensor.c {
                for h in 0..tensor.h {
                    for w in 0..tensor.w {
                        let val = tensor.get(n, c, h, w);
                        self.input_min = self.input_min.min(val);
                        self.input_max = self.input_max.max(val);
                    }
                }
            }
        }
    }

    pub fn observe_layer(&mut self, layer_idx: usize, tensor: &Tensor) {
        for n in 0..tensor.n {
            for c in 0..tensor.c {
                for h in 0..tensor.h {
                    for w in 0..tensor.w {
                        let val = tensor.get(n, c, h, w);
                        self.layer_mins[layer_idx] = self.layer_mins[layer_idx].min(val);
                        self.layer_maxs[layer_idx] = self.layer_maxs[layer_idx].max(val);
                    }
                }
            }
        }
    }

    pub fn finish_sample(&mut self) {
        self.num_samples += 1;
    }

    pub fn input_params(&self) -> QuantParams {
        QuantParams::asymmetric(self.input_min, self.input_max)
    }

    pub fn layer_params(&self) -> Vec<QuantParams> {
        self.layer_mins
            .iter()
            .zip(self.layer_maxs.iter())
            .map(|(&min, &max)| QuantParams::asymmetric(min, max))
            .collect()
    }

    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn input_params_i4(&self) -> QuantParams {
        QuantParams::asymmetric_i4(self.input_min, self.input_max)
    }

    pub fn layer_params_i4(&self) -> Vec<QuantParams> {
        self.layer_mins
            .iter()
            .zip(self.layer_maxs.iter())
            .map(|(&min, &max)| QuantParams::asymmetric_i4(min, max))
            .collect()
    }
}
