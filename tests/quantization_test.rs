use microcnn::tensor::Tensor;
use microcnn::quantization::*;

#[test]
fn symmetric_i8_roundtrip() {
    let t = Tensor::new(1, 1, 1, 4);
    t.set(0, 0, 0, 0, 0.0);
    t.set(0, 0, 0, 1, 1.0);
    t.set(0, 0, 0, 2, -1.0);
    t.set(0, 0, 0, 3, 0.5);

    let (quantized, params) = quantize_tensor_symmetric(&t);
    let dequantized = dequantize_tensor(&quantized, &params);

    for w in 0..4 {
        let orig = t.get(0, 0, 0, w);
        let restored = dequantized.get(0, 0, 0, w);
        assert!(
            (orig - restored).abs() < 0.02,
            "INT8 symmetric roundtrip error too large at index {}: orig={}, restored={}",
            w, orig, restored
        );
    }
}

#[test]
fn asymmetric_i8_roundtrip() {
    let t = Tensor::new(1, 1, 1, 4);
    t.set(0, 0, 0, 0, 0.0);
    t.set(0, 0, 0, 1, 0.5);
    t.set(0, 0, 0, 2, 0.8);
    t.set(0, 0, 0, 3, 1.0);

    let params = QuantParams::asymmetric(0.0, 1.0);
    let quantized = quantize_tensor_asymmetric(&t, &params);
    let dequantized = dequantize_tensor(&quantized, &params);

    for w in 0..4 {
        let orig = t.get(0, 0, 0, w);
        let restored = dequantized.get(0, 0, 0, w);
        assert!(
            (orig - restored).abs() < 0.02,
            "INT8 asymmetric roundtrip error too large at index {}: orig={}, restored={}",
            w, orig, restored
        );
    }
}

#[test]
fn symmetric_i4_roundtrip() {
    let t = Tensor::new(1, 1, 1, 4);
    t.set(0, 0, 0, 0, 0.0);
    t.set(0, 0, 0, 1, 1.0);
    t.set(0, 0, 0, 2, -1.0);
    t.set(0, 0, 0, 3, 0.5);

    let (quantized, params) = quantize_tensor_symmetric_i4(&t);
    let dequantized = dequantize_tensor_i4(&quantized, &params);

    for w in 0..4 {
        let orig = t.get(0, 0, 0, w);
        let restored = dequantized.get(0, 0, 0, w);
        assert!(
            (orig - restored).abs() < 0.2,
            "INT4 symmetric roundtrip error too large at index {}: orig={}, restored={}",
            w, orig, restored
        );
    }
}

#[test]
fn asymmetric_i4_roundtrip() {
    let t = Tensor::new(1, 1, 1, 4);
    t.set(0, 0, 0, 0, 0.0);
    t.set(0, 0, 0, 1, 0.3);
    t.set(0, 0, 0, 2, 0.7);
    t.set(0, 0, 0, 3, 1.0);

    let params = QuantParams::asymmetric_i4(0.0, 1.0);
    let quantized = quantize_tensor_asymmetric_i4(&t, &params);
    let dequantized = dequantize_tensor_i4(&quantized, &params);

    for w in 0..4 {
        let orig = t.get(0, 0, 0, w);
        let restored = dequantized.get(0, 0, 0, w);
        assert!(
            (orig - restored).abs() < 0.15,
            "INT4 asymmetric roundtrip error too large at index {}: orig={}, restored={}",
            w, orig, restored
        );
    }
}

#[test]
fn quant_params_symmetric_zero_centered() {
    let params = QuantParams::symmetric(1.0);
    assert_eq!(params.zero_point, 0);
    assert!((params.scale - 1.0 / 127.0).abs() < 1e-6);
}

#[test]
fn quant_params_symmetric_i4_zero_centered() {
    let params = QuantParams::symmetric_i4(7.0);
    assert_eq!(params.zero_point, 0);
    assert!((params.scale - 1.0).abs() < 1e-6);
}
