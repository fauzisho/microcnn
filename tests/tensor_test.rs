use microcnn::tensor::Tensor;
use microcnn::tensor_i8::TensorI8;
use microcnn::tensor_i4::TensorI4;

#[test]
fn tensor_get_set() {
    let t = Tensor::new(1, 2, 3, 4);
    t.set(0, 1, 2, 3, 42.0);
    assert_eq!(t.get(0, 1, 2, 3), 42.0);
}

#[test]
fn tensor_fill() {
    let t = Tensor::new(1, 1, 2, 2);
    t.fill(7.0);
    assert_eq!(t.get(0, 0, 0, 0), 7.0);
    assert_eq!(t.get(0, 0, 1, 1), 7.0);
}

#[test]
fn tensor_slice_shares_data() {
    let t = Tensor::new(4, 1, 1, 1);
    t.set(0, 0, 0, 0, 10.0);
    t.set(1, 0, 0, 0, 20.0);
    t.set(2, 0, 0, 0, 30.0);

    let s = t.slice(1, 2);
    assert_eq!(s.n, 2);
    assert_eq!(s.get(0, 0, 0, 0), 20.0);
    assert_eq!(s.get(1, 0, 0, 0), 30.0);

    // Mutation through slice is visible in original
    s.set(0, 0, 0, 0, 99.0);
    assert_eq!(t.get(1, 0, 0, 0), 99.0);
}

#[test]
fn tensor_empty() {
    let t = Tensor::empty();
    assert!(t.is_empty());
}

#[test]
fn tensor_i8_get_set() {
    let mut t = TensorI8::new(1, 1, 2, 2);
    t.set(0, 0, 0, 0, 127);
    t.set(0, 0, 1, 1, -128);
    assert_eq!(t.get(0, 0, 0, 0), 127);
    assert_eq!(t.get(0, 0, 1, 1), -128);
}

#[test]
fn tensor_i8_fill() {
    let mut t = TensorI8::new(1, 1, 2, 2);
    t.fill(42);
    assert_eq!(t.get(0, 0, 0, 0), 42);
    assert_eq!(t.get(0, 0, 1, 1), 42);
}

#[test]
fn tensor_i4_pack_unpack_roundtrip() {
    let mut t = TensorI4::new(1, 1, 1, 8);
    let vals: [i8; 8] = [-8, -7, -1, 0, 1, 5, 6, 7];
    for (i, &v) in vals.iter().enumerate() {
        t.set(0, 0, 0, i, v);
    }
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(t.get(0, 0, 0, i), v, "mismatch at index {}", i);
    }
}

#[test]
fn tensor_i4_clamping() {
    let mut t = TensorI4::new(1, 1, 1, 2);
    // Values outside [-8, 7] should be clamped
    t.set(0, 0, 0, 0, -10);
    t.set(0, 0, 0, 1, 10);
    assert_eq!(t.get(0, 0, 0, 0), -8);
    assert_eq!(t.get(0, 0, 0, 1), 7);
}

#[test]
fn tensor_i4_fill() {
    let mut t = TensorI4::new(1, 1, 1, 4);
    t.fill(3);
    for i in 0..4 {
        assert_eq!(t.get(0, 0, 0, i), 3);
    }
    t.fill(-5);
    for i in 0..4 {
        assert_eq!(t.get(0, 0, 0, i), -5);
    }
}

#[test]
fn tensor_i4_odd_element_count() {
    // Odd number of elements: make sure the last nibble works
    let mut t = TensorI4::new(1, 1, 1, 3);
    t.set(0, 0, 0, 0, 1);
    t.set(0, 0, 0, 1, -3);
    t.set(0, 0, 0, 2, 7);
    assert_eq!(t.get(0, 0, 0, 0), 1);
    assert_eq!(t.get(0, 0, 0, 1), -3);
    assert_eq!(t.get(0, 0, 0, 2), 7);
}
