use std::fmt;

/// A 4D INT8 tensor for quantized inference.
///
/// Layout is NCHW. Each element is stored as a signed 8-bit integer.
#[derive(Clone)]
pub struct TensorI8 {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<i8>,
}

impl TensorI8 {
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        TensorI8 {
            n,
            c,
            h,
            w,
            data: vec![0i8; n * c * h * w],
        }
    }

    pub fn new1(n: usize) -> Self {
        Self::new(n, 1, 1, 1)
    }

    pub fn new2(n: usize, c: usize) -> Self {
        Self::new(n, c, 1, 1)
    }

    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> i8 {
        let idx = n * self.c * self.h * self.w + c * self.h * self.w + h * self.w + w;
        self.data[idx]
    }

    pub fn set(&mut self, n: usize, c: usize, h: usize, w: usize, val: i8) {
        let idx = n * self.c * self.h * self.w + c * self.h * self.w + h * self.w + w;
        self.data[idx] = val;
    }

    pub fn fill(&mut self, val: i8) {
        self.data.fill(val);
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Display for TensorI8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}x{} (i8)", self.n, self.c, self.h, self.w)
    }
}
