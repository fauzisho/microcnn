use core::fmt;
use alloc::vec;
use alloc::vec::Vec;

/// A 4D INT4 packed tensor for quantized inference.
///
/// Two signed 4-bit values (range [-8, 7]) are packed per byte.
/// Even-indexed elements use the lower nibble; odd-indexed elements use the upper nibble.
#[derive(Clone)]
pub struct TensorI4 {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<u8>,
    num_elements: usize,
}

impl TensorI4 {
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        let num_elements = n * c * h * w;
        let num_bytes = (num_elements + 1) / 2;
        TensorI4 {
            n,
            c,
            h,
            w,
            data: vec![0u8; num_bytes],
            num_elements,
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
        self.get_linear(idx)
    }

    pub fn set(&mut self, n: usize, c: usize, h: usize, w: usize, val: i8) {
        let idx = n * self.c * self.h * self.w + c * self.h * self.w + h * self.w + w;
        self.set_linear(idx, val);
    }

    fn get_linear(&self, idx: usize) -> i8 {
        let byte_idx = idx / 2;
        let byte = self.data[byte_idx];
        let nibble = if idx % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        // Sign-extend from 4 bits to i8
        if nibble & 0x08 != 0 {
            (nibble | 0xF0) as i8
        } else {
            nibble as i8
        }
    }

    fn set_linear(&mut self, idx: usize, val: i8) {
        let clamped = val.max(-8).min(7);
        let nibble = (clamped as u8) & 0x0F;
        let byte_idx = idx / 2;
        if idx % 2 == 0 {
            self.data[byte_idx] = (self.data[byte_idx] & 0xF0) | nibble;
        } else {
            self.data[byte_idx] = (self.data[byte_idx] & 0x0F) | (nibble << 4);
        }
    }

    pub fn fill(&mut self, val: i8) {
        let clamped = val.max(-8).min(7);
        let nibble = (clamped as u8) & 0x0F;
        let byte = nibble | (nibble << 4);
        self.data.fill(byte);
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    pub fn num_elements(&self) -> usize {
        self.num_elements
    }
}

impl fmt::Display for TensorI4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}x{} (i4)", self.n, self.c, self.h, self.w)
    }
}
