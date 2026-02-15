use std::fmt;
use std::sync::Arc;
use std::cell::RefCell;

/// A 4D floating-point tensor with shared-memory storage.
///
/// Layout is NCHW (batch, channels, height, width). Multiple tensors can share
/// the same underlying data via `Arc<RefCell<Vec<f32>>>`, enabling zero-copy slicing.
#[derive(Clone)]
pub struct Tensor {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    offset: usize,
    data: Arc<RefCell<Vec<f32>>>,
}

impl Tensor {
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        Tensor {
            n,
            c,
            h,
            w,
            offset: 0,
            data: Arc::new(RefCell::new(vec![0.0; n * c * h * w])),
        }
    }

    pub fn empty() -> Self {
        Tensor::new(0, 0, 0, 0)
    }

    pub fn new1(n: usize) -> Self {
        Tensor::new(n, 1, 1, 1)
    }

    pub fn new2(n: usize, c: usize) -> Self {
        Tensor::new(n, c, 1, 1)
    }

    pub fn new3(n: usize, c: usize, h: usize) -> Self {
        Tensor::new(n, c, h, 1)
    }

    pub fn from_shared(
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        offset: usize,
        data: Arc<RefCell<Vec<f32>>>,
    ) -> Self {
        Tensor {
            n,
            c,
            h,
            w,
            offset,
            data,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.borrow().is_empty()
    }

    pub fn data_ptr(&self) -> *mut f32 {
        self.data.borrow_mut().as_mut_ptr()
    }

    pub fn fill(&self, val: f32) {
        self.data.borrow_mut().fill(val);
    }

    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        let idx = self.offset + n * self.c * self.h * self.w + c * self.h * self.w + h * self.w + w;
        self.data.borrow()[idx]
    }

    pub fn set(&self, n: usize, c: usize, h: usize, w: usize, val: f32) {
        let idx = self.offset + n * self.c * self.h * self.w + c * self.h * self.w + h * self.w + w;
        self.data.borrow_mut()[idx] = val;
    }

    pub fn get_mut(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        self.get(n, c, h, w)
    }

    pub fn add(&self, n: usize, c: usize, h: usize, w: usize, val: f32) {
        let idx = self.offset + n * self.c * self.h * self.w + c * self.h * self.w + h * self.w + w;
        self.data.borrow_mut()[idx] += val;
    }

    pub fn slice(&self, idx: usize, num: usize) -> Tensor {
        let offset = self.offset + idx * self.c * self.h * self.w;
        Tensor::from_shared(num, self.c, self.h, self.w, offset, Arc::clone(&self.data))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}x{}x{}", self.n, self.c, self.h, self.w)
    }
}
