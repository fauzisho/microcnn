use std::fs::File;
use std::io::Read;
use std::boxed::Box;
use std::vec;
use std::vec::Vec;
use std::println;
use std::print;

use flate2::read::GzDecoder;

use crate::tensor::Tensor;

const PRE_PAD: usize = if cfg!(feature = "mnist_pre_pad") { 2 } else { 0 };

/// MNIST image dataset loader.
///
/// Reads the IDX3 image format and provides access to individual 28x28 grayscale images
/// as [`Tensor`] values normalized to [0, 1].
pub struct MNIST {
    imgs: Tensor,
    pad: usize,
}

impl MNIST {
    pub fn new(path: &str) -> Self {
        let mut mnist = MNIST {
            imgs: Tensor::empty(),
            pad: PRE_PAD,
        };
        mnist.load(path);
        mnist
    }

    fn load(&mut self, path: &str) {
        let mut file = File::open(path).expect("failed to open MNIST file");

        let mut buf4 = [0u8; 4];

        file.read_exact(&mut buf4).unwrap();
        let magic_number = u32::from_be_bytes(buf4);

        file.read_exact(&mut buf4).unwrap();
        let num_imgs = u32::from_be_bytes(buf4) as usize;

        file.read_exact(&mut buf4).unwrap();
        let num_rows = u32::from_be_bytes(buf4) as usize;

        file.read_exact(&mut buf4).unwrap();
        let num_cols = u32::from_be_bytes(buf4) as usize;

        assert!(magic_number == 0x00000803, "expected MNIST image file format");
        assert!(num_rows == 28 && num_cols == 28, "expected images of size 28x28");

        self.imgs = Tensor::new(num_imgs, 1, 28 + 2 * self.pad, 28 + 2 * self.pad);
        if self.pad > 0 {
            self.imgs.fill(0.0);
        }

        let mut byte_buf = [0u8; 1];
        for n in 0..num_imgs {
            for h in 0..28 {
                for w in 0..28 {
                    file.read_exact(&mut byte_buf).unwrap();
                    let pixel = byte_buf[0] as f32 / 255.0;
                    self.imgs.set(n, 0, h + self.pad, w + self.pad, pixel);
                }
            }
        }
    }

    pub fn at(&self, idx: usize) -> Tensor {
        self.slice(idx, 1)
    }

    pub fn slice(&self, idx: usize, num: usize) -> Tensor {
        assert!(idx + num < self.imgs.n, "index out of bounds");
        self.imgs.slice(idx, num)
    }

    pub fn print(&self, idx: usize) {
        let img = self.at(idx);
        for h in self.pad..(28 + self.pad) {
            for w in self.pad..(28 + self.pad) {
                let val = (img.get(0, 0, h, w) * 255.0) as i32;
                if val > 0 {
                    print!("x");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
    }
}

/// MNIST label dataset loader.
///
/// Reads the IDX1 label format (optionally gzip-compressed).
pub struct MNISTLabels {
    labels: Vec<u8>,
}

impl MNISTLabels {
    pub fn new(path: &str) -> Self {
        let file = File::open(path).expect("failed to open MNIST labels file");
        let mut reader: Box<dyn Read> = if path.ends_with(".gz") {
            Box::new(GzDecoder::new(file))
        } else {
            Box::new(file)
        };
        let mut buf4 = [0u8; 4];

        reader.read_exact(&mut buf4).unwrap();
        let magic = u32::from_be_bytes(buf4);
        assert!(magic == 0x00000801, "expected MNIST label file format");

        reader.read_exact(&mut buf4).unwrap();
        let num_labels = u32::from_be_bytes(buf4) as usize;

        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels).unwrap();

        MNISTLabels { labels }
    }

    pub fn at(&self, idx: usize) -> u8 {
        self.labels[idx]
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }
}
