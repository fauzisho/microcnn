/// FFT-based convolution.
///
/// Uses radix-2 Cooley-Tukey FFT. Zero-pads input and kernel to next power of 2,
/// multiplies in frequency domain, then inverse FFTs back.

use std::f32::consts::PI;

#[derive(Clone, Copy)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }

    fn mul(self, other: Self) -> Self {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn conj(self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    fn add(self, other: Self) -> Self {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

fn next_power_of_2(n: usize) -> usize {
    let mut v = 1;
    while v < n {
        v <<= 1;
    }
    v
}

/// Bit-reversal permutation for in-place FFT.
fn bit_reverse(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// In-place radix-2 Cooley-Tukey FFT.
/// If `inverse` is true, performs the inverse FFT (with 1/N scaling).
fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two());

    bit_reverse(data);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / len as f32
        } else {
            -2.0 * PI / len as f32
        };
        let wn = Complex::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = data[i + j];
                let v = w.mul(data[i + j + half]);
                data[i + j] = u.add(v);
                data[i + j + half] = u.sub(v);
                w = w.mul(wn);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for d in data.iter_mut() {
            d.re *= scale;
            d.im *= scale;
        }
    }
}

/// 2D FFT: row-then-column 1D FFTs.
fn fft_2d(data: &mut [Complex], rows: usize, cols: usize, inverse: bool) {
    let mut row_buf = vec![Complex::zero(); cols];
    for r in 0..rows {
        let off = r * cols;
        row_buf.copy_from_slice(&data[off..off + cols]);
        fft_1d(&mut row_buf, inverse);
        data[off..off + cols].copy_from_slice(&row_buf);
    }

    let mut col_buf = vec![Complex::zero(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = data[r * cols + c];
        }
        fft_1d(&mut col_buf, inverse);
        for r in 0..rows {
            data[r * cols + c] = col_buf[r];
        }
    }
}

/// FFT-based 2D cross-correlation for a single (input_channel, output_channel) pair.
///
/// Computes: output[r][c] += sum_kh,kw kernel[kh][kw] * input[r+kh][c+kw]
/// using: IFFT(conj(FFT(kernel)) * FFT(input))
fn fft_conv2d_single(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    kernel: &[f32],
    k_h: usize,
    k_w: usize,
    out_h: usize,
    out_w: usize,
    fft_h: usize,
    fft_w: usize,
    input_fft: &mut [Complex],
    kernel_fft: &mut [Complex],
    output_acc: &mut [f32],
) {
    let fft_size = fft_h * fft_w;

    // Zero-pad input into FFT buffer
    for v in input_fft.iter_mut() {
        *v = Complex::zero();
    }
    for r in 0..in_h {
        for c in 0..in_w {
            input_fft[r * fft_w + c] = Complex::new(input[r * in_w + c], 0.0);
        }
    }

    // Zero-pad kernel into FFT buffer (no flip needed, we use conjugate instead)
    for v in kernel_fft.iter_mut() {
        *v = Complex::zero();
    }
    for r in 0..k_h {
        for c in 0..k_w {
            kernel_fft[r * fft_w + c] = Complex::new(kernel[r * k_w + c], 0.0);
        }
    }

    // Forward FFT both
    fft_2d(input_fft, fft_h, fft_w, false);
    fft_2d(kernel_fft, fft_h, fft_w, false);

    // Pointwise multiply: conj(kernel_fft) * input_fft for cross-correlation
    for i in 0..fft_size {
        input_fft[i] = kernel_fft[i].conj().mul(input_fft[i]);
    }

    // Inverse FFT
    fft_2d(input_fft, fft_h, fft_w, true);

    // Accumulate valid region into output (starts at index 0 for cross-correlation)
    for r in 0..out_h {
        for c in 0..out_w {
            output_acc[r * out_w + c] += input_fft[r * fft_w + c].re;
        }
    }
}

/// FFT-based convolution.
///
/// Zero-pads input and kernel to next power of 2, multiplies in frequency domain.
/// Note: slower than naive for small kernels, included for completeness.
pub fn conv2d_fft(
    input: &[f32],
    batch: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    weights: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    out_h: usize,
    out_w: usize,
    output: &mut [f32],
) {
    let full_out_h = in_h - kernel_size + 1;
    let full_out_w = in_w - kernel_size + 1;

    let fft_h = next_power_of_2(in_h);
    let fft_w = next_power_of_2(in_w);
    let fft_size = fft_h * fft_w;

    let mut input_fft = vec![Complex::zero(); fft_size];
    let mut kernel_fft = vec![Complex::zero(); fft_size];

    let in_spatial = in_channels * in_h * in_w;
    let out_spatial = out_channels * out_h * out_w;

    let mut full_output = if stride > 1 {
        vec![0.0f32; full_out_h * full_out_w]
    } else {
        Vec::new()
    };

    for n in 0..batch {
        let in_off = n * in_spatial;
        let out_off = n * out_spatial;

        for oc in 0..out_channels {
            if stride > 1 {
                full_output.iter_mut().for_each(|v| *v = 0.0);
            } else {
                let ch_off = out_off + oc * out_h * out_w;
                for v in output[ch_off..ch_off + out_h * out_w].iter_mut() {
                    *v = 0.0;
                }
            }

            for ic in 0..in_channels {
                let input_plane = &input[in_off + ic * in_h * in_w..];
                let kernel_off = oc * in_channels * kernel_size * kernel_size
                    + ic * kernel_size * kernel_size;
                let kernel_plane = &weights[kernel_off..kernel_off + kernel_size * kernel_size];

                if stride > 1 {
                    fft_conv2d_single(
                        input_plane, in_h, in_w,
                        kernel_plane, kernel_size, kernel_size,
                        full_out_h, full_out_w,
                        fft_h, fft_w,
                        &mut input_fft, &mut kernel_fft,
                        &mut full_output,
                    );
                } else {
                    let ch_off = out_off + oc * out_h * out_w;
                    fft_conv2d_single(
                        input_plane, in_h, in_w,
                        kernel_plane, kernel_size, kernel_size,
                        out_h, out_w,
                        fft_h, fft_w,
                        &mut input_fft, &mut kernel_fft,
                        &mut output[ch_off..],
                    );
                }
            }

            // Add bias
            if stride > 1 {
                let ch_off = out_off + oc * out_h * out_w;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        output[ch_off + oh * out_w + ow] =
                            full_output[oh * stride * full_out_w + ow * stride] + bias[oc];
                    }
                }
            } else {
                let ch_off = out_off + oc * out_h * out_w;
                for v in output[ch_off..ch_off + out_h * out_w].iter_mut() {
                    *v += bias[oc];
                }
            }
        }
    }
}
