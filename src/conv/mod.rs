/// Convolution algorithm implementations.
///
/// Provides multiple algorithms for 2D convolution: naive (reference),
/// Im2col+GEMM, Winograd F(2x2, 3x3), and FFT-based.

mod naive;
mod im2col;
mod winograd;
mod fft;

pub use naive::conv2d_naive;
pub use im2col::conv2d_im2col;
pub use winograd::conv2d_winograd;
pub use fft::conv2d_fft;

/// Selects which convolution algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvAlgorithm {
    /// Naive 6-nested-loop (reference implementation).
    Naive,
    /// Im2col unfolding + tiled GEMM.
    Im2col,
    /// Winograd F(2x2, 3x3). Falls back to Im2col for non-3x3 kernels or stride != 1.
    Winograd,
    /// FFT-based convolution. Slower for small kernels but included for completeness.
    Fft,
}

/// Dispatch convolution to the selected algorithm.
///
/// Input should already be padded. All data is in NCHW layout as flat `f32` slices.
pub fn conv2d(
    algorithm: ConvAlgorithm,
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
    match algorithm {
        ConvAlgorithm::Naive => {
            conv2d_naive(
                input, batch, in_channels, in_h, in_w, weights, bias,
                out_channels, kernel_size, stride, out_h, out_w, output,
            );
        }
        ConvAlgorithm::Im2col => {
            conv2d_im2col(
                input, batch, in_channels, in_h, in_w, weights, bias,
                out_channels, kernel_size, stride, out_h, out_w, output,
            );
        }
        ConvAlgorithm::Winograd => {
            // Winograd only supports 3x3 with stride=1; fall back to Im2col otherwise
            if kernel_size == 3 && stride == 1 {
                conv2d_winograd(
                    input, batch, in_channels, in_h, in_w, weights, bias,
                    out_channels, out_h, out_w, output,
                );
            } else {
                conv2d_im2col(
                    input, batch, in_channels, in_h, in_w, weights, bias,
                    out_channels, kernel_size, stride, out_h, out_w, output,
                );
            }
        }
        ConvAlgorithm::Fft => {
            conv2d_fft(
                input, batch, in_channels, in_h, in_w, weights, bias,
                out_channels, kernel_size, stride, out_h, out_w, output,
            );
        }
    }
}
