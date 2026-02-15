/// Naive 6-nested-loop convolution (reference implementation).
///
/// Operates on plain `Vec<f32>` slices in NCHW layout to avoid `RefCell` borrow overhead.
pub fn conv2d_naive(
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
    for n in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ic in 0..in_channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let in_idx = n * in_channels * in_h * in_w
                                    + ic * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                let w_idx = oc * in_channels * kernel_size * kernel_size
                                    + ic * kernel_size * kernel_size
                                    + kh * kernel_size
                                    + kw;
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                    let out_idx = n * out_channels * out_h * out_w
                        + oc * out_h * out_w
                        + oh * out_w
                        + ow;
                    output[out_idx] = sum + bias[oc];
                }
            }
        }
    }
}
