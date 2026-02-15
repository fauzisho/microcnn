# microcnn

A minimal CNN framework in Rust with INT8 and INT4 quantization.

## Features

- FP32, INT8, and INT4 inference
- Post-training quantization with calibration
- NEON SIMD acceleration (aarch64)
- Multiple convolution algorithms (Naive, Im2col, Winograd, FFT)
- Reference LeNet-5 implementation for MNIST

## Benchmarks

Tested on LeNet-5 with 1000 MNIST samples (Apple Silicon, NEON SIMD enabled):

| Precision | Inference Time | Memory | Speedup | Savings | Accuracy |
|-----------|----------------|--------|---------|---------|----------|
| FP32      | 688.4ms        | 241 KB | 1.00x   | —       | 98.7%    |
| INT8      | 120.8ms        | 61 KB  | 5.70x   | 75%     | 98.7%    |
| INT4      | 845.3ms        | 31 KB  | 0.81x   | 87%     | 95.3%    |

### Per-Layer Performance

| Layer | Type      | FP32 Time | INT8 Time | INT4 Time | INT8 MSE | INT4 MSE |
|-------|-----------|-----------|-----------|-----------|----------|----------|
| 0     | Conv2d    | 0.13ms    | 0.05ms    | 0.23ms    | 0.000115 | 0.022301 |
| 1     | ReLU      | 0.01ms    | 0.00ms    | 0.01ms    | 0.000080 | 0.017441 |
| 2     | MaxPool2d | 0.01ms    | 0.01ms    | 0.01ms    | 0.000087 | 0.019562 |
| 3     | Conv2d    | 0.31ms    | 0.04ms    | 0.47ms    | 0.000822 | 0.213480 |
| 4     | ReLU      | 0.00ms    | 0.00ms    | 0.00ms    | 0.000188 | 0.059043 |
| 5     | MaxPool2d | 0.00ms    | 0.00ms    | 0.00ms    | 0.000370 | 0.116998 |
| 6     | Conv2d    | 0.19ms    | 0.01ms    | 0.09ms    | 0.000895 | 0.331971 |
| 7     | ReLU      | 0.00ms    | 0.00ms    | 0.00ms    | 0.000383 | 0.124720 |
| 8     | Linear    | 0.02ms    | 0.01ms    | 0.02ms    | 0.000362 | 0.202737 |
| 9     | ReLU      | 0.00ms    | 0.00ms    | 0.00ms    | 0.000174 | 0.096129 |
| 10    | Linear    | 0.00ms    | 0.00ms    | 0.00ms    | 0.001202 | 1.060178 |
| 11    | Softmax   | 0.00ms    | 0.00ms    | 0.00ms    | 0.000000 | 0.000236 |

### Convolution Algorithm Comparison (FP32)

| Algorithm | Total Time | Per Image | Speedup | Max Error vs Naive |
|-----------|------------|-----------|---------|-------------------|
| Naive     | 685.5ms    | 685.5µs   | 1.00x   | —                 |
| Im2col    | 553.3ms    | 553.3µs   | 1.24x   | 1.86e-7           |
| Winograd  | 552.7ms    | 552.7µs   | 1.24x   | 1.86e-7           |
| FFT       | 7996.8ms   | 7996.8µs  | 0.09x   | 9.54e-7           |

### SIMD Im2col Performance

| Layer     | FP32 Im2col | INT8 Im2col | INT8 Speedup |
|-----------|-------------|-------------|--------------|
| Conv2d #0 | 68.0µs      | 48.4µs      | 1.40x        |
| Conv2d #1 | 105.3µs     | 39.4µs      | 2.67x        |
| Conv2d #2 | 320.5µs     | 5.4µs       | 59.01x       |
| **Total** | 493.8µs     | 93.3µs      | **5.29x**    |

**Key findings:**

- INT8 achieves **5.70x speedup** with **zero accuracy loss**
- INT4 reduces memory by **87%** with only 3.4% accuracy drop
- Conv2d layers benefit most from quantization (up to 59x speedup on layer 2)
- Im2col and Winograd provide 1.24x speedup over naive convolution
- NEON SIMD delivers massive gains for INT8 convolutions

📊 [View detailed benchmark results](https://github.com/fauzisho/microcnn/blob/main/result/lenet_mnist)

## Quick Start

### Install

```bash
cargo install microcnn
```

Running the above command will globally install the microcnn binary.

### Install as library

Run the following Cargo command in your project directory:

```bash
cargo add microcnn
```

Or add to your `Cargo.toml`:

```toml
[dependencies]
microcnn = "0.1"
```

### Usage

```rust
use microcnn::lenet::lenet;

let mut net = lenet(false);
net.load("data/lenet.raw");
```

## Running the Example

```bash
cargo run --release --example lenet_mnist
```

Or copy the example code directly into your `main.rs`:

```rust
use microcnn::lenet::lenet;
use microcnn::mnist::MNIST;

fn main() {
    // Load model
    let mut net = lenet(false);
    net.load("data/lenet.raw");

    // Load MNIST test images and pick a random one
    let test_images = MNIST::new("data/t10k-images-idx3-ubyte");
    let idx = 0; // change sample index here
    let input = test_images.at(idx);

    // Print the image to terminal
    test_images.print(idx);

    // Run inference
    let output = net.predict(input);
    let prediction = (0..output.c)
        .max_by(|&a, &b| {
            output.get(0, a, 0, 0)
                .partial_cmp(&output.get(0, b, 0, 0))
                .unwrap()
        })
        .unwrap();

    println!("Predicted digit: {}", prediction);
}
```

Requires MNIST data files in `data/`.

## License

MIT
