# microcnn

A minimal CNN framework in Rust with INT8 and INT4 quantization.

## Features

- FP32, INT8, and INT4 inference
- Post-training quantization with calibration
- Reference LeNet-5 implementation for MNIST

## Benchmarks

Tested on LeNet-5 with 1000 MNIST samples:

| Precision | Inference Time | Memory | Speedup | Savings | Accuracy |
|-----------|----------------|--------|---------|---------|----------|
| FP32      | 43.37s         | 241 KB | 1.00x   | —       | 98.7%    |
| INT8      | 25.36s         | 61 KB  | 1.71x   | 75%     | 98.7%    |
| INT4      | 35.04s         | 31 KB  | 1.24x   | 87%     | 95.3%    |

### Per-Layer Performance

| Layer | Type      | FP32 Time | INT8 Time | INT8 MSE | INT4 MSE |
|-------|-----------|-----------|-----------|----------|----------|
| 0     | Conv2d    | 12.48ms   | 7.47ms    | 0.000115 | 0.022301 |
| 1     | ReLU      | 0.47ms    | 0.30ms    | 0.000080 | 0.017441 |
| 2     | MaxPool2d | 0.42ms    | 0.28ms    | 0.000087 | 0.019562 |
| 3     | Conv2d    | 24.16ms   | 13.89ms   | 0.000822 | 0.213480 |
| 4     | ReLU      | 0.17ms    | 0.10ms    | 0.000188 | 0.059043 |
| 5     | MaxPool2d | 0.14ms    | 0.10ms    | 0.000370 | 0.116998 |
| 6     | Conv2d    | 4.87ms    | 2.71ms    | 0.000895 | 0.331971 |
| 7     | ReLU      | 0.01ms    | 0.01ms    | 0.000383 | 0.124720 |
| 8     | Linear    | 1.01ms    | 0.51ms    | 0.000362 | 0.202737 |
| 9     | ReLU      | 0.01ms    | 0.01ms    | 0.000174 | 0.096129 |
| 10    | Linear    | 0.07ms    | 0.04ms    | 0.001202 | 1.060178 |
| 11    | Softmax   | 0.00ms    | 0.00ms    | 0.000000 | 0.000236 |

**Key findings:**
- INT8 achieves **1.71x speedup** with **zero accuracy loss**
- INT4 reduces memory by **87%** with only 3.4% accuracy drop
- Conv2d layers benefit most from quantization

## Quick Start

```toml
[dependencies]
microcnn = "0.1"
```

```rust
use microcnn::lenet::lenet;

let mut net = lenet(false);
net.load("data/lenet.raw");
```

## Running the Example

```bash
cargo run --example lenet_mnist
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
