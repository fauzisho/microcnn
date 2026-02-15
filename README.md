# microcnn

A minimal CNN framework in Rust with INT8 and INT4 quantization.

## Features

- FP32, INT8, and INT4 inference
- Post-training quantization with calibration
- Reference LeNet-5 implementation for MNIST

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

Requires MNIST data files in `data/`.

## License

MIT
