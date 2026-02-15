//! Full FP32/INT8/INT4 LeNet-5 pipeline on MNIST.
//!
//! Run with: `cargo run --example lenet_mnist`
//!
//! Requires data files in `data/`:
//! - `lenet.raw` (FP32 weights)
//! - `t10k-images-idx3-ubyte` (MNIST test images)
//! - `t10k-labels-idx1-ubyte.gz` (MNIST test labels)

use microcnn::mnist::{MNIST, MNISTLabels};
use microcnn::lenet::{lenet, lenet_quantized, lenet_quantized_i4};
use microcnn::quantization::Calibrator;
use microcnn::benchmark::{run_benchmark, print_report};

fn main() {
    let weights_path = "data/lenet.raw";
    let test_images_path = "data/t10k-images-idx3-ubyte";
    let test_labels_path = "data/t10k-labels-idx1-ubyte.gz";

    // Load FP32 model
    println!("Loading FP32 LeNet model...");
    let mut fp32_net = lenet(false);
    fp32_net.load(weights_path);

    // Load test data
    println!("Loading MNIST test set...");
    let test_images = MNIST::new(test_images_path);
    let test_labels = MNISTLabels::new(test_labels_path);

    // Calibrate
    let num_calibration = 100;
    println!("Calibrating with {} samples...", num_calibration);
    let mut calibrator = Calibrator::new(fp32_net.num_layers());
    for i in 0..num_calibration {
        let input = test_images.at(i);
        calibrator.observe_input(&input);
        let intermediates = fp32_net.predict_with_intermediates(input);
        for (j, tensor) in intermediates.iter().enumerate() {
            calibrator.observe_layer(j, tensor);
        }
        calibrator.finish_sample();
    }
    println!("Calibration complete ({} samples)", calibrator.num_samples());

    // Build quantized models
    println!("Building INT8 quantized model...");
    let input_params = calibrator.input_params();
    let layer_params = calibrator.layer_params();
    let int8_net = lenet_quantized(&fp32_net, input_params, &layer_params);

    println!("Building INT4 quantized model...");
    let input_params_i4 = calibrator.input_params_i4();
    let layer_params_i4 = calibrator.layer_params_i4();
    let int4_net = lenet_quantized_i4(&fp32_net, input_params_i4, &layer_params_i4);

    // Quick verification
    println!("\n=== Quick Verification (sample 0) ===");
    let input = test_images.at(0);
    test_images.print(0);

    let fp32_pred = fp32_net.predict(input.clone());
    let int8_pred = int8_net.predict(&input);
    let int4_pred = int4_net.predict(&input);

    println!("\nFP32 predictions:");
    for c in 0..fp32_pred.c {
        println!("  {}: {:.4}", c, fp32_pred.get(0, c, 0, 0));
    }

    println!("\nINT8 predictions:");
    for c in 0..int8_pred.c {
        println!("  {}: {:.4}", c, int8_pred.get(0, c, 0, 0));
    }

    println!("\nINT4 predictions:");
    for c in 0..int4_pred.c {
        println!("  {}: {:.4}", c, int4_pred.get(0, c, 0, 0));
    }

    // Benchmark
    let num_benchmark = 1000;
    println!("\nRunning benchmark ({} samples)...", num_benchmark);
    let result = run_benchmark(&mut fp32_net, &int8_net, &int4_net, &test_images, &test_labels, num_benchmark);
    print_report(&result);
}
