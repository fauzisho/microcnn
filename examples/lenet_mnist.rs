//! Full FP32/INT8/INT4 LeNet-5 pipeline on MNIST.
//!
//! Run with: `cargo run --example lenet_mnist`
//!
//! Requires data files in `data/`:
//! - `lenet.raw` (FP32 weights)
//! - `t10k-images-idx3-ubyte` (MNIST test images)
//! - `t10k-labels-idx1-ubyte.gz` (MNIST test labels)

use std::time::{Duration, Instant};

use microcnn::conv::ConvAlgorithm;
use microcnn::mnist::{MNIST, MNISTLabels};
use microcnn::lenet::{lenet, lenet_with_algorithm, lenet_quantized, lenet_quantized_i4};
use microcnn::quantization::Calibrator;
use microcnn::benchmark::{run_benchmark, print_report};
use microcnn::tensor::Tensor;

fn argmax(tensor: &Tensor) -> usize {
    let mut max_val = f32::MIN;
    let mut max_idx = 0;
    for c in 0..tensor.c {
        let val = tensor.get(0, c, 0, 0);
        if val > max_val {
            max_val = val;
            max_idx = c;
        }
    }
    max_idx
}

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

    // Benchmark (FP32 vs INT8 vs INT4)
    let num_benchmark = 1000;
    println!("\nRunning benchmark ({} samples)...", num_benchmark);
    let result = run_benchmark(&mut fp32_net, &int8_net, &int4_net, &test_images, &test_labels, num_benchmark);
    print_report(&result);

    // ===== Convolution Algorithm Comparison =====
    println!("\n{}", "=".repeat(60));
    println!("=== Convolution Algorithm Comparison (FP32) ===");
    println!("{}", "=".repeat(60));

    let algorithms = [
        (ConvAlgorithm::Naive, "Naive"),
        (ConvAlgorithm::Im2col, "Im2col"),
        (ConvAlgorithm::Winograd, "Winograd"),
        (ConvAlgorithm::Fft, "FFT"),
    ];

    let num_conv_benchmark = 1000;

    // Build and load a network for each algorithm
    let mut algo_nets: Vec<_> = algorithms
        .iter()
        .map(|(algo, name)| {
            let mut net = lenet_with_algorithm(false, *algo);
            net.load(weights_path);
            (*algo, *name, net)
        })
        .collect();

    // Verify correctness: all algorithms should produce the same predictions
    println!("\n--- Correctness Check (first 100 samples) ---");
    let num_check = 100;
    let mut mismatches = vec![0usize; algorithms.len()];
    let mut max_diffs = vec![0.0f32; algorithms.len()];

    // Use Naive as the reference
    let (ref_nets, other_nets) = algo_nets.split_at_mut(1);
    let ref_net = &mut ref_nets[0].2;

    for i in 0..num_check {
        let input = test_images.at(i);
        let ref_pred = ref_net.predict(input.clone());
        let ref_class = argmax(&ref_pred);

        for (j, (_, _, net)) in other_nets.iter_mut().enumerate() {
            let pred = net.predict(input.clone());
            let pred_class = argmax(&pred);
            if pred_class != ref_class {
                mismatches[j + 1] += 1;
            }
            // Track max absolute difference
            for c in 0..ref_pred.c {
                let diff = (ref_pred.get(0, c, 0, 0) - pred.get(0, c, 0, 0)).abs();
                if diff > max_diffs[j + 1] {
                    max_diffs[j + 1] = diff;
                }
            }
        }
    }

    println!("{:<12} {:>12} {:>16}", "Algorithm", "Mismatches", "Max Abs Diff");
    for (i, (_, name, _)) in algo_nets.iter().enumerate() {
        if i == 0 {
            println!("{:<12} {:>12} {:>16}", name, "(reference)", "-");
        } else {
            println!(
                "{:<12} {:>12} {:>16.6e}",
                name, mismatches[i], max_diffs[i]
            );
        }
    }

    // Timing benchmark
    println!("\n--- Timing Benchmark ({} samples) ---", num_conv_benchmark);

    let mut algo_results: Vec<(&str, Duration, usize, f32)> = Vec::new();

    for (_, name, net) in algo_nets.iter_mut() {
        let mut total_time = Duration::ZERO;
        let mut correct = 0usize;
        let actual = num_conv_benchmark.min(test_labels.len());

        for i in 0..actual {
            let input = test_images.at(i);
            let label = test_labels.at(i) as usize;

            let start = Instant::now();
            let pred = net.predict(input);
            total_time += start.elapsed();

            if argmax(&pred) == label {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / actual as f32 * 100.0;
        algo_results.push((name, total_time, actual, accuracy));
    }

    // Print results
    let baseline_ms = algo_results[0].1.as_micros() as f64 / 1000.0;

    println!(
        "\n{:<12} {:>12} {:>10} {:>12} {:>10}",
        "Algorithm", "Total Time", "Per Image", "Speedup", "Accuracy"
    );
    println!("{}", "-".repeat(60));

    for (name, total_time, _, accuracy) in &algo_results {
        let total_ms = total_time.as_micros() as f64 / 1000.0;
        let per_image_us = total_time.as_micros() as f64 / num_conv_benchmark as f64;
        let speedup = if total_ms > 0.001 {
            baseline_ms / total_ms
        } else {
            0.0
        };

        println!(
            "{:<12} {:>9.1}ms {:>8.1}us {:>11.2}x {:>9.1}%",
            name, total_ms, per_image_us, speedup, accuracy
        );
    }

    // Per-layer timing breakdown for each algorithm
    println!("\n--- Per-Layer Conv2d Timing (single inference) ---");
    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Algorithm", "Conv2d #0", "Conv2d #1", "Conv2d #2"
    );
    println!("{}", "-".repeat(52));

    for (_, name, net) in algo_nets.iter_mut() {
        let input = test_images.at(0);
        let (_, timings) = net.predict_timed(input);

        // Conv2d layers are at indices 0, 3, 6
        let conv_indices = [0, 3, 6];
        let conv_times: Vec<String> = conv_indices
            .iter()
            .map(|&idx| {
                if idx < timings.len() {
                    format!("{:.1}us", timings[idx].1.as_micros())
                } else {
                    "-".to_string()
                }
            })
            .collect();

        println!(
            "{:<12} {:>12} {:>12} {:>12}",
            name, conv_times[0], conv_times[1], conv_times[2]
        );
    }
}
