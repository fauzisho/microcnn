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

    // ===== SIMD Platform Detection =====
    println!("\n{}", "=".repeat(60));
    println!("=== SIMD Platform Info ===");
    if cfg!(all(target_arch = "aarch64", feature = "simd")) {
        println!("  NEON SIMD:  ENABLED (aarch64 + simd feature)");
    } else if cfg!(target_arch = "aarch64") {
        println!("  NEON SIMD:  DISABLED (simd feature not enabled)");
    } else {
        println!("  NEON SIMD:  NOT AVAILABLE (not aarch64)");
    }

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

    // ===== SIMD Im2col Benchmark: FP32 vs INT8 vs INT4 =====
    println!("\n{}", "=".repeat(60));
    println!("=== SIMD Im2col Benchmark (FP32 vs INT8 vs INT4) ===");
    println!("{}", "=".repeat(60));

    if cfg!(all(target_arch = "aarch64", feature = "simd")) {
        println!("  Backend: NEON SIMD (aarch64)");
    } else {
        println!("  Backend: Scalar fallback");
    }

    // Per-layer Conv2d timing comparison (single inference, averaged over warmup)
    let warmup = 10;
    let timing_iters = 100;

    // Warm up
    for _ in 0..warmup {
        let input = test_images.at(0);
        let _ = algo_nets.iter_mut()
            .find(|(a, _, _)| *a == ConvAlgorithm::Im2col)
            .map(|(_, _, net)| net.predict(input.clone()));
        let _ = int8_net.predict(&input);
        let _ = int4_net.predict(&input);
    }

    // Collect per-layer timings (averaged)
    let mut fp32_conv_total = [Duration::ZERO; 3];
    let mut int8_conv_total = [Duration::ZERO; 3];
    let mut int4_conv_total = [Duration::ZERO; 3];
    let mut fp32_full_total = Duration::ZERO;
    let mut int8_full_total = Duration::ZERO;
    let mut int4_full_total = Duration::ZERO;

    let conv_layer_indices = [0, 3, 6]; // Conv2d layer positions in LeNet

    let im2col_net = algo_nets.iter_mut()
        .find(|(a, _, _)| *a == ConvAlgorithm::Im2col)
        .map(|(_, _, net)| net)
        .unwrap();

    for i in 0..timing_iters {
        let input = test_images.at(i % 100);

        let (_, fp32_timings) = im2col_net.predict_timed(input.clone());
        let (_, int8_timings) = int8_net.predict_timed(&input);
        let (_, int4_timings) = int4_net.predict_timed(&input);

        for (ci, &layer_idx) in conv_layer_indices.iter().enumerate() {
            if layer_idx < fp32_timings.len() {
                fp32_conv_total[ci] += fp32_timings[layer_idx].1;
            }
            if layer_idx < int8_timings.len() {
                int8_conv_total[ci] += int8_timings[layer_idx].1;
            }
            if layer_idx < int4_timings.len() {
                int4_conv_total[ci] += int4_timings[layer_idx].1;
            }
        }

        fp32_full_total += fp32_timings.iter().map(|(_, d)| *d).sum::<Duration>();
        int8_full_total += int8_timings.iter().map(|(_, d)| *d).sum::<Duration>();
        int4_full_total += int4_timings.iter().map(|(_, d)| *d).sum::<Duration>();
    }

    println!("\n--- Per-Layer Conv2d Timing (avg over {} inferences) ---", timing_iters);
    println!(
        "{:<10} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "Layer", "FP32 Im2col", "INT8 Im2col", "INT4", "INT8 Speedup", "INT4 Speedup"
    );
    println!("{}", "-".repeat(84));

    let layer_names = ["Conv2d #0", "Conv2d #1", "Conv2d #2"];
    for ci in 0..3 {
        let fp32_us = fp32_conv_total[ci].as_nanos() as f64 / timing_iters as f64 / 1000.0;
        let int8_us = int8_conv_total[ci].as_nanos() as f64 / timing_iters as f64 / 1000.0;
        let int4_us = int4_conv_total[ci].as_nanos() as f64 / timing_iters as f64 / 1000.0;
        let int8_speedup = if int8_us > 0.001 { fp32_us / int8_us } else { 0.0 };
        let int4_speedup = if int4_us > 0.001 { fp32_us / int4_us } else { 0.0 };

        println!(
            "{:<10} {:>12.1}us {:>12.1}us {:>12.1}us {:>13.2}x {:>13.2}x",
            layer_names[ci], fp32_us, int8_us, int4_us, int8_speedup, int4_speedup
        );
    }

    // Total conv time
    let fp32_conv_us: f64 = fp32_conv_total.iter()
        .map(|d| d.as_nanos() as f64 / timing_iters as f64 / 1000.0)
        .sum();
    let int8_conv_us: f64 = int8_conv_total.iter()
        .map(|d| d.as_nanos() as f64 / timing_iters as f64 / 1000.0)
        .sum();
    let int4_conv_us: f64 = int4_conv_total.iter()
        .map(|d| d.as_nanos() as f64 / timing_iters as f64 / 1000.0)
        .sum();
    let int8_conv_speedup = if int8_conv_us > 0.001 { fp32_conv_us / int8_conv_us } else { 0.0 };
    let int4_conv_speedup = if int4_conv_us > 0.001 { fp32_conv_us / int4_conv_us } else { 0.0 };

    println!("{}", "-".repeat(84));
    println!(
        "{:<10} {:>12.1}us {:>12.1}us {:>12.1}us {:>13.2}x {:>13.2}x",
        "Conv Total", fp32_conv_us, int8_conv_us, int4_conv_us, int8_conv_speedup, int4_conv_speedup
    );

    // Full network timing
    let fp32_full_us = fp32_full_total.as_nanos() as f64 / timing_iters as f64 / 1000.0;
    let int8_full_us = int8_full_total.as_nanos() as f64 / timing_iters as f64 / 1000.0;
    let int4_full_us = int4_full_total.as_nanos() as f64 / timing_iters as f64 / 1000.0;
    let int8_full_speedup = if int8_full_us > 0.001 { fp32_full_us / int8_full_us } else { 0.0 };
    let int4_full_speedup = if int4_full_us > 0.001 { fp32_full_us / int4_full_us } else { 0.0 };

    println!(
        "{:<10} {:>12.1}us {:>12.1}us {:>12.1}us {:>13.2}x {:>13.2}x",
        "Full Net", fp32_full_us, int8_full_us, int4_full_us, int8_full_speedup, int4_full_speedup
    );
}
