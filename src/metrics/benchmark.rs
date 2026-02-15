use std::time::{Duration, Instant};

use crate::tensor::Tensor;
use crate::network::{NeuralNetwork, QuantizedNeuralNetwork, QuantizedNeuralNetworkI4, LayerType};
use crate::mnist::{MNIST, MNISTLabels};

/// Per-layer performance and accuracy metrics.
pub struct LayerMetrics {
    pub layer_idx: usize,
    pub layer_type: LayerType,
    pub fp32_time: Duration,
    pub int8_time: Duration,
    pub int4_time: Duration,
    pub fp32_mem: usize,
    pub int8_mem: usize,
    pub int4_mem: usize,
    pub int8_mse: f32,
    pub int4_mse: f32,
}

/// Aggregate benchmark results comparing FP32, INT8, and INT4 inference.
pub struct BenchmarkResult {
    pub layer_metrics: Vec<LayerMetrics>,
    pub fp32_total_time: Duration,
    pub int8_total_time: Duration,
    pub int4_total_time: Duration,
    pub fp32_total_mem: usize,
    pub int8_total_mem: usize,
    pub int4_total_mem: usize,
    pub fp32_accuracy: f32,
    pub int8_accuracy: f32,
    pub int4_accuracy: f32,
    pub int8_agreement: f32,
    pub int4_agreement: f32,
    pub num_samples: usize,
}

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

fn tensor_mse(a: &Tensor, b: &Tensor) -> f32 {
    let total = a.n * a.c * a.h * a.w;
    if total == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for n in 0..a.n {
        for c in 0..a.c {
            for h in 0..a.h {
                for w in 0..a.w {
                    let diff = (a.get(n, c, h, w) - b.get(n, c, h, w)) as f64;
                    sum += diff * diff;
                }
            }
        }
    }
    (sum / total as f64) as f32
}

/// Run a benchmark comparing FP32, INT8, and INT4 networks on MNIST test data.
pub fn run_benchmark(
    fp32_net: &mut NeuralNetwork,
    int8_net: &QuantizedNeuralNetwork,
    int4_net: &QuantizedNeuralNetworkI4,
    test_images: &MNIST,
    test_labels: &MNISTLabels,
    num_samples: usize,
) -> BenchmarkResult {
    let first_input = test_images.at(0);

    let (fp32_intermediates, fp32_layer_times) = fp32_net.predict_timed(first_input.clone());
    let (int8_intermediates, int8_layer_times) = int8_net.predict_timed(&first_input);
    let (int4_intermediates, int4_layer_times) = int4_net.predict_timed(&first_input);

    let fp32_layer_mem = fp32_net.layer_weight_memory();
    let int8_layer_mem = int8_net.layer_weight_memory();
    let int4_layer_mem = int4_net.layer_weight_memory();

    let num_layers = fp32_layer_times.len().min(int8_layer_times.len()).min(int4_layer_times.len());
    let mut layer_metrics = Vec::new();

    for i in 0..num_layers {
        let int8_mse = if i < fp32_intermediates.len() && i < int8_intermediates.len() {
            tensor_mse(&fp32_intermediates[i], &int8_intermediates[i])
        } else {
            0.0
        };
        let int4_mse = if i < fp32_intermediates.len() && i < int4_intermediates.len() {
            tensor_mse(&fp32_intermediates[i], &int4_intermediates[i])
        } else {
            0.0
        };

        layer_metrics.push(LayerMetrics {
            layer_idx: i,
            layer_type: fp32_layer_times[i].0,
            fp32_time: fp32_layer_times[i].1,
            int8_time: int8_layer_times[i].1,
            int4_time: int4_layer_times[i].1,
            fp32_mem: if i < fp32_layer_mem.len() { fp32_layer_mem[i] } else { 0 },
            int8_mem: if i < int8_layer_mem.len() { int8_layer_mem[i] } else { 0 },
            int4_mem: if i < int4_layer_mem.len() { int4_layer_mem[i] } else { 0 },
            int8_mse,
            int4_mse,
        });
    }

    let mut fp32_correct = 0usize;
    let mut int8_correct = 0usize;
    let mut int4_correct = 0usize;
    let mut int8_agree = 0usize;
    let mut int4_agree = 0usize;
    let mut fp32_total_time = Duration::ZERO;
    let mut int8_total_time = Duration::ZERO;
    let mut int4_total_time = Duration::ZERO;

    let actual_samples = num_samples.min(test_labels.len());

    for i in 0..actual_samples {
        let input = test_images.at(i);
        let label = test_labels.at(i) as usize;

        let start = Instant::now();
        let fp32_pred = fp32_net.predict(input.clone());
        fp32_total_time += start.elapsed();

        let start = Instant::now();
        let int8_pred = int8_net.predict(&input);
        int8_total_time += start.elapsed();

        let start = Instant::now();
        let int4_pred = int4_net.predict(&input);
        int4_total_time += start.elapsed();

        let fp32_class = argmax(&fp32_pred);
        let int8_class = argmax(&int8_pred);
        let int4_class = argmax(&int4_pred);

        if fp32_class == label { fp32_correct += 1; }
        if int8_class == label { int8_correct += 1; }
        if int4_class == label { int4_correct += 1; }
        if fp32_class == int8_class { int8_agree += 1; }
        if fp32_class == int4_class { int4_agree += 1; }
    }

    let fp32_total_mem: usize = fp32_layer_mem.iter().sum();
    let int8_total_mem: usize = int8_layer_mem.iter().sum();
    let int4_total_mem: usize = int4_layer_mem.iter().sum();

    BenchmarkResult {
        layer_metrics,
        fp32_total_time,
        int8_total_time,
        int4_total_time,
        fp32_total_mem,
        int8_total_mem,
        int4_total_mem,
        fp32_accuracy: fp32_correct as f32 / actual_samples as f32 * 100.0,
        int8_accuracy: int8_correct as f32 / actual_samples as f32 * 100.0,
        int4_accuracy: int4_correct as f32 / actual_samples as f32 * 100.0,
        int8_agreement: int8_agree as f32 / actual_samples as f32 * 100.0,
        int4_agreement: int4_agree as f32 / actual_samples as f32 * 100.0,
        num_samples: actual_samples,
    }
}

/// Print a formatted benchmark report to stdout.
pub fn print_report(result: &BenchmarkResult) {
    println!("\n=== Per-Layer Metrics ===");
    println!(
        "{:<6} {:<10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "Layer", "Type", "FP32 Time", "INT8 Time", "INT4 Time", "FP32 Mem", "INT8 Mem", "INT4 Mem", "INT8 MSE", "INT4 MSE"
    );

    for m in &result.layer_metrics {
        let fp32_ms = m.fp32_time.as_micros() as f64 / 1000.0;
        let int8_ms = m.int8_time.as_micros() as f64 / 1000.0;
        let int4_ms = m.int4_time.as_micros() as f64 / 1000.0;

        println!(
            "{:<6} {:<10} {:>8.2}ms {:>8.2}ms {:>8.2}ms {:>8}B {:>8}B {:>8}B {:>10.6} {:>10.6}",
            m.layer_idx, m.layer_type, fp32_ms, int8_ms, int4_ms,
            m.fp32_mem, m.int8_mem, m.int4_mem, m.int8_mse, m.int4_mse
        );
    }

    let fp32_total_ms = result.fp32_total_time.as_micros() as f64 / 1000.0;
    let int8_total_ms = result.int8_total_time.as_micros() as f64 / 1000.0;
    let int4_total_ms = result.int4_total_time.as_micros() as f64 / 1000.0;
    let int8_speedup = if int8_total_ms > 0.001 { fp32_total_ms / int8_total_ms } else { 0.0 };
    let int4_speedup = if int4_total_ms > 0.001 { fp32_total_ms / int4_total_ms } else { 0.0 };
    let int8_savings = if result.fp32_total_mem > 0 {
        (1.0 - result.int8_total_mem as f64 / result.fp32_total_mem as f64) * 100.0
    } else {
        0.0
    };
    let int4_savings = if result.fp32_total_mem > 0 {
        (1.0 - result.int4_total_mem as f64 / result.fp32_total_mem as f64) * 100.0
    } else {
        0.0
    };

    println!("\n=== Totals ===");
    println!(
        "FP32: {:.1}ms / {}B",
        fp32_total_ms, result.fp32_total_mem,
    );
    println!(
        "INT8: {:.1}ms / {}B    Speedup: {:.2}x  Savings: {:.0}%",
        int8_total_ms, result.int8_total_mem,
        int8_speedup, int8_savings
    );
    println!(
        "INT4: {:.1}ms / {}B    Speedup: {:.2}x  Savings: {:.0}%",
        int4_total_ms, result.int4_total_mem,
        int4_speedup, int4_savings
    );

    println!("\n=== Accuracy ({} samples) ===", result.num_samples);
    println!(
        "FP32: {:.1}%    INT8: {:.1}% (drop {:.1}%, agree {:.1}%)    INT4: {:.1}% (drop {:.1}%, agree {:.1}%)",
        result.fp32_accuracy,
        result.int8_accuracy,
        result.fp32_accuracy - result.int8_accuracy,
        result.int8_agreement,
        result.int4_accuracy,
        result.fp32_accuracy - result.int4_accuracy,
        result.int4_agreement,
    );
}
