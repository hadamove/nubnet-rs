use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

use pv021_project::{activators::Activator, model::Model};

const NUM_THREADS: usize = 1;

const NUM_CLASSES: usize = 10;

const TEST_VECTORS_PATH: &str = "data/fashion_mnist_test_vectors.csv";
const TEST_LABELS_PATH: &str = "data/fashion_mnist_test_labels.csv";

const TRAIN_VECTORS_PATH: &str = "data/fashion_mnist_train_vectors.csv";
const TRAIN_LABELS_PATH: &str = "data/fashion_mnist_train_labels.csv";

const TRAIN_PREDICTIONS_PATH: &str = "train_predictions.csv";
const TEST_PREDICTIONS_PATH: &str = "test_predictions.csv";

const LEARNING_RATE: f64 = 0.001;
const NUM_EPOCHS: usize = 20;

// Run with `cargo run --release` to get reasonable performance
fn main() -> Result<()> {
    let train_data = load_vectors(TRAIN_VECTORS_PATH)?;
    // TODO: Refactor this mess
    let train_data: Vec<_> = train_data.into_iter().map(Arc::new).collect();

    let train_labels = load_labels(TRAIN_LABELS_PATH)?;
    // TODO: Refactor this mess
    let train_labels: Vec<_> = train_labels.into_iter().map(label_to_one_hot).map(Arc::new).collect();

    let test_data = load_vectors(TEST_VECTORS_PATH)?;
    let test_labels = load_labels(TEST_LABELS_PATH)?;

    let mut model = Model::<NUM_THREADS>::default()
        .with_layer(784, Activator::Identity)
        .with_layer(128, Activator::Tanh)
        .with_layer(64, Activator::Tanh)
        .with_layer(10, Activator::Softmax)
        .with_learning_rate(LEARNING_RATE);

    let t0 = std::time::Instant::now();

    for epoch in 0..NUM_EPOCHS {
        // TODO: Refactor this (use `step_by`)
        for batch in 0..(train_data.len() / NUM_THREADS) {
            let batch_start = batch * NUM_THREADS;
            let batch_end = batch_start + NUM_THREADS;

            let batch_data = &train_data[batch_start..batch_end];
            let batch_labels = &train_labels[batch_start..batch_end];

            model.train_on_batch(batch_data, batch_labels);
        }

        // TODO: Remove this scope before submitting (>Any implementation that uses testing input vectors for anything else than final evaluation will result in failure)
        // TODO: Also remove `humantime` dependency and `calculate_accuracy` function
        {
            let elapsed = humantime::format_duration(std::time::Duration::from_secs(t0.elapsed().as_secs()));
            let acurracy = calculate_accuracy(&mut model, &test_data, &test_labels);

            println!(
                "Epoch: {}  🎯 Accuracy: {:.2}%  ⏳ Time elapsed: {}",
                epoch, acurracy, elapsed
            );
        }
    }

    // TODO: Uncomment and fix parameters
    // export_predictions(&mut model, &train_data, TRAIN_PREDICTIONS_PATH)?;
    // export_predictions(&mut model, &test_data, TEST_PREDICTIONS_PATH)?;

    Ok(())
}

fn load_vectors(filename: &str) -> Result<Vec<Vec<f64>>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let vectors: Result<Vec<Vec<f64>>> = reader
        .lines()
        .map_while(Result::ok)
        .map(|line| {
            line.split(',')
                .map(|value| value.parse::<f64>().map_err(Into::into))
                .map(|value| value.map(|value| value / 255.0))
                .collect()
        })
        .collect();

    vectors
}

fn load_labels(filename: &str) -> Result<Vec<usize>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let labels: Result<Vec<usize>> = reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.parse::<usize>().map_err(Into::into))
        .collect();

    labels
}

fn export_predictions(model: &mut Model<NUM_THREADS>, data: &[Vec<f64>], filename: &str) -> Result<()> {
    let mut file = File::create(filename)?;

    for input in data {
        let prediction = model.predict(input);
        let label = one_hot_to_label(prediction);
        writeln!(file, "{}", label)?;
    }

    Ok(())
}

fn calculate_accuracy(model: &mut Model<NUM_THREADS>, data: &[Vec<f64>], labels: &[usize]) -> f64 {
    let correct = data
        .iter()
        .zip(labels)
        .filter(|(data, label)| {
            let prediction = model.predict(data);
            one_hot_to_label(prediction) == **label
        })
        .count();

    correct as f64 / data.len() as f64 * 100.0
}

fn label_to_one_hot(label: usize) -> Vec<f64> {
    let mut one_hot = vec![0.0; NUM_CLASSES];
    one_hot[label] = 1.0;
    one_hot
}

fn one_hot_to_label(one_hot: &[f64]) -> usize {
    one_hot
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0
}
