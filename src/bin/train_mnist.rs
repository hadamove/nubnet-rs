use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::Result;
use pv021_project::{activators::Activator, model::Model};

const NUM_CLASSES: usize = 10;

const TEST_VECTORS_PATH: &str = "data/fashion_mnist_test_vectors.csv";
const TEST_LABELS_PATH: &str = "data/fashion_mnist_test_labels.csv";

const TRAIN_VECTORS_PATH: &str = "data/fashion_mnist_train_vectors.csv";
const TRAIN_LABELS_PATH: &str = "data/fashion_mnist_train_labels.csv";

const LEARNING_RATE: f64 = 0.001;
const NUM_EPOCHS: usize = 20;

// Run with `cargo run --release` to get reasonable performance
// Use `export RUST_LOG=info` environment variable to get progress information
fn main() -> Result<()> {
    let train_data = load_vectors(TRAIN_VECTORS_PATH)?;
    let train_labels = load_labels(TRAIN_LABELS_PATH)?;

    let test_data = load_vectors(TEST_VECTORS_PATH)?;
    let test_labels = load_labels(TEST_LABELS_PATH)?;

    let mut network = Model::default()
        .with_layer(784, Activator::Identity)
        .with_layer(128, Activator::Tanh)
        .with_layer(64, Activator::Tanh)
        .with_layer(10, Activator::Softmax)
        .with_learning_rate(LEARNING_RATE);

    let t0 = std::time::Instant::now();

    for epoch in 0..NUM_EPOCHS {
        for k in 0..train_data.len() {
            network.train_on_single(&train_data[k], &label_to_one_hot(train_labels[k]));
        }

        let elapsed = humantime::format_duration(std::time::Duration::from_secs(t0.elapsed().as_secs()));
        let acurracy = test_data_accuracy(&mut network, &test_data, &test_labels);

        println!(
            "Epoch: {}  🎯 Accuracy: {:.2}%  ⏳ Time elapsed: {}",
            epoch, acurracy, elapsed
        );
    }

    Ok(())
}

fn test_data_accuracy(network: &mut Model, test_data: &[Vec<f64>], test_labels: &[usize]) -> f64 {
    let correct = test_data
        .iter()
        .zip(test_labels)
        .filter(|(data, label)| {
            let prediction = network.predict(data);
            one_hot_to_label(prediction) == **label
        })
        .count();

    correct as f64 / test_data.len() as f64 * 100.0
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
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
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
