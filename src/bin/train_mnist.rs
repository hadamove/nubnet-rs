use anyhow::Result;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

use pv021_project::activators::Activator;
use pv021_project::model::{InputData, Model};

// Hyperparameters
const NUM_THREADS: usize = 32;
const NUM_EPOCHS: usize = 30;
const LEARNING_RATE: f64 = 0.001;
const LEARNING_RATE_DECAY: f64 = 0.003;

// Constants
const NUM_CLASSES: usize = 10;
const TRAIN_VECTORS_PATH: &str = "data/fashion_mnist_train_vectors.csv";
const TRAIN_LABELS_PATH: &str = "data/fashion_mnist_train_labels.csv";
const TEST_VECTORS_PATH: &str = "data/fashion_mnist_test_vectors.csv";
const TRAIN_PREDICTIONS_PATH: &str = "train_predictions.csv";
const TEST_PREDICTIONS_PATH: &str = "test_predictions.csv";

// Run with `cargo run --release` to get reasonable performance
fn main() -> Result<()> {
    println!("Loading data...");

    let train_data = load_vectors(TRAIN_VECTORS_PATH)?;
    let train_labels = load_labels(TRAIN_LABELS_PATH)?;

    let test_data = load_vectors(TEST_VECTORS_PATH)?;

    let mut model = Model::new(NUM_THREADS)
        .with_layer(784, Activator::Identity)
        .with_layer(128, Activator::Tanh)
        .with_layer(64, Activator::Tanh)
        .with_layer(10, Activator::Softmax);

    let t0 = std::time::Instant::now();
    let mut learning_rate = LEARNING_RATE;

    print!("Training... ");

    for epoch in 0..NUM_EPOCHS {
        // Shuffle the data at the beginning of each epoch
        let (train_data, train_labels) = shuffle_data(&train_data, &train_labels);

        // Slice the data into batches, the batch size is equal to the number of threads (each thread gets one input)
        for batch_start in (0..train_data.len()).step_by(NUM_THREADS) {
            let batch_end = usize::min(batch_start + NUM_THREADS, train_data.len());
            let batch_start = batch_end - NUM_THREADS;

            let inputs = &train_data[batch_start..batch_end];
            let labels = &train_labels[batch_start..batch_end];

            model.train_on_batch(inputs, labels, learning_rate);
        }

        // Apply decay to the learning rate
        learning_rate *= 1.0 / (1.0 + epoch as f64 * LEARNING_RATE_DECAY);
    }

    println!("Done (in {} seconds).", t0.elapsed().as_secs());
    println!("Exporting predictions...");

    export_predictions(&mut model, &train_data, TRAIN_PREDICTIONS_PATH)?;
    export_predictions(&mut model, &test_data, TEST_PREDICTIONS_PATH)?;

    Ok(())
}

fn load_vectors(filename: &str) -> Result<Vec<InputData>> {
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

    // Wrap the vectors in Arc so that they can be shared between threads by pointer without copying
    vectors.map(|v| v.into_iter().map(Arc::new).collect::<Vec<_>>())
}

fn load_labels(filename: &str) -> Result<Vec<InputData>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let labels: Result<Vec<usize>> = reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.parse::<usize>().map_err(Into::into))
        .collect();

    // Convert the labels to one hot vectors, then we wrap them in Arc similarly to the vectors
    labels.map(|v| v.into_iter().map(label_to_one_hot).map(Arc::new).collect::<Vec<_>>())
}

fn shuffle_data(data: &[InputData], labels: &[InputData]) -> (Vec<InputData>, Vec<InputData>) {
    let mut order = (0..data.len()).collect::<Vec<_>>();
    order.shuffle(&mut rand::thread_rng());

    // Clone here is cheap because we only copy the Arc pointers to the vectors, not the vectors themselves
    let shuffle_vec = |v: &[InputData]| order.iter().map(|&i| v[i].clone()).collect::<Vec<_>>();

    (shuffle_vec(data), shuffle_vec(labels))
}

fn export_predictions(model: &mut Model, data: &[InputData], filename: &str) -> Result<()> {
    let mut file = File::create(filename)?;

    for input in data {
        let prediction = model.predict(input);
        let label = one_hot_to_label(prediction);
        writeln!(file, "{}", label)?;
    }

    Ok(())
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
