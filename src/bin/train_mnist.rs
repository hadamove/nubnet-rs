use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::Result;
use pv021_project::{activators::Activator, model::Model};

const NUM_CLASSES: usize = 10;
const NUM_TRAINING_EXAMPLES: usize = 10_000;

const INPUT_PATH: &str = "data/fashion_mnist_test_vectors.csv";
const LABELS_PATH: &str = "data/fashion_mnist_test_labels.csv";

const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 30;

// Run with `cargo run --release` to get reasonable performance
fn main() -> Result<()> {
    let data = load_vectors(INPUT_PATH)?;
    let labels = load_labels(LABELS_PATH)?;

    let mut network = Model::default()
        .with_layer(784, Activator::Identity)
        .with_layer(128, Activator::Tanh)
        .with_layer(64, Activator::Tanh)
        .with_layer(10, Activator::Softmax)
        .with_learning_rate(LEARNING_RATE);

    // Very dumb training loop, record by record
    // TODO: shuffle the data and use batches
    for it in 0..EPOCHS * NUM_TRAINING_EXAMPLES {
        let k = rand::random::<usize>() % data.len();

        network.train_on_single(&data[k], &label_to_one_hot(labels[k]));

        // Show accuracy on 1000 random examples (should be separated from training data in the future)
        if it % 10_000 == 9_999 {
            let mut correct = 0;
            for _ in 0..1000 {
                let k = rand::random::<usize>() % data.len();

                // Add breakpoint here and see prediction
                // It seems that we will need to use softmax and cross-entropy
                let prediction = network.predict(&data[k]);

                if one_hot_to_label(prediction) == labels[k] {
                    correct += 1;
                }
            }

            println!(
                "Epoch: {}, accuracy: {:.2}%",
                it / NUM_TRAINING_EXAMPLES,
                correct as f64 / 10.
            );
        }
    }

    // Check accuracy on the whole dataset
    let mut correct = 0;
    for k in 0..data.len() {
        let prediction = network.predict(&data[k]);

        if one_hot_to_label(prediction) == labels[k] {
            correct += 1;
        }
    }

    println!("Final accuracy: {:.2}%", correct as f64 / data.len() as f64 * 100.0);

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
