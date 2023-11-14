use anyhow::Result;
use clap::Parser;

use pv021_project::SimpleNetwork;

const NUM_CLASSES: usize = 10;
const NUM_TRAINING_EXAMPLES: usize = 10_000;

#[derive(Parser)]
struct CliArguments {
    #[arg(short, long, default_value = "data/fashion_mnist_test_vectors.csv")]
    input_path: String,
    #[arg(short, long, default_value = "data/fashion_mnist_test_labels.csv")]
    labels_path: String,

    #[arg(short = 'r', long, default_value_t = 0.1)]
    learning_rate: f64,
    #[arg(short, long, default_value_t = 0.0)]
    momentum: f64,
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,
}

// Run with `cargo run --release` to get reasonable performance
fn main() -> Result<()> {
    let args = CliArguments::parse();

    let data = load_csv_data(&args.input_path)?;

    // Transform data from 0..255 to 0..1
    // TODO: Refactor this
    let data: Vec<Vec<_>> = data
        .into_iter()
        .map(|row| row.into_iter().map(|x| x / 255.0).collect())
        .collect();

    let labels = load_csv_labels(&args.labels_path)?;

    let mut network = SimpleNetwork::new(&[784, 128, 64, 10], args.learning_rate, args.momentum);

    // Very dumb training loop, record by record
    // TODO: shuffle the data and use batches
    for it in 0..args.epochs * NUM_TRAINING_EXAMPLES {
        let k = rand::random::<usize>() % data.len();

        network.train_on_single(&data[k], &label_to_one_hot(labels[k]));

        // Show accuracy on 1000 random examples (should be separated from training data in the future)
        if it % 10_000 == 0 {
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

    println!(
        "Final accuracy: {:.2}%",
        correct as f64 / data.len() as f64 * 100.0
    );

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

fn load_csv_data(filename: &str) -> Result<Vec<Vec<f64>>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)?;

    let data = reader
        .records()
        .map(|record| {
            record?
                .iter()
                .map(|field| field.parse::<f64>().map_err(|e| e.into()))
                .collect()
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(data)
}

fn load_csv_labels(filename: &str) -> Result<Vec<usize>> {
    let data = load_csv_data(filename)?;

    // Each row should contain a single value
    Ok(data.into_iter().flatten().map(|x| x as usize).collect())
}
