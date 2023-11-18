use std::iter::once;

use rand::Rng;

enum Activator {
    Identity,
    Tanh,
}

impl Activator {
    fn function(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => x,
            Activator::Tanh => x.tanh(),
        }
    }

    fn prime(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => 1.0,
            Activator::Tanh => 1.0 - self.function(x).powi(2),
        }
    }
}

enum OutputTransformer {
    Softmax,
}

impl OutputTransformer {
    fn apply(&self, values: &mut [f64]) {
        match self {
            OutputTransformer::Softmax => {
                let exp_sum: f64 = values.iter().map(|&x| x.exp()).sum();
                values.iter_mut().for_each(|x| *x = x.exp() / exp_sum);
            }
        }
    }
}

struct Layer {
    size: usize,
    potential: Vec<f64>,
    activation: Vec<f64>,
    delta: Vec<f64>,
    input_weights: Vec<Vec<f64>>,
    activator: Activator,
}

impl Layer {
    fn new(size: usize, input_size: usize, activator: Activator) -> Self {
        let mut thread_rng = rand::thread_rng();

        let input_weights: Vec<Vec<f64>> = (0..size)
            .map(|_| {
                (0..input_size)
                    .map(|_| thread_rng.gen::<f64>() / input_size as f64)
                    .collect()
            })
            .collect();

        Self {
            size,
            potential: vec![1.0; size],
            activation: vec![1.0; size],
            delta: vec![0.0; size],
            input_weights,
            activator,
        }
    }
}

pub struct SimpleNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    output_transformer: OutputTransformer,
}

impl SimpleNetwork {
    pub fn new(shape: &[usize], learning_rate: f64) -> Self {
        assert!(shape.len() >= 2, "Network must have at least two layers");

        // Add one extra neuron to each layer to model bias
        let input_layer = Layer::new(shape[0] + 1, 0, Activator::Identity);

        // Hidden layers and output layer take input from previous layer
        let hidden_layers = (1..shape.len() - 1).map(|l| Layer::new(shape[l] + 1, shape[l - 1] + 1, Activator::Tanh));

        let output_layer = Layer::new(
            shape[shape.len() - 1] + 1,
            shape[shape.len() - 2] + 1,
            Activator::Identity,
        );

        Self {
            learning_rate,
            layers: once(input_layer)
                .chain(hidden_layers)
                .chain(once(output_layer))
                .collect(),
            output_transformer: OutputTransformer::Softmax,
        }
    }

    fn feed_forward(&mut self, input: &[f64]) {
        // Copy input to the first layer
        self.layers[0].activation[1..].copy_from_slice(input);

        // Calculate potentials and activations for each layer
        for l in 1..self.layers.len() {
            for i in 1..self.layers[l].size {
                let weighted_sum = (0..self.layers[l - 1].size)
                    .map(|k| self.layers[l].input_weights[i][k] * self.layers[l - 1].activation[k])
                    .sum::<f64>();

                self.layers[l].potential[i] = weighted_sum;
                self.layers[l].activation[i] = self.layers[l].activator.function(weighted_sum);
            }
        }

        let output_layer = self.layers.len() - 1;
        self.output_transformer
            .apply(&mut self.layers[output_layer].activation[1..]);
    }

    fn feed_backward(&mut self, output: &[f64]) {
        for l in (1..self.layers.len()).rev() {
            // Calculate delta for each neuron
            (0..self.layers[l].size).for_each(|i| {
                let weighted_sum = if l == self.layers.len() - 1 {
                    // TODO: bring this outside of for loop
                    // For output layer, delta is the difference between desired output and actual output
                    output[i] - self.layers[l].activation[i]
                } else {
                    // For hidden layers, delta is the weighted sum of deltas from the layer above
                    (0..self.layers[l + 1].size)
                        .map(|k| self.layers[l + 1].delta[k] * self.layers[l + 1].input_weights[k][i])
                        .sum()
                };

                self.layers[l].delta[i] = self.layers[l].activator.prime(self.layers[l].potential[i]) * weighted_sum;
            });
        }
    }

    fn update_weights(&mut self) {
        for l in 1..self.layers.len() {
            for i in 0..self.layers[l].size {
                for j in 0..self.layers[l].input_weights[i].len() {
                    self.layers[l].input_weights[i][j] +=
                        self.learning_rate * self.layers[l].delta[i] * self.layers[l - 1].activation[j]
                }
            }
        }
    }

    pub fn train_on_single(&mut self, input: &[f64], output: &[f64]) {
        assert_eq!(input.len(), self.layers[0].size - 1);
        assert_eq!(output.len(), self.layers[self.layers.len() - 1].size - 1);

        self.feed_forward(input);

        // TODO: process all outputs to have 1.0 at the beginning in advance (for the bias neuron)
        let output = once(1.0).chain(output.iter().copied()).collect::<Vec<_>>();

        self.feed_backward(&output);
        self.update_weights();
    }

    pub fn predict(&mut self, input: &[f64]) -> &[f64] {
        assert_eq!(input.len(), self.layers[0].size - 1);

        self.feed_forward(input);
        &self.layers[self.layers.len() - 1].activation[1..]
    }
}

#[cfg(test)]
mod tests {
    use super::SimpleNetwork;
    use more_asserts::*;
    use rand::{distributions::Uniform, Rng};

    #[test]
    fn test_xor() {
        let mut network = SimpleNetwork::new(&[2, 4, 1], 0.1);

        let data = [
            (&[0., 0.], &[0.]),
            (&[1., 0.], &[1.]),
            (&[0., 1.], &[1.]),
            (&[1., 1.], &[0.]),
        ];

        let mut thread_rng = rand::thread_rng();

        for _ in 0..10_000 {
            let k = thread_rng.sample(Uniform::new(0, data.len()));
            let (input, desired) = data[k];

            network.train_on_single(input, desired);
        }

        const ALLOWED_ERROR: f64 = 0.1;

        for (input, desired) in data {
            let result = network.predict(input)[0];
            println!(
                "[{:.3}, {:.3}], [{:.3}] -> [{:.3}]",
                input[0], input[1], desired[0], result
            );

            assert_le!(f64::abs(result - desired[0]), ALLOWED_ERROR);
        }
    }
}
