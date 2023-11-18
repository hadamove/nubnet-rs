use std::iter::once;

use rand::Rng;

pub enum Activator {
    Identity,
    ReLU,
    Tanh,
}

impl Activator {
    fn function(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => x,
            Activator::ReLU => x.max(0.0),
            Activator::Tanh => x.tanh(),
        }
    }

    fn prime(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => 1.0,
            Activator::ReLU => (x > 0.0) as u8 as f64,
            Activator::Tanh => 1.0 - self.function(x).powi(2),
        }
    }
}

#[derive(Default)]
pub enum OutputTransform {
    #[default]
    Identity,
    Softmax,
}

impl OutputTransform {
    fn apply(&self, values: &mut [f64]) {
        match self {
            OutputTransform::Identity => {}
            OutputTransform::Softmax => {
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

#[derive(Default)]
pub struct SimpleNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    output_transform: OutputTransform,
}

impl SimpleNetwork {
    pub fn with_layer(mut self, size: usize, activator: Activator) -> Self {
        let input_size = self.layers.last().map_or(0, |l| l.size);
        self.layers.push(Layer::new(size + 1, input_size, activator)); // Plus one for the bias neuron
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_transform(mut self, transformer: OutputTransform) -> Self {
        self.output_transform = transformer;
        self
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

        // Apply transform to the output layer
        let output_layer = self.layers.len() - 1;
        self.output_transform
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
    use super::{Activator, SimpleNetwork};
    use more_asserts::*;
    use rand::{distributions::Uniform, Rng};

    #[test]
    fn test_xor() {
        let mut network = SimpleNetwork::default()
            .with_layer(2, Activator::Identity)
            .with_layer(2, Activator::Tanh)
            .with_layer(1, Activator::Identity)
            .with_learning_rate(0.1);

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
