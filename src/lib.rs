use std::iter::once;

use rand::Rng;

#[allow(dead_code)]
enum Activator {
    Identity,
    Relu,
    Sigmoid,
    Tanh,
}

impl Activator {
    fn function(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => x,
            Activator::Relu => f64::max(0.0, x),
            Activator::Sigmoid => 1.0 / (1.0 + x.exp()),
            Activator::Tanh => x.tanh(),
        }
    }

    fn prime(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => 1.0,
            Activator::Relu => (x > 0.0) as usize as f64,
            Activator::Sigmoid => self.function(x) * (1.0 - self.function(x)),
            Activator::Tanh => 1.0 - self.function(x).powi(2),
        }
    }
}

struct Layer {
    size: usize,
    potential: Vec<f64>,
    activation: Vec<f64>,
    delta: Vec<f64>,
    previous_delta: Vec<f64>,
    input_weights: Vec<Vec<f64>>,
    activator: Activator,
}

impl Layer {
    fn new(size: usize, input_size: usize, activator: Activator) -> Self {
        let mut thread_rng = rand::thread_rng();

        let input_weights: Vec<Vec<f64>> = (0..size)
            .map(|_| (0..input_size).map(|_| thread_rng.gen()).collect())
            .collect();

        Self {
            size,
            potential: vec![1.0; size],
            activation: vec![1.0; size],
            delta: vec![0.0; size],
            previous_delta: vec![0.0; size],
            input_weights,
            activator,
        }
    }
}

pub struct SimpleNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    momentum: f64,
}

impl SimpleNetwork {
    pub fn new(shape: &[usize], learning_rate: f64, momentum: f64) -> Self {
        assert!(shape.len() >= 2, "Network must have at least two layers");

        // Add one extra neuron to each layer to model bias
        let input_layer = Layer::new(shape[0] + 1, 0, Activator::Identity);

        // Hidden layers and output layer take input from previous layer
        let remaining_layers = (1..shape.len()).map(|l| Layer::new(shape[l] + 1, shape[l - 1] + 1, Activator::Tanh));

        Self {
            learning_rate,
            momentum,
            layers: once(input_layer).chain(remaining_layers).collect(),
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
    }

    fn feed_backward(&mut self, output: &[f64]) {
        for l in (1..self.layers.len()).rev() {
            // Save previous delta for momentum
            self.layers[l].previous_delta = self.layers[l].delta.clone();

            // Calculate delta for each neuron
            (0..self.layers[l].size).for_each(|i| {
                let weighted_sum = if l == self.layers.len() - 1 {
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
                            + self.momentum * -self.layers[l].previous_delta[i];
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
        let mut network = SimpleNetwork::new(&[2, 4, 1], 0.1, 0.01);

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
