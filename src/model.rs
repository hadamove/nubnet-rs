use crate::{activators::Activator, layer::Layer};

pub struct Model {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            layers: vec![],
            learning_rate: 0.1,
        }
    }
}

impl Model {
    /// Adds new layer on top of the layer stack and initializes weights between the layers.
    /// The first call to this method will create the input layer which does not have any incoming weights.
    pub fn with_layer(mut self, size: usize, activator: Activator) -> Self {
        let input_size = self.layers.last().map_or(0, |l| l.size);
        self.layers.push(Layer::new(size, input_size, activator));
        self
    }

    /// Sets the learning rate for the model. Default is 0.1.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Trains the model on a single example. Internally, it performs forward pass, backward pass and weights update.
    /// The input and desired slices must be of the same size as the first and last layers, respectively.
    pub fn train_on_single(&mut self, input: &[f64], desired: &[f64]) {
        assert_eq!(input.len(), self.layers[0].size - 1);
        assert_eq!(desired.len(), self.layers[self.layers.len() - 1].size - 1);

        self.feed_forward(input);
        self.feed_backward(desired);
        self.update_weights();
    }

    /// Predicts the output for the given input. The input slice must be of the same size as the first layer.
    pub fn predict(&mut self, input: &[f64]) -> &[f64] {
        assert_eq!(input.len(), self.layers[0].size - 1);

        self.feed_forward(input);
        self.get_output()
    }

    fn feed_forward(&mut self, input: &[f64]) {
        self.layers[0].activation[1..].copy_from_slice(input);

        self.layers.iter_mut().reduce(|layer_below, layer| {
            layer.potential[1..].iter_mut().zip(1..).for_each(|(potential, i)| {
                *potential = (0..layer_below.size)
                    .map(|j| layer.inbound_weights[i][j] * layer_below.activation[j])
                    .sum();
            });

            layer.activation[1..].copy_from_slice(&layer.potential[1..]);
            layer.activator.apply(&mut layer.activation[1..]);

            layer
        });
    }

    fn feed_backward(&mut self, desired: &[f64]) {
        let output_layer = self.layers.last_mut().expect("No layers in the model");
        for i in 1..output_layer.size {
            output_layer.delta[i] = desired[i - 1] - output_layer.activation[i];
        }

        self.layers[1..].iter_mut().rev().reduce(|layer_above, layer| {
            layer.activator.apply_prime(&mut layer.potential);
            layer.delta.iter_mut().zip(0..).for_each(|(delta, i)| {
                *delta = layer.potential[i]
                    * (0..layer_above.size)
                        .map(|j| layer_above.delta[j] * layer_above.inbound_weights[j][i])
                        .sum::<f64>();
            });

            layer
        });
    }

    fn update_weights(&mut self) {
        self.layers.iter_mut().reduce(|layer_below, layer| {
            layer.inbound_weights.iter_mut().zip(0..).for_each(|(weights, i)| {
                weights
                    .iter_mut()
                    .zip(0..layer_below.size)
                    .for_each(|(weight, j)| *weight += self.learning_rate * layer.delta[i] * layer_below.activation[j]);
            });

            layer
        });
    }

    fn get_output(&self) -> &[f64] {
        &self.layers.last().expect("No layers in the model").activation[1..]
    }
}

#[cfg(test)]
mod tests {
    use super::{Activator, Model};

    #[test]
    fn test_xor() {
        let mut network = Model::default()
            .with_layer(2, Activator::Identity)
            .with_layer(2, Activator::Tanh)
            .with_layer(1, Activator::Identity)
            .with_learning_rate(0.03);

        let data = [
            (&[0., 0.], &[0.]),
            (&[1., 0.], &[1.]),
            (&[0., 1.], &[1.]),
            (&[1., 1.], &[0.]),
        ];

        for i in 0..10_000 {
            let (input, desired) = data[i % data.len()];
            network.train_on_single(input, desired);
        }

        for (input, [desired]) in data {
            let result = network.predict(input)[0];
            assert!(f64::abs(result - desired) < 0.1);
        }
    }
}
