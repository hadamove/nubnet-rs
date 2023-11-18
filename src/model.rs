use crate::{
    activators::{Activator, OutputTransform},
    layer::Layer,
};

pub struct Model {
    layers: Vec<Layer>,
    learning_rate: f64,
    output_transform: OutputTransform,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            layers: vec![],
            learning_rate: 0.1,
            output_transform: OutputTransform::Identity,
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

    /// Sets the output transform for the model. Default is identity.
    pub fn with_transform(mut self, transformer: OutputTransform) -> Self {
        self.output_transform = transformer;
        self
    }

    /// Trains the model on a single example. Internally, it performs forward pass, backward pass and weight update.
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

    fn feed_backward(&mut self, desired: &[f64]) {
        let output = self.layers.last_mut().expect("No layers in the model");
        // For output layer, delta is the difference between desired output and actual output multiplied by the derivative of the activation function
        // The first delta is for the bias neuron so we skip it
        for i in 1..output.size {
            output.delta[i] = (desired[i - 1] - output.activation[i]) * output.activator.prime(output.potential[i]);
        }

        for l in (1..self.layers.len() - 1).rev() {
            for i in 0..self.layers[l].size {
                // For hidden layers, delta is the weighted sum of deltas from the layer above
                let weighted_sum: f64 = (0..self.layers[l + 1].size)
                    .map(|k| self.layers[l + 1].delta[k] * self.layers[l + 1].input_weights[k][i])
                    .sum();

                self.layers[l].delta[i] = self.layers[l].activator.prime(self.layers[l].potential[i]) * weighted_sum;
            }
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

    fn get_output(&self) -> &[f64] {
        &self.layers.last().expect("No layers in the model").activation[1..]
    }
}

#[cfg(test)]
mod tests {
    use super::{Activator, Model};
    use more_asserts::*;
    use rand::{distributions::Uniform, Rng};

    // The training is not deterministic, so we need to allow some error
    const MAX_ERR: f64 = 0.1;

    #[test]
    fn test_xor() {
        let mut network = Model::default()
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

        for (input, desired) in data {
            let result = network.predict(input)[0];
            assert_le!(f64::abs(result - desired[0]), MAX_ERR);
        }
    }
}
