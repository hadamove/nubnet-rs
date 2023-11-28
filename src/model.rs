use std::sync::{mpsc, Arc, Barrier, RwLock};

use crate::{activators::Activator, layer::Layer};

// TODO: Better naming
struct ThreadState {
    layers: Vec<Layer>,
}

impl ThreadState {
    fn feed_forward(&mut self, input: &[f64]) {
        self.layers[0].activation[1..].copy_from_slice(input);

        self.layers.iter_mut().reduce(|layer_below, layer| {
            {
                let inbound_weights = layer.inbound_weights.try_read().unwrap();
                layer.potential[1..].iter_mut().zip(1..).for_each(|(potential, i)| {
                    *potential = (0..layer_below.size)
                        .map(|j| inbound_weights[i][j] * layer_below.activation[j])
                        .sum();
                });
            }

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
            let outbound_weights = layer_above.inbound_weights.try_read().unwrap();
            layer.activator.apply_prime(&mut layer.potential);
            layer.delta.iter_mut().zip(0..).for_each(|(delta, i)| {
                *delta = layer.potential[i]
                    * (0..layer_above.size)
                        .map(|j| layer_above.delta[j] * outbound_weights[j][i])
                        .sum::<f64>();
            });

            layer
        });
    }
}

type InputData = Arc<Vec<f64>>;

pub struct Model<const N: usize> {
    // TODO: Better naming
    main_state: ThreadState,
    barrier: Arc<Barrier>,
    // TODO: Better naming
    threads: Vec<Arc<RwLock<ThreadState>>>,
    channels: Vec<mpsc::Sender<(InputData, InputData)>>,
    learning_rate: f64,
}

fn thread_lifecycle(
    thread_state: Arc<RwLock<ThreadState>>,
    receiver: mpsc::Receiver<(InputData, InputData)>,
    barrier: Arc<Barrier>,
) {
    loop {
        // receiver.recv() blocks until main thread feeds us more data
        // between the barrier.wait() and the receiver.recv() the main thread updates the weights
        let Ok((input, desired)) = receiver.recv() else {
            // The main thread has dropped the sender, so we can exit
            break;
        };

        // Acquire lock to state and feed forward and backward
        {
            let mut thread_state = thread_state.try_write().unwrap();
            thread_state.feed_forward(&input);
            thread_state.feed_backward(&desired);
        }
        // End of scope -> lock is released

        // Wait for all threads to finish forward and backward pass
        barrier.wait();
    }
}

impl<const N: usize> Default for Model<N> {
    fn default() -> Self {
        let barrier = Arc::new(Barrier::new(N + 1)); // shouldnt this be N+1?
        let mut threads = Vec::new();
        let mut channels = Vec::new();

        for _ in 0..N {
            let barrier = barrier.clone();
            let thread_state = Arc::new(RwLock::new(ThreadState { layers: Vec::new() }));
            let (sender, receiver) = mpsc::channel();

            threads.push(thread_state.clone());
            channels.push(sender);

            std::thread::spawn(move || {
                thread_lifecycle(thread_state, receiver, barrier);
            });
        }

        Self {
            main_state: ThreadState { layers: Vec::new() },
            barrier,
            threads,
            channels,
            learning_rate: 0.1,
        }
    }
}

impl<const N: usize> Model<N> {
    pub fn with_layer(mut self, size: usize, activator: Activator) -> Self {
        let input_size = self.main_state.layers.last().map(|layer| layer.size).unwrap_or(N);

        let new_layer = Layer::new(size, input_size, activator);
        for thread in self.threads.iter_mut() {
            thread.try_write().unwrap().layers.push(new_layer.clone());
        }

        self.main_state.layers.push(new_layer);

        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// The inputs have to be `Arc<Vec<f64>>` so that they can be sent to the child threads safely and without cloning.
    pub fn train_on_batch(&mut self, inputs: &[InputData], labels: &[InputData]) {
        // Send inputs and labels to all threads
        for ((input, desired), thread_channel) in inputs.iter().zip(labels).zip(&self.channels) {
            thread_channel.send((input.clone(), desired.clone())).unwrap();
        }

        // Wait for all child threads to finish forward and backward pass
        self.barrier.wait();

        // Update weights
        self.update_weights();
    }

    /// Predicts the output for the given input. The input slice must be of the same size as the first layer.
    pub fn predict(&mut self, input: &[f64]) -> &[f64] {
        // Run the forward pass on the first thread
        self.main_state.feed_forward(input);
        self.get_output()
    }

    fn update_weights(&mut self) {
        for thread_state in self.threads.iter_mut() {
            // It should be safe to aquire the lock here, since only the main thread is running, and the child threads are waiting for the next input
            let mut thread_state = thread_state.try_write().unwrap();

            thread_state.layers.iter_mut().reduce(|layer_below, layer| {
                {
                    let mut inbound_weights = layer.inbound_weights.try_write().unwrap();
                    inbound_weights.iter_mut().zip(0..).for_each(|(weights, i)| {
                        weights.iter_mut().zip(0..layer_below.size).for_each(|(weight, j)| {
                            *weight += self.learning_rate * layer.delta[i] * layer_below.activation[j]
                        });
                    });
                }

                layer
            });
        }
    }

    fn get_output(&self) -> &[f64] {
        &self
            .main_state
            .layers
            .last()
            .expect("No layers in the model")
            .activation[1..]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{Activator, Model};

    const NUM_THREADS: usize = 1;

    #[test]
    fn test_xor() {
        let mut network = Model::<NUM_THREADS>::default()
            .with_layer(2, Activator::Identity)
            .with_layer(2, Activator::Tanh)
            .with_layer(1, Activator::Identity)
            .with_learning_rate(0.03);

        let data = [
            Arc::new(vec![0., 0.]),
            Arc::new(vec![1., 0.]),
            Arc::new(vec![0., 1.]),
            Arc::new(vec![1., 1.]),
        ];

        let labels = [
            Arc::new(vec![0.]),
            Arc::new(vec![1.]),
            Arc::new(vec![1.]),
            Arc::new(vec![0.]),
        ];

        for _ in 0..10_000 {
            network.train_on_batch(&data, &data);
        }

        for (input, desired) in data.iter().zip(&labels) {
            let result = network.predict(&input)[0];
            println!("{} -> {}", input[0], result);
            // assert!(f64::abs(result - desired[0]) < 0.1);
        }
    }
}
