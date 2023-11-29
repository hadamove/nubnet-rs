use std::sync::{mpsc, Arc, Barrier, RwLock};

use crate::{activators::Activator, layer::Layer};

pub struct Model<const N_THREADS: usize> {
    state: ModelState,
    barrier: Arc<Barrier>,
    threads: Vec<ThreadHandle>,
    learning_rate: f64,
}

impl<const N_THREADS: usize> Model<N_THREADS> {
    /// Initializes the model with `N_THREADS` child threads and a learning rate.
    pub fn new(learning_rate: f64) -> Self {
        // Create a barrier synchronizes N_THREADS child threads and the main thread
        // The FW and BW pass is done in the child threads, the weight update is done in the main thread
        let barrier = Arc::new(Barrier::new(N_THREADS + 1));

        // Spawn N_THREADS threads for computing the FW and BW pass
        let threads = (0..N_THREADS)
            .map(|_| {
                let barrier = barrier.clone();
                // Create a channel for sending inputs and labels from the main thread to the child threads
                let (sender, receiver) = mpsc::channel();

                // One copy of the pointer to the state for the main thread, one for the child thread
                let state = Arc::new(RwLock::new(ModelState { layers: Vec::new() }));
                let state_clone = state.clone();

                std::thread::spawn(move || {
                    Self::run_thread(state_clone, receiver, barrier);
                });

                ThreadHandle { state, sender }
            })
            .collect();

        Self {
            state: ModelState { layers: Vec::new() },
            barrier,
            threads,
            learning_rate,
        }
    }

    /// Adds a new layer to the model. The layer is added to all child threads states and the main thread state.
    pub fn with_layer(mut self, size: usize, activator: Activator) -> Self {
        // The input size of the new layer is the size of the last layer, or 0 if there are no layers yet (input layer)
        let input_size = self.state.layers.last().map(|layer| layer.size).unwrap_or(0);

        let new_layer = Layer::new(size, input_size, activator);
        for thread in self.threads.iter_mut() {
            thread.state.try_write().unwrap().layers.push(new_layer.clone());
        }

        self.state.layers.push(new_layer);

        self
    }

    /// The inputs have to be `Arc<Vec<f64>>` so that they can be sent to the child threads safely and without cloning.
    pub fn train_on_batch(&mut self, inputs: &[InputData], labels: &[InputData]) {
        // Send inputs and labels to all threads
        for ((input, desired), thread) in inputs.iter().zip(labels).zip(&self.threads) {
            thread.sender.send((input.clone(), desired.clone())).unwrap();
        }

        // Wait for all child threads to finish forward and backward pass
        self.barrier.wait();

        // Update weights
        self.update_weights();
    }

    /// Predicts the output for the given input. The input slice must be of the same size as the first layer.
    pub fn predict(&mut self, input: &[f64]) -> &[f64] {
        // Run the forward pass on the main thread
        self.state.feed_forward(input);
        self.get_output()
    }

    fn run_thread(
        state: Arc<RwLock<ModelState>>,
        receiver: mpsc::Receiver<(InputData, InputData)>,
        barrier: Arc<Barrier>,
    ) {
        loop {
            // receiver.recv() blocks this thread until main thread feeds us more data
            let Ok((input, desired)) = receiver.recv() else {
                // The main thread has dropped the sender, so we can exit
                break;
            };

            // Acquire lock to state and feed forward and backward
            {
                let mut state = state.try_write().unwrap();
                state.feed_forward(&input);
                state.feed_backward(&desired);
            }
            // End of scope -> lock is released

            // Wait for all threads to finish forward and backward pass
            barrier.wait();
        }
    }

    fn update_weights(&mut self) {
        for thread in self.threads.iter_mut() {
            // It is safe to aquire the lock here, the child threads are waiting for the next input
            let mut state = thread.state.try_write().unwrap();

            state.layers.iter_mut().reduce(|layer_below, layer| {
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
        &self.state.layers.last().expect("No layers in the model").activation[1..]
    }
}

type InputData = Arc<Vec<f64>>;

struct ThreadHandle {
    state: Arc<RwLock<ModelState>>,
    sender: mpsc::Sender<(InputData, InputData)>,
}

struct ModelState {
    layers: Vec<Layer>,
}

impl ModelState {
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
