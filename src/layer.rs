use crate::activators::Activator;
use rand::Rng;

pub(crate) struct Layer {
    pub(crate) size: usize,
    pub(crate) potential: Vec<f64>,
    pub(crate) activation: Vec<f64>,
    pub(crate) delta: Vec<f64>,
    pub(crate) inbound_weights: Vec<Vec<f64>>,
    pub(crate) activator: Activator,
}

impl Layer {
    pub(crate) fn new(size: usize, input_size: usize, activator: Activator) -> Self {
        let mut thread_rng = rand::thread_rng();

        let size_with_bias = size + 1; // +1 for bias neuron

        let input_weights: Vec<Vec<f64>> = (0..size_with_bias)
            .map(|_| {
                (0..input_size)
                    .map(|_| thread_rng.gen::<f64>() / input_size as f64)
                    .collect()
            })
            .collect();

        Self {
            size: size_with_bias,
            potential: vec![1.0; size_with_bias],
            activation: vec![1.0; size_with_bias],
            delta: vec![0.0; size_with_bias],
            inbound_weights: input_weights,
            activator,
        }
    }
}
