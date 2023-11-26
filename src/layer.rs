use crate::activators::Activator;
use rand::Rng;

pub struct Layer {
    pub size: usize,
    pub potential: Vec<f64>,
    pub activation: Vec<f64>,
    pub delta: Vec<f64>,
    pub inbound_weights: Vec<Vec<f64>>,
    pub activator: Activator,
}

impl Layer {
    pub fn new(size: usize, input_size: usize, activator: Activator) -> Self {
        let size_with_bias = size + 1;

        let mut thread_rng = rand::thread_rng();
        let inbound_weights: Vec<Vec<f64>> = (0..size_with_bias)
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
            inbound_weights,
            activator,
        }
    }
}
