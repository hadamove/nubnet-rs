pub enum Activator {
    Identity,
    Relu,
    Tanh,
    Softmax,
}

impl Activator {
    pub fn apply(&self, values: &mut [f64]) {
        match self {
            Activator::Identity => {}
            Activator::Relu => values.iter_mut().for_each(|x| *x = x.max(0.0)),
            Activator::Tanh => values.iter_mut().for_each(|x| *x = x.tanh()),
            Activator::Softmax => {
                let exp_sum: f64 = values.iter().map(|&x| x.exp()).sum();
                values.iter_mut().for_each(|x| *x = x.exp() / exp_sum);
            }
        }
    }

    pub fn apply_prime(&self, values: &mut [f64]) {
        match self {
            Activator::Identity => {}
            Activator::Relu => values.iter_mut().for_each(|x| *x = (*x > 0.0) as u8 as f64),
            Activator::Tanh => values.iter_mut().for_each(|x| *x = 1.0 - x.tanh().powi(2)),
            Activator::Softmax => {}
        }
    }
}
