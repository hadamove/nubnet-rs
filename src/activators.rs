#[derive(Clone)]
pub enum Activator {
    Identity,
    Tanh,
    Softmax,
}

impl Activator {
    pub fn apply(&self, values: &mut [f64]) {
        match self {
            Activator::Identity => {}
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
            Activator::Tanh => values.iter_mut().for_each(|x| *x = 1.0 - x.tanh().powi(2)),
            Activator::Softmax => {}
        }
    }
}
