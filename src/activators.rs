pub enum Activator {
    Identity,
    ReLU,
    Tanh,
}

impl Activator {
    pub fn function(&self, x: f64) -> f64 {
        match self {
            Activator::Identity => x,
            Activator::ReLU => x.max(0.0),
            Activator::Tanh => x.tanh(),
        }
    }

    pub fn prime(&self, x: f64) -> f64 {
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
    pub fn apply(&self, values: &mut [f64]) {
        match self {
            OutputTransform::Identity => {}
            OutputTransform::Softmax => {
                let exp_sum: f64 = values.iter().map(|&x| x.exp()).sum();
                values.iter_mut().for_each(|x| *x = x.exp() / exp_sum);
            }
        }
    }
}
