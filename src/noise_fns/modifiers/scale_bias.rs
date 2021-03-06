// Copyright (c) 2017 The Noise-rs Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT
// or http://opensource.org/licenses/MIT>, at your option. All files in the
// project carrying such notice may not be copied, modified, or distributed
// except according to those terms.

use noise_fns::NoiseFn;

/// Noise function that applies a scaling factor and a bias to the output value
/// from the source function.
///
/// The function retrieves the output value from the source function, multiplies
/// it with the scaling factor, adds the bias to it, then outputs the value.
pub struct ScaleBias<'a, T: 'a> {
    /// Outputs a value.
    pub source: &'a NoiseFn<T>,

    /// Scaling factor to apply to the output value from the source function.
    /// The default value is 1.0.
    pub scale: f64,

    /// Bias to apply to the scaled output value from the source function.
    /// The default value is 0.0.
    pub bias: f64,
}

impl<'a, T> ScaleBias<'a, T> {
    pub fn new(source: &'a NoiseFn<T>) -> ScaleBias<'a, T> {
        ScaleBias {
            source: source,
            scale: 1.0,
            bias: 0.0,
        }
    }

    pub fn set_scale(self, scale: f64) -> ScaleBias<'a, T> {
        ScaleBias {
            scale: scale,
            ..self
        }
    }

    pub fn set_bias(self, bias: f64) -> ScaleBias<'a, T> {
        ScaleBias { bias: bias, ..self }
    }
}

impl<'a, T> NoiseFn<T> for ScaleBias<'a, T> {
    fn get(&self, point: T) -> f64 {
        (self.source.get(point)).mul_add(self.scale, self.bias)
    }
}
