// Copyright 2015 The noise-rs developers. 
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! An example of using simplex noise

#![feature(core)]

extern crate noise;

use noise::{simplex2, simplex3, simplex4, Seed, Point2};

mod debug;

fn main() {
    debug::render_png("simplex2.png", &Seed::new(0), 1024, 1024, scaled_simplex2);
    debug::render_png("simplex3.png", &Seed::new(0), 1024, 1024, scaled_simplex3);
    debug::render_png("simplex4.png", &Seed::new(0), 1024, 1024, scaled_simplex4);
    println!("\nGenerated simplex2.png, simplex3.png, and simplex4.png");
}

fn scaled_simplex2(seed: &Seed, point: &Point2<f32>) -> f32 {
    simplex2(seed, &[point[0] / 16.0, point[1] / 16.0])
}

fn scaled_simplex3(seed: &Seed, point: &Point2<f32>) -> f32 {
    simplex3(seed, &[point[0] / 16.0, point[1] / 16.0, 0.0])
}

fn scaled_simplex4(seed: &Seed, point: &Point2<f32>) -> f32 {
    simplex4(seed, &[point[0] / 16.0, point[1] / 16.0, 0.0, 0.0])
}
