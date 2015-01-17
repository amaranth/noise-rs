// Copyright 2015 The noise-rs developers. For a full listing of the authors,
// refer to the AUTHORS file at the top-level directory of this distribution.
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

use std::num::Float;

use {gradient, math, Seed};

fn skew2_constant<T: Float>() -> T { math::cast(0.366025403784) } // (sqrt(3.0) - 1.0) / 2.0
fn skew3_constant<T: Float>() -> T { math::cast(0.333333333333) } // 1.0 / 3.0
fn skew4_constant<T: Float>() -> T { math::cast(0.309016994375) } // (sqrt(5.0) - 1.0) / 4.0
fn unskew2_constant<T: Float>() -> T { math::cast(0.211324865405) } // (3.0 - sqrt(3.0)) / 6.0
fn unskew3_constant<T: Float>() -> T { math::cast(0.166666666667) } // 1.0 / 6.0
fn unskew4_constant<T: Float>() -> T { math::cast(0.138196601125) } // (5.0 - sqrt(5.0)) / 20.0

fn simplex2_size<T: Float>() -> T { math::cast(0.5) }
fn simplex3_size<T: Float>() -> T { math::cast(0.6) }
fn simplex4_size<T: Float>() -> T { math::cast(0.6) }

fn norm2_constant<T: Float>() -> T { math::cast(70.14805) }
fn norm3_constant<T: Float>() -> T { math::cast(23.93986) }
fn norm4_constant<T: Float>() -> T { math::cast(27.24064) }

/// 2-dimensional simplex noise
pub fn simplex2<T: Float>(seed: &Seed, point: &math::Point2<T>) -> T {
    #[inline(always)]
    fn gradient<T: Float>(seed: &Seed, whole: math::Point2<isize>, frac: math::Point2<T>) -> T {
        let attn = simplex2_size::<T>() - math::dot2(frac, frac);
        if attn > math::cast(0) {
            math::pow4(attn) * math::dot2(frac, gradient::get2(seed.get2(whole)))
        } else {
            math::cast(0)
        }
    }

    let zero: T = math::cast(0.0);
    let one: T = math::cast(1.0);
    let two: T = math::cast(2.0);

    // Skew the (x,y) space to determine which cell of 2 simplices we're in
 	let skew_offset = (point[0] + point[1]) * skew2_constant();
    let x_cell = (point[0] + skew_offset).floor();
    let y_cell = (point[1] + skew_offset).floor();

    // Unskew the cell origin back to (x,y) space
    let unskew_offset = (x_cell + y_cell) * unskew2_constant();
    let x_origin = x_cell - unskew_offset;
    let y_origin = y_cell - unskew_offset;

    // Compute the (x,y) distances from the cell origin
    let dx0 = point[0] - x_origin;
    let dy0 = point[1] - y_origin;

    // Find out whether we are above or below the x=y diagonal to
    // determine which of the two triangles we're in.
    let (x1_offset, y1_offset) = if dx0 > dy0 { (one, zero) } else { (zero, one) };

    // Compute the (x,y) distances from the second point
    let dx1 = dx0 - x1_offset + unskew2_constant();
    let dy1 = dy0 - y1_offset + unskew2_constant();

    // Compute the (x,y) distances from the third point
    let dx2 = dx0 - one + two * unskew2_constant();
    let dy2 = dy0 - one + two * unskew2_constant();

    let n0 = gradient(seed, [math::cast(x_cell), math::cast(y_cell)], [dx0, dy0]);
    let n1 = gradient(seed, [math::cast(x_cell + x1_offset), math::cast(y_cell + y1_offset)], [dx1, dy1]);
    let n2 = gradient(seed, [1is + math::cast::<_, isize>(x_cell), 1is + math::cast::<_, isize>(y_cell)], [dx2, dy2]);

    // Sum up and scale the result to cover the range [-1,1]
    return (n0 + n1 + n2) * norm2_constant();
}


/// 3-dimensional simplex noise
pub fn simplex3<T: Float>(seed: &Seed, point: &math::Point3<T>) -> T {
    #[inline(always)]
    fn gradient<T: Float>(seed: &Seed, whole: math::Point3<isize>, frac: math::Point3<T>) -> T {
        let attn = simplex3_size::<T>() - math::dot3(frac, frac);
        if attn > math::cast(0) {
            math::pow4(attn) * math::dot3(frac, gradient::get3(seed.get3(whole)))
        } else {
            math::cast(0)
        }
    }

    let zero: T = math::cast(0.0);
    let one: T = math::cast(1.0);
    let two: T = math::cast(2.0);
    let three: T = math::cast(3.0);

    // Skew the (x,y,z) space to determine which cell of 3 simplices we're in
    let skew_offset = (point[0] + point[1] + point[2]) * skew3_constant();
    let x_cell = (point[0] + skew_offset).floor();
    let y_cell = (point[1] + skew_offset).floor();
    let z_cell = (point[2] + skew_offset).floor();

    // Unskew the cell origin back to (x,y,z) space
    let unskew_offset = (x_cell + y_cell + z_cell) * unskew3_constant();
    let x_origin = x_cell - unskew_offset;
    let y_origin = y_cell - unskew_offset;
    let z_origin = z_cell - unskew_offset;

    // Compute the (x,y,z) distances from the cell origin
    let dx0 = point[0] - x_origin;
    let dy0 = point[1] - y_origin;
    let dz0 = point[2] - z_origin;

    // Determine which of the six possible tetrahedra we're in
    let (offset1, offset2) = if dx0 >= dy0 {
        if dy0 >= dz0 {
            ([one, zero, zero], [one, one, zero])
        } else if dx0 >= dz0 {
            ([one, zero, zero], [one, zero, one])
        } else {
            ([zero, zero, one], [one, zero, one])
        }
    } else {
        if dy0 < dz0 {
            ([zero, zero, one], [zero, one, one])
        } else if dx0 < dz0 {
            ([zero, one, zero], [zero, one, one])
        } else {
            ([zero, one, zero], [one, one, zero])
        }
    };

    // Compute the (x,y,z) distances from the second point
    let dx1 = dx0 - offset1[0] + unskew3_constant();
    let dy1 = dy0 - offset1[1] + unskew3_constant();
    let dz1 = dz0 - offset1[2] + unskew3_constant();

    // Compute the (x,y,z) distances from the third point
    let dx2 = dx0 - offset2[0] + two * unskew3_constant();
    let dy2 = dy0 - offset2[1] + two * unskew3_constant();
    let dz2 = dz0 - offset2[2] + two * unskew3_constant();

    // Compute the (x,y,z) distances from the fourth point
    let dx3 = dx0 - one + three * unskew3_constant();
    let dy3 = dy0 - one + three * unskew3_constant();
    let dz3 = dz0 - one + three * unskew3_constant();

    let n0 = gradient(seed, [math::cast(x_cell), math::cast(y_cell), math::cast(z_cell)], [dx0, dy0, dz0]);
    let n1 = gradient(seed, [math::cast(x_cell + offset1[0]), math::cast(y_cell + offset1[1]), math::cast(z_cell + offset1[2])], [dx1, dy1, dz1]);
    let n2 = gradient(seed, [math::cast(x_cell + offset2[0]), math::cast(y_cell + offset2[1]), math::cast(z_cell + offset2[2])], [dx2, dy2, dz2]);
    let n3 = gradient(seed, [1is + math::cast::<_, isize>(x_cell), 1is + math::cast::<_, isize>(y_cell), 1is + math::cast::<_, isize>(z_cell)], [dx3, dy3, dz3]);

    // Sum up and scale the result to cover the range [-1,1]
    return (n0 + n1 + n2 + n3) * norm3_constant();
}

/// 4-dimensional simplex noise
pub fn simplex4<T: Float>(seed: &Seed, point: &math::Point4<T>) -> T {
    #[inline(always)]
    fn gradient<T: Float>(seed: &Seed, whole: math::Point4<isize>, frac: math::Point4<T>) -> T {
        let attn = simplex4_size::<T>() - math::dot4(frac, frac);
        if attn > math::cast(0) {
            math::pow4(attn) * math::dot4(frac, gradient::get4(seed.get4(whole)))
        } else {
            math::cast(0)
        }
    }

    let zero: T = math::cast(0.0);
    let one: T = math::cast(1.0);
    let two: T = math::cast(2.0);
    let three: T = math::cast(3.0);
    let four: T = math::cast(4.0);

    // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
    let skew_offset = (point[0] + point[1] + point[2] + point[3]) * skew4_constant();
    let x_cell = (point[0] + skew_offset).floor();
    let y_cell = (point[1] + skew_offset).floor();
    let z_cell = (point[2] + skew_offset).floor();
    let w_cell = (point[3] + skew_offset).floor();

    // Unskew the cell origin back to (x,y,z,w) space
    let unskew_offset = (x_cell + y_cell + z_cell + w_cell) * unskew4_constant();
    let x_origin = x_cell - unskew_offset;
    let y_origin = y_cell - unskew_offset;
    let z_origin = z_cell - unskew_offset;
    let w_origin = w_cell - unskew_offset;

    // Compute the (x,y,z,w) distances from the cell origin
    let dx0 = point[0] - x_origin;
    let dy0 = point[1] - y_origin;
    let dz0 = point[2] - z_origin;
    let dw0 = point[3] - w_origin;

    // Determine magnitude ordering of dx0, dy0, dz0, dw0 to find out which
    // of the 24 possible simplices we're in
    let is_x = math::step3([dy0, dz0, dw0], [dx0, dx0, dx0]);
    let mut offset0 = [is_x[0] + is_x[1] + is_x[2], one - is_x[0], one - is_x[1], one - is_x[2]];

    let is_y = math::step2([dz0, dw0], [dy0, dy0]);
    offset0[1] = offset0[1] + is_y[0] + is_y[1];
    offset0[2] = offset0[2] + one - is_y[0];
    offset0[3] = offset0[3] + one - is_y[1];

    let is_z = math::step(dw0, dz0); 
    offset0[2] = offset0[2] + is_z;
    offset0[3] = offset0[3] + one - is_z;

    // offset0 now contains the unique values 0,1,2,3 in each channel
    // 3 for the channel greater than other channels
    // 2 for the channel that is less than one but greater than the others
    // 1 for the channel that is greater than one other
    // 0 for the channel less than other channels
    // Equality ties are broken in favor of first x, then y, then z, then w

    // offset3 contains 1 in each channel that was 1, 2, or 3
    let offset3 = math::clamp4(offset0, zero, one);
    // offset2 contains 1 in each channel that was 1 or 2
    let offset2 = math::clamp4(math::sub4(offset0, math::one4()), zero, one);
    // offset 1 contains 1 in each channel that was 1
    let offset1 = math::clamp4(math::sub4(offset0, [two, two, two, two]), zero, one);

    // Compute the (x,y,z,w) distances from the second point
    let dx1 = dx0 - offset1[0] + unskew4_constant();
    let dy1 = dy0 - offset1[1] + unskew4_constant();
    let dz1 = dz0 - offset1[2] + unskew4_constant();
    let dw1 = dw0 - offset1[3] + unskew4_constant();

    // Compute the (x,y,z,w) distances from the third point
    let dx2 = dx0 - offset2[0] + two * unskew4_constant();
    let dy2 = dy0 - offset2[1] + two * unskew4_constant();
    let dz2 = dz0 - offset2[2] + two * unskew4_constant();
    let dw2 = dw0 - offset2[3] + two * unskew4_constant();

    // Compute the (x,y,z,w) distances from the fourth point
    let dx3 = dx0 - offset3[0] + three * unskew4_constant();
    let dy3 = dy0 - offset3[1] + three * unskew4_constant();
    let dz3 = dz0 - offset3[2] + three * unskew4_constant();
    let dw3 = dw0 - offset3[3] + three * unskew4_constant();

    // Compute the (x,y,z,w) distances from the fifth point
    let dx4 = dx0 - one + four * unskew4_constant();
    let dy4 = dy0 - one + four * unskew4_constant();
    let dz4 = dz0 - one + four * unskew4_constant();
    let dw4 = dw0 - one + four * unskew4_constant();

    let n0 = gradient(seed, [math::cast(x_cell), math::cast(y_cell), math::cast(z_cell), math::cast(w_cell)], [dx0, dy0, dz0, dw0]);
    let n1 = gradient(seed, [math::cast(x_cell + offset1[0]), math::cast(y_cell + offset1[1]), math::cast(z_cell + offset1[2]), math::cast(w_cell + offset1[3])], [dx1, dy1, dz1, dw1]);
    let n2 = gradient(seed, [math::cast(x_cell + offset2[0]), math::cast(y_cell + offset2[1]), math::cast(z_cell + offset2[2]), math::cast(w_cell + offset2[3])], [dx2, dy2, dz2, dw2]);
    let n3 = gradient(seed, [math::cast(x_cell + offset3[0]), math::cast(y_cell + offset3[1]), math::cast(z_cell + offset3[2]), math::cast(w_cell + offset3[3])], [dx3, dy3, dz3, dw3]);
    let n4 = gradient(seed, [1is + math::cast::<_, isize>(x_cell), 1is + math::cast::<_, isize>(y_cell), 1is + math::cast::<_, isize>(z_cell), 1is + math::cast::<_, isize>(w_cell)], [dx4, dy4, dz4, dw4]);

    // Sum up and scale the result to cover the range [-1,1]
    return (n0 + n1 + n2 + n3 + n4) * norm4_constant();
}
