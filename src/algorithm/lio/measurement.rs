mod imu;
mod points;

use std::ops::{Deref, DerefMut};

pub use imu::{ImuInit, ImuMeasured, StampedImu};
use nalgebra::{RealField, Scalar};
pub use points::{LidarPoint, PointsProcessBuffer, StampedPoints};
use simba::scalar::SupersetOf;

use crate::{eskf::state::common::AccState, utils::ToRadians};

use super::LIO;

pub struct MeasureNoiseConfig<T: Scalar> {
    pub imu_acc: AccState<T>,
    pub lidar_point: T,
}

pub struct StampedMeasurement<T, M> {
    pub timestamp: T,
    pub measured: M,
}

impl<T> Default for MeasureNoiseConfig<T>
where
    T: Scalar + SupersetOf<f64>,
{
    fn default() -> Self {
        Self {
            imu_acc: AccState::new(
                nalgebra::convert(0.1),
                nalgebra::convert(0.1),
                nalgebra::convert(0.1),
                nalgebra::convert(0.01),
                nalgebra::convert(0.01),
                nalgebra::convert(0.01),
            ),
            lidar_point: nalgebra::convert(10.0),
        }
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians,
{
    pub fn update_points_with_imus<P>(
        &mut self,
        points: StampedPoints<T, P>,
        imus: impl IntoIterator<Item = StampedImu<T>>,
    ) where
        P: IntoIterator<Item: LidarPoint<T>>,
    {
        self.extend(imus);
        self.update_stamped_points(points);
    }

    pub fn update_point_clouds_with_imus<P>(
        &mut self,
        point_clouds: impl IntoIterator<Item = StampedPoints<T, P>>,
        imus: impl IntoIterator<Item = StampedImu<T>>,
    ) where
        P: IntoIterator<Item: LidarPoint<T>>,
    {
        let mut imus = imus.into_iter();

        // TODO: could this be optimized by using `rayon`?
        point_clouds.into_iter().for_each(|points| {
            let imus_before_points = imus
                .by_ref()
                .take_while(|imu_measured| imu_measured.timestamp < points.timestamp);
            self.extend(imus_before_points);
            self.update_stamped_points(points);
        });
    }
}

impl<T: Scalar, M> StampedMeasurement<T, M> {
    pub const fn new(timestamp: T, measured: M) -> Self {
        Self {
            timestamp,
            measured,
        }
    }

    pub fn from_tuple((timestamp, measured): (T, M)) -> Self {
        Self {
            timestamp,
            measured,
        }
    }
}

impl<T: Scalar, M> Deref for StampedMeasurement<T, M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.measured
    }
}

impl<T: Scalar> DerefMut for StampedImu<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.measured
    }
}

impl<T, P> Extend<(StampedImu<T>, StampedPoints<T, P>)> for LIO<T>
where
    T: RealField + ToRadians,
    P: IntoIterator<Item: LidarPoint<T>>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (StampedImu<T>, StampedPoints<T, P>)>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter()
            .for_each(|(imu, points)| self.update_points_with_imus(points, [imu]));
    }
}
