mod imu;
mod points;

pub use imu::{ImuInit, ImuMeasured, ImuMeasuredStamped};
use nalgebra::{RealField, Scalar};
pub use points::{LidarPoint, PointsStamped};
use simba::scalar::SupersetOf;

use crate::{eskf::state::common::AccState, utils::ToRadians};

use super::LIO;

pub struct MeasureNoiseConfig<T: Scalar> {
    pub imu_acc: AccState<T>,
    pub lidar_point: T,
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
    pub fn extend_point_cloud_with_imu<'a, P>(
        &mut self,
        imus: impl IntoIterator<Item = ImuMeasuredStamped<T>>,
        (end_timestamp, points): PointsStamped<'a, T, P>,
    ) where
        P: IntoIterator<Item: LidarPoint<T>, IntoIter: Clone> + 'a,
    {
        self.extend(imus);
        self.extend_points(end_timestamp, points);
    }

    pub fn extend_measurements<'a, P>(
        &mut self,
        point_clouds: impl IntoIterator<Item = PointsStamped<'a, T, P>>,
        imus: impl IntoIterator<Item = ImuMeasuredStamped<T>>,
    ) where
        P: IntoIterator<Item: LidarPoint<T>, IntoIter: Clone> + 'a,
    {
        let point_clouds = point_clouds.into_iter();
        let mut imus = imus.into_iter();

        // TODO: could this be optimized by using `rayon`?
        point_clouds.for_each(|(points_time, points)| {
            let imus_before_points = imus
                .by_ref()
                .take_while(|imu_measured| imu_measured.timestamp < points_time);
            self.extend(imus_before_points);
            self.extend_points(points_time, points);
        });
    }
}
