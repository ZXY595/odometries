mod imu;
mod points;

pub use imu::{ImuMeasured, ImuMeasuredStamped};
use nalgebra::{RealField, Scalar};
pub use points::{LidarPoint, PointsStamped};
use simba::scalar::SupersetOf;

use crate::{eskf::state::common::AccState, utils::ToRadians};

use super::LIO;

pub struct MeasureNoiseConfig<T: Scalar> {
    pub imu_acc: AccState<T>,
    pub lidar_point: T,
}

impl<T: Scalar + SupersetOf<f64>> Default for MeasureNoiseConfig<T> {
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
            lidar_point: nalgebra::convert(0.01),
        }
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn extend_measurements<'a, P>(
        &mut self,
        points: impl IntoIterator<Item = PointsStamped<'a, T, P>>,
        imus: impl IntoIterator<Item = ImuMeasuredStamped<T>>,
    ) where
        P: IntoIterator<Item: LidarPoint<T>, IntoIter: Clone> + 'a,
    {
        let points = points.into_iter();
        let mut imus = imus.into_iter();

        // TODO: could this be optimized by using `rayon`?
        points.for_each(|(points_time, point_chunk)| {
            let imus_before_points = imus
                .by_ref()
                .take_while(|imu_measured| imu_measured.timestamp < points_time);
            self.extend(imus_before_points);
            self.extend_points(points_time, point_chunk);
        });
    }
}
