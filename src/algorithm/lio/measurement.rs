mod imu;
mod point_chunk;

pub use imu::{ImuMeasured, ImuMeasuredStamped};
use nalgebra::RealField;
pub use point_chunk::{LidarPoint, PointChunkStamped};

use crate::utils::ToRadians;

use super::LIO;

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn extend_measurements<'a, P: LidarPoint<T> + 'a>(
        &mut self,
        points: impl IntoIterator<Item = PointChunkStamped<'a, T, P>>,
        imus: impl IntoIterator<Item = ImuMeasuredStamped<'a, T>>,
    ) {
        let points = points.into_iter();
        let mut imus = imus.into_iter();

        // TODO: could this be optimized by using `rayon`?
        points.for_each(|(points_time, point_chunk)| {
            let imus_before_points = imus
                .by_ref()
                .take_while(|(imu_time, _)| imu_time < &points_time);
            self.extend(imus_before_points);
            self.extend_point_chunk(points_time, point_chunk);
        });
    }
}
