use std::ops::Deref;

use nalgebra::{Point3, RealField, Scalar, stack};

use crate::{
    algorithm::lio::estimate::PointsObserved,
    frame::{BodyPoint, Framed},
    utils::ToRadians,
    voxel_map::uncertain::{UncertainBodyPoint, UncertainWorldPoint},
};

use super::LIO;

pub type PointChunkStamped<'a, T, P> = (T, &'a [P]);

pub trait LidarPoint<T: Scalar>: Clone {
    fn to_body_point(self) -> BodyPoint<T>;
}

impl<T: Scalar> LidarPoint<T> for [T; 3] {
    #[inline]
    fn to_body_point(self) -> BodyPoint<T> {
        let point = Point3::from(self);
        BodyPoint::new(point)
    }
}

impl<T: Scalar> LidarPoint<T> for BodyPoint<T> {
    #[inline(always)]
    fn to_body_point(self) -> BodyPoint<T> {
        self
    }
}

impl<T: Scalar, P: LidarPoint<T>> LidarPoint<T> for (P, T) {
    #[inline(always)]
    fn to_body_point(self) -> BodyPoint<T> {
        self.0.to_body_point()
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    /// The reason why `&'a [impl LidarPoint<T>]` instead of `impl IntoIterator<Item = LidarPoint<T>>`
    /// is that we need to iterate the point chunk twice, once for observation and once for updating the map.
    pub fn extend_point_chunk(&mut self, timestamp: T, points: &[impl LidarPoint<T>]) {
        self.eskf_update(timestamp, |ilo| {
            ilo.observe_points(points.iter().cloned().map(LidarPoint::to_body_point))
        });

        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        let new_world_points = points
            .iter()
            .cloned()
            .map(LidarPoint::to_body_point)
            .map(|point| {
                UncertainWorldPoint::from_body_point(
                    point,
                    self.body_point_process_cov.clone(),
                    body_to_imu,
                    imu_to_world,
                    &body_to_world,
                    &self.eskf.cov,
                )
            });
        self.map.extend(new_world_points);
    }

    fn observe_points(
        &self,
        points: impl Iterator<Item = BodyPoint<T>>,
    ) -> Option<PointsObserved<T>> {
        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        // TODO: could this be optimized by using `rayon`?
        let observation = points
            .filter_map(|point| {
                let body_point = UncertainBodyPoint::<T>::from_body_point(
                    point,
                    self.body_point_process_cov.clone(),
                );

                let imu_point = body_point.deref() * body_to_imu;
                let cross_matrix_imu = imu_point.coords.cross_matrix();

                let world_point = UncertainWorldPoint::from_uncertain_body_point(
                    body_point,
                    imu_to_world,
                    &body_to_world,
                    Framed::new(&cross_matrix_imu),
                    &self.eskf.cov,
                );
                let residual = self.map.get_residual(world_point).or_else(|| {
                    // TODO: search for one nearest voxel in the map
                    todo!()
                })?;
                let plane_normal = residual.plane_normal;
                let crossmatrix_rotation_t_normal =
                    cross_matrix_imu * self.eskf.pose.rotation.transpose() * plane_normal;

                #[expect(clippy::toplevel_ref_arg)]
                let model = stack![crossmatrix_rotation_t_normal; plane_normal];

                // TODO: do we really need to [`neg`](std::ops::Neg) this measurement?
                let measurement = -residual.distance_to_plane;

                let noise = self.measure_noise.lidar_point.clone() * residual.sigma;

                Some((measurement, model, noise))
            })
            .collect::<PointsObserved<T>>();

        if observation.get_dim().0 == 0 {
            return None;
        }
        Some(observation)
    }
}

impl<'a, T, P> Extend<PointChunkStamped<'a, T, P>> for LIO<T>
where
    T: RealField + ToRadians + Default,
    P: LidarPoint<T>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (T, &'a [P])>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter().for_each(|(timestamp, points)| {
            self.extend_point_chunk(timestamp, points);
        });
    }
}
