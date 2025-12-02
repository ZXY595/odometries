use std::ops::Deref;

use nalgebra::{Dyn, Matrix3, Point3, RealField, Scalar, Vector3, stack};

use crate::{
    algorithm::lio::{downsample::Downsample, state::State},
    eskf::{Eskf, observe::UnbiasedObservation, state::common::PoseState},
    frame::{BodyPoint, Framed, frames},
    utils::{CollectTo, ToRadians},
    voxel_map::{VoxelMap, uncertain::UncertainWorldPoint},
};

use super::LIO;

pub type PointsStamped<'a, T, P> = (T, P);

pub type PointsObserved<T> = UnbiasedObservation<PoseState<T>, State<T>, Dyn>;

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

impl<T: Scalar> LidarPoint<T> for (T, T, T) {
    #[inline]
    fn to_body_point(self) -> BodyPoint<T> {
        let point = Point3::new(self.0, self.1, self.2);
        BodyPoint::new(point)
    }
}

impl<T: Scalar> LidarPoint<T> for Vector3<T> {
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

pub type PointsProcessBuffer<T> = Vec<(
    BodyPoint<T>,
    UncertainWorldPoint<T>,
    Framed<Matrix3<T>, frames::Imu>,
)>;

impl<T> LIO<T>
where
    T: RealField + ToRadians,
{
    pub fn extend_points(
        &mut self,
        timestamp: T,
        points: impl IntoIterator<Item = impl LidarPoint<T>>,
    ) {
        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        self.points_process_buffer.clear();
        points
            .into_iter()
            .map(LidarPoint::to_body_point)
            .voxel_grid_downsample(&self.downsampler.resolution, &mut self.downsampler.grid)
            .map(|body_point| {
                let (world_point, cross_matrix_imu) = UncertainWorldPoint::from_body_point(
                    body_point,
                    self.body_point_process_cov.clone(),
                    body_to_imu,
                    imu_to_world,
                    &body_to_world,
                    &self.eskf.cov,
                );
                (body_point.clone(), world_point, cross_matrix_imu)
            })
            .collect_to(&mut self.points_process_buffer);

        let is_updated = self
            .eskf
            .update(timestamp, |eskf| {
                eskf.observe_points(
                    &self.map,
                    &self.measure_noise.lidar_point,
                    self.points_process_buffer
                        .iter()
                        .map(|(_, world_point, cross_matrix_imu)| (world_point, cross_matrix_imu)),
                )
            })
            .is_some();

        let processing_points = self.points_process_buffer.iter();

        if is_updated {
            // re-compute the world points based on the updated state
            processing_points
                .map(|(body_point, _, _)| {
                    UncertainWorldPoint::from_body_point(
                        body_point,
                        self.body_point_process_cov.clone(),
                        &self.extrinsics,
                        self.eskf.pose.deref(),
                        &body_to_world,
                        &self.eskf.cov,
                    )
                    .0
                })
                .collect_to(&mut self.map);
        } else {
            processing_points
                .map(|(_, world_point, _)| world_point.clone())
                .collect_to(&mut self.map);
        };
    }
}

impl<T> Eskf<State<T>>
where
    T: RealField + ToRadians,
{
    fn observe_points<'a>(
        &self,
        map: &VoxelMap<T>,
        measure_noise: &T,
        points: impl IntoIterator<
            Item = (
                &'a UncertainWorldPoint<T>,
                &'a Framed<Matrix3<T>, frames::Imu>,
            ),
        >,
    ) -> Option<PointsObserved<T>> {
        // TODO: could this be optimized by using `rayon`?
        let observation = points
            .into_iter()
            .filter_map(|(world_point, cross_matrix_imu)| {
                let residual = map.get_residual_or_nearest(world_point)?;

                let plane_normal = residual.plane_normal();
                let cross_matrix_rotation_t_normal =
                    cross_matrix_imu.deref() * self.pose.rotation.transpose() * plane_normal;

                #[expect(clippy::toplevel_ref_arg)]
                let model = stack![cross_matrix_rotation_t_normal; plane_normal];

                let measurement = -residual.distance_to_plane;

                let noise = measure_noise.clone() * residual.sigma;

                Some((measurement, model, noise))
            })
            .collect::<PointsObserved<T>>();

        if observation.get_dim().0 == 0 {
            return None;
        }
        Some(observation)
    }
}

impl<'a, T, P> Extend<PointsStamped<'a, T, P>> for LIO<T>
where
    T: RealField + ToRadians,
    P: IntoIterator<Item: LidarPoint<T>>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (T, P)>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter().for_each(|(timestamp, points)| {
            self.extend_points(timestamp, points);
        });
    }
}
