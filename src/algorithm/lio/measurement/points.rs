use std::ops::Deref;

use nalgebra::{Dyn, Point3, RealField, Scalar, Vector3, stack};

use crate::{
    algorithm::lio::{downsample::Downsample, state::State},
    eskf::{Eskf, observe::UnbiasedObservation, state::common::PoseState, uncertain::Uncertained},
    frame::{BodyPoint, CrossMatrixFramed, Framed, IsometryFramed, frames},
    utils::{CollectTo, ToRadians},
    voxel_map::{
        VoxelMap,
        uncertain::{UncertainBodyPoint, UncertainWorldPoint},
    },
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
    UncertainBodyPoint<T>,
    UncertainWorldPoint<T>,
    CrossMatrixFramed<T, frames::Imu>,
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

        debug_assert_eq!(self.points_process_buffer.len(), 0);

        // TODO: parallel optimizable
        points
            .into_iter()
            .map(LidarPoint::to_body_point)
            .voxel_grid_downsample(&self.downsampler.resolution, &mut self.downsampler.grid)
            .map(|body_point| {
                UncertainBodyPoint::from_body_point(body_point, self.body_point_process_cov.clone())
            })
            .map(|body_point| {
                let imu_point = body_point.deref() * body_to_imu;
                let cross_matrix_imu = Framed::new(imu_point.coords.cross_matrix());

                let world_point = UncertainWorldPoint::from_uncertain_body_point(
                    body_point.clone(),
                    imu_to_world,
                    &body_to_world,
                    cross_matrix_imu.as_ref(),
                    &self.eskf.cov,
                );
                (body_point, world_point, cross_matrix_imu)
            })
            .collect_to(&mut self.points_process_buffer);

        let is_updated = self
            .eskf
            .update(timestamp, |eskf| {
                eskf.observe_points(
                    &self.map,
                    &self.measure_noise.lidar_point,
                    &body_to_world,
                    self.points_process_buffer.iter(),
                )
            })
            .is_some();

        let processing_points = self.points_process_buffer.drain(..);

        if is_updated {
            // TODO: parallel optimizable
            // re-compute the world points based on the updated state
            processing_points
                .map(|(body_point, _, cross_matrix_imu)| {
                    UncertainWorldPoint::from_uncertain_body_point(
                        body_point,
                        self.eskf.pose.deref(),
                        &body_to_world,
                        cross_matrix_imu.as_ref(),
                        &self.eskf.cov,
                    )
                })
                .collect_to(&mut self.map);
        } else {
            // TODO: parallel optimizable
            processing_points
                .map(|(_, world_point, _)| world_point)
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
        body_to_world: &IsometryFramed<T, fn(frames::Body) -> frames::World>,
        points: impl IntoIterator<
            Item = &'a (
                UncertainBodyPoint<T>,
                UncertainWorldPoint<T>,
                CrossMatrixFramed<T, frames::Imu>,
            ),
        >,
    ) -> Option<PointsObserved<T>> {
        // TODO: could this be optimized by using `rayon`?
        let observation = points
            .into_iter()
            .filter_map(|(body_point, world_point, cross_matrix_imu)| {
                let Uncertained {
                    state: residual,
                    cov: residual_cov,
                } = map
                    .get_or_nearest_residual(world_point)?
                    .to_uncertained(body_point, body_to_world);

                let plane_normal = residual.plane_normal();
                let cross_matrix_rotation_t_normal =
                    cross_matrix_imu.deref() * self.pose.rotation.transpose() * plane_normal;

                #[expect(clippy::toplevel_ref_arg)]
                let model = stack![cross_matrix_rotation_t_normal; plane_normal];

                let measurement = -residual.distance_to_plane;

                let noise = measure_noise.clone() * residual_cov.to_scalar();

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
