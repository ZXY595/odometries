use std::ops::Deref;

use nalgebra::{Dyn, Point3, RealField, Scalar, Vector3, stack};

use crate::{
    algorithm::lio::state::State,
    eskf::{StateFilter, observe::UnbiasedObservation, state::common::PoseState},
    frame::{BodyPoint, Framed},
    utils::ToRadians,
    voxel_map::uncertain::{UncertainBodyPoint, UncertainWorldPoint},
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

impl<T: Scalar, P: LidarPoint<T>> LidarPoint<T> for (T, P) {
    #[inline(always)]
    fn to_body_point(self) -> BodyPoint<T> {
        self.1.to_body_point()
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn extend_points(
        &mut self,
        timestamp: T,
        points: impl IntoIterator<Item = impl LidarPoint<T>, IntoIter: Clone>,
    ) {
        let points = points.into_iter();
        self.update(timestamp, |ilo| {
            ilo.observe_points(points.clone().map(LidarPoint::to_body_point))
        });

        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        let new_world_points = points.clone().map(LidarPoint::to_body_point).map(|point| {
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
                let residual = self.map.get_residual(world_point)
                // .or_else(|| {
                //     // TODO: search for one nearest voxel in the map
                //     todo!()
                // })
                ?;
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

impl<'a, T, P> Extend<PointsStamped<'a, T, P>> for LIO<T>
where
    T: RealField + ToRadians + Default,
    P: IntoIterator<Item: LidarPoint<T>, IntoIter: Clone>,
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
