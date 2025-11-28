use nalgebra::{DefaultAllocator, Matrix3, RealField, Scalar, U3, allocator::Allocator};

use crate::{
    eskf::{
        Covariance,
        state::{
            KFState, SubStateOf,
            common::{PositionState, RotationState},
        },
    },
    frame::{BodyPoint, Framed, IsometryFramed, frames},
    utils::ToRadians,
    voxel_map::uncertain::{UncertainBodyPoint, body_point},
};

use super::UncertainWorldPoint;

pub struct UncertainPoint<T: Scalar> {
    pub world_point: UncertainWorldPoint<T>,
    pub cross_matrix_imu: Matrix3<T>,
}

impl<T> UncertainWorldPoint<T>
where
    T: RealField,
{
    pub fn from_uncertain_body_point<S>(
        body_point: UncertainBodyPoint<T>,
        imu_to_world: &IsometryFramed<T, fn(frames::Imu) -> frames::World>,
        body_to_world: &IsometryFramed<T, fn(frames::Body) -> frames::World>,
        cross_matrix_imu: Framed<&Matrix3<T>, frames::Imu>,
        eskf_cov: &Covariance<S>,
    ) -> Self
    where
        S: KFState<Element = T>,
        RotationState<T>: SubStateOf<S, Dim = U3>,
        PositionState<T>: SubStateOf<S, Dim = U3>,
        DefaultAllocator: Allocator<S::Dim, S::Dim>,
    {
        let world_point = body_point.as_deref_ref() * body_to_world;
        let rot_cov = eskf_cov.sub_covariance::<RotationState<T>>();
        let pos_cov = eskf_cov.sub_covariance::<PositionState<T>>();

        let mut cov = pos_cov.into_owned();
        cov.quadform_tr(
            T::one(),
            body_to_world.rotation.matrix(),
            &body_point.cov,
            T::one(),
        );
        cov.quadform_tr(
            T::one(),
            &(&imu_to_world.rotation * *cross_matrix_imu),
            &rot_cov,
            T::one(),
        );

        Self::new_with_cov(world_point, cov)
    }

    pub fn from_body_point<S>(
        point: &BodyPoint<T>,
        config: body_point::ProcessCov<T>,
        body_to_imu: &IsometryFramed<T, fn(frames::Body) -> frames::Imu>,
        imu_to_world: &IsometryFramed<T, fn(frames::Imu) -> frames::World>,
        body_to_world: &IsometryFramed<T, fn(frames::Body) -> frames::World>,
        eskf_cov: &Covariance<S>,
    ) -> (Self, Framed<Matrix3<T>, frames::Imu>)
    where
        T: ToRadians,
        S: KFState<Element = T>,
        RotationState<T>: SubStateOf<S, Dim = U3>,
        PositionState<T>: SubStateOf<S, Dim = U3>,
        DefaultAllocator: Allocator<S::Dim, S::Dim>,
    {
        let body_point = UncertainBodyPoint::<T>::from_body_point_ref(point, config);

        let imu_point = body_point.as_deref_ref() * body_to_imu;
        let cross_matrix_imu = imu_point.coords.cross_matrix();

        (
            Self::from_uncertain_body_point(
                body_point,
                imu_to_world,
                body_to_world,
                Framed::new(&cross_matrix_imu),
                eskf_cov,
            ),
            Framed::new(cross_matrix_imu),
        )
    }
}
