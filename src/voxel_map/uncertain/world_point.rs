use std::ops::Deref;

use nalgebra::{DefaultAllocator, RealField, Scalar, U3, allocator::Allocator};

use crate::{
    eskf::{
        Covariance,
        state::{
            KFState, SubStateOf,
            common::{PositionState, RotationState},
        },
    },
    frame::{IsometryFramed, frames},
};

use super::{UncertainBodyPoint, UncertainWorldPoint};

use num_traits::{One, Zero};

impl<T> UncertainWorldPoint<T>
where
    T: Scalar + RealField + One + Zero + Default,
{
    pub fn from_uncertain_body_point<S>(
        body_point: UncertainBodyPoint<T>,
        body_to_imu: &IsometryFramed<T, fn(frames::Body) -> frames::Imu>,
        imu_to_world: &IsometryFramed<T, fn(frames::Imu) -> frames::World>,
        eskf_cov: &Covariance<S>,
    ) -> Self
    where
        S: KFState<Element = T>,
        RotationState<T>: SubStateOf<S, Dim = U3>,
        PositionState<T>: SubStateOf<S, Dim = U3>,
        DefaultAllocator: Allocator<S::Dim, S::Dim> + Allocator<U3, U3>,
    {
        let imu_point = body_point.deref() * body_to_imu;
        let world_point = &imu_point * imu_to_world;
        let body_to_world = body_to_imu * imu_to_world;

        let cross_matrix_world = &imu_to_world.rotation * imu_point.coords.cross_matrix();
        let rot_cov = eskf_cov.sub_covariance::<RotationState<T>>();
        let pos_cov = eskf_cov.sub_covariance::<PositionState<T>>();

        let mut cov = pos_cov.clone_owned();
        cov.quadform_tr(
            T::one(),
            body_to_world.rotation.matrix(),
            &body_point.cov,
            T::one(),
        );
        cov.quadform_tr(T::one(), &cross_matrix_world, &rot_cov, T::one());

        Self::new_with_cov(world_point, cov)
    }
}
