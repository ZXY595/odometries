use std::ops::Deref;

use nalgebra::{Matrix3, RealField};

use crate::{
    frame::{IsometryFramed, frames},
    voxel_map::uncertain::UncertainWorldPoint,
};

use super::{Residual, UncertainBodyPoint, UncertainResidual};

impl<'a, T: RealField> Residual<'a, T> {
    pub fn to_uncertained(
        self,
        body_point: &UncertainBodyPoint<T>,
        body_to_world: &IsometryFramed<T, fn(frames::Body) -> frames::World>,
    ) -> UncertainResidual<'a, T> {
        UncertainResidual::from_body_point_and_plane(self, body_point, body_to_world)
    }
}

impl<'a, T: RealField> UncertainResidual<'a, T> {
    pub fn from_body_point_and_plane(
        residual: Residual<'a, T>,
        body_point: &UncertainBodyPoint<T>,
        body_to_world: &IsometryFramed<T, fn(frames::Body) -> frames::World>,
    ) -> Self {
        let mut cov = Matrix3::zeros();
        cov.quadform_tr(
            T::one(),
            body_to_world.rotation.matrix(),
            &body_point.cov,
            T::zero(),
        );
        let world_point = body_point.deref() * body_to_world;

        let cov = residual
            .plane
            .sigma_to(&UncertainWorldPoint::new_with_cov(world_point, cov));

        Self::new_with_cov(residual, cov)
    }
}
