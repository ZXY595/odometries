use nalgebra::{Matrix1, Matrix2, Matrix3, Matrix3x2, RealField, Vector3};

use crate::{
    frame::BodyPoint,
    utils::{Substitutive, ToRadians},
    voxel_map::uncertain::UncertainBodyPoint,
};

#[derive(Debug, Clone)]
pub struct ProcessCovConfig<T> {
    pub distance: T,
    pub direction: T,
}

impl<T> UncertainBodyPoint<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn from_body_point(point: BodyPoint<T>, config: ProcessCovConfig<T>) -> Self {
        let point_coords = &point.coords;
        let range = point_coords.norm();
        let direction = point_coords.normalize();

        let base1 = Vector3::new(T::one(), T::one(), {
            -(direction.x.clone() + direction.y.clone())
                / direction.z.clone().non_zero_or_substitute()
        })
        .normalize();

        let base2 = base1.cross(&direction).normalize();

        let point_base_coords =
            direction.cross_matrix() * range * Matrix3x2::from_columns(&[base1, base2]);

        let distance_cov = Matrix1::new(config.distance.powi(2));

        let direction_cov =
            Matrix2::from_diagonal_element(config.direction.to_radians().sin().powi(2));

        // TODO: can we avoid init with zeros?
        let mut cov = Matrix3::zeros();
        cov.quadform_tr(T::one(), &direction, &distance_cov, T::zero());
        cov.quadform_tr(T::one(), &point_base_coords, &direction_cov, T::one());

        Self::new_with_cov(point, cov)
    }
}
