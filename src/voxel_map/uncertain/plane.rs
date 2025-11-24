use std::ops::Deref;

use crate::{
    eskf::state::KFState,
    frame::WorldPoint,
    utils::VectorSquareSum,
    voxel_map::uncertain::{UncertainPlane, UncertainWorldPoint},
};

use nalgebra::{
    Matrix1, Matrix3, Matrix6, Point3, RealField, RowVector3, Scalar, SymmetricEigen, U6, Vector3,
    stack,
};
use num_traits::Zero;
use simba::scalar::SupersetOf;

pub struct Plane<T: Scalar> {
    pub normal: Vector3<T>,
    pub center: WorldPoint<T>,
    pub radius: T,
}

pub struct PlaneConfig<T> {
    pub max_layer: u8,
    /// minimum number of points to init a tree
    pub plane_init_threshold: usize,
    /// minimum number of points to update a plane
    pub update_threshold: usize,
    /// maximum eigen value of a plane to be considered as a valid plane
    pub plane_eigen_threshold: T,
    /// maximum number of points for a tree
    pub max_points: usize,
}

impl<T> Default for PlaneConfig<T>
where
    T: SupersetOf<f64>,
{
    fn default() -> Self {
        Self {
            max_layer: 4,
            plane_init_threshold: 5,
            update_threshold: 5,
            plane_eigen_threshold: nalgebra::convert(0.01),
            max_points: 50,
        }
    }
}

impl<T: Scalar> KFState for Plane<T> {
    type Element = T;
    type Dim = U6;
}

impl<T> Default for Plane<T>
where
    T: Scalar + Default + Zero,
{
    fn default() -> Self {
        Self {
            normal: Default::default(),
            center: Default::default(),
            radius: T::zero(),
        }
    }
}

pub enum PlaneInitError {
    TooFewPoints,
    EigenValueTooBig,
}

impl<T> UncertainPlane<T>
where
    T: RealField,
{
    pub fn from_uncertain_world_points(
        points: &[UncertainWorldPoint<T>],
        config: &PlaneConfig<T>,
    ) -> Result<Self, PlaneInitError> {
        if points.len() < config.plane_init_threshold {
            return Err(PlaneInitError::TooFewPoints);
        }

        let sum = points
            .iter()
            .map(|uncertain_point| &uncertain_point.coords)
            .sum::<VectorSquareSum<T>>();

        let points_count = sum.count();
        let (center, covariance) = sum.mean();
        let center = WorldPoint::new(Point3::from(center));

        let SymmetricEigen {
            eigenvectors,
            eigenvalues,
        } = covariance.symmetric_eigen();

        let (min_eigen_index, min_eigen_value) = eigenvalues.argmin();

        if min_eigen_value >= config.plane_eigen_threshold {
            return Err(PlaneInitError::EigenValueTooBig);
        }

        let min_eigenvector = eigenvectors.column(min_eigen_index);

        let covariance = points
            .iter()
            .map(|uncertain_point| {
                let points_count: T = nalgebra::convert(points_count as f64);

                let rows: [RowVector3<T>; 3] = std::array::from_fn(|i| {
                    if i == min_eigen_index {
                        return RowVector3::zeros();
                    }
                    let eigenvector = eigenvectors.column(i);

                    // TODO: do we need BLAS here?
                    (uncertain_point.deref() - &center).transpose()
                        / (points_count.clone()
                            * (min_eigen_value.clone() - eigenvalues[i].clone()))
                        * (&eigenvector * min_eigenvector.transpose()
                            + &min_eigenvector * eigenvector.transpose())
                });
                let normal_error = &eigenvectors * Matrix3::from_rows(&rows);
                let position_error = Matrix3::from_diagonal_element(points_count.recip());

                #[expect(clippy::toplevel_ref_arg)]
                let error_matrix = stack![normal_error; position_error];

                // TODO: can we avoid init with zeros?
                let mut cov = Matrix6::zeros();
                cov.quadform_tr(T::one(), &error_matrix, &uncertain_point.cov, T::zero());
                cov
            })
            .sum::<Matrix6<T>>();

        let normal = eigenvectors.column(min_eigen_index);
        let radius = eigenvalues.max().sqrt();

        // let distance_to_origin = normal.dot(&center);

        Ok(Self::new_with_cov(
            Plane {
                normal: normal.into(),
                center,
                radius,
            },
            covariance,
        ))
    }

    pub fn sigma_to(&self, world_point: &UncertainWorldPoint<T>) -> T {
        let distance_error = world_point.deref() - &self.center;
        let normal_error = -&self.normal;

        #[expect(clippy::toplevel_ref_arg)]
        let error_matrix = stack![distance_error; normal_error];

        let mut sigma = Matrix1::zeros();
        sigma.quadform(T::one(), self.cov.deref(), &error_matrix, T::zero());
        sigma.quadform(T::one(), world_point.cov.deref(), &self.normal, T::one());
        sigma.to_scalar()
    }
}
