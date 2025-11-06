use std::cmp::Ordering;
use std::ops::Deref;

use nalgebra::{Point3, RealField, Scalar, Vector3};

use crate::utils::ToVoxelIndex;
use crate::voxel_map::uncertain::UncertainWorldPoint;

use super::VoxelMap;

pub struct Residual<'a, T: Scalar> {
    pub plane_normal: &'a Vector3<T>,
    pub distance_to_plane: T,
    pub sigma: T,
    distance_to_plane_squared: T,
    sigma_sqrt: T,
}

impl<T> VoxelMap<T>
where
    T: RealField + Default,
    Point3<T>: ToVoxelIndex<VoxelSize = T>,
{
    pub fn get_residual(&self, point: UncertainWorldPoint<T>) -> Option<Residual<'_, T>> {
        let index = point.to_voxel_index(self.config.voxel_size.clone());
        let root = self.roots.get(&index)?;
        let radius_factor: T = nalgebra::convert(3.0);
        // TODO: could be optimized by using `rayon`
        root.iter_planes()
            .map(|plane| {
                let normal = &plane.normal;
                let distance_to_plane =
                    normal.dot(&point.coords) - normal.dot(&plane.center.coords);
                let distance_to_plane_squared =
                    distance_to_plane.clone() * distance_to_plane.clone();
                (plane, distance_to_plane, distance_to_plane_squared)
            })
            .filter(|(plane, _, distance_to_plane_squared)| {
                let distance_to_center_squared = (point.deref() - &plane.center).norm_squared();
                let range_distance =
                    (distance_to_center_squared - distance_to_plane_squared.clone()).sqrt();
                range_distance <= radius_factor.clone() * plane.radius.clone()
            })
            .map(|(plane, distance_to_plane, distance_to_plane_squared)| {
                let sigma = plane.sigma_to(&point);
                Residual {
                    plane_normal: &plane.normal,
                    distance_to_plane,
                    distance_to_plane_squared,
                    sigma_sqrt: sigma.clone().sqrt(),
                    sigma,
                }
            })
            .filter(|residual| {
                residual.distance_to_plane
                    < self.config.sigma_ratio.clone() * residual.sigma_sqrt.clone()
            })
            .min_by(|a, b| {
                a.probability()
                    .partial_cmp(&b.probability())
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl<'a, T: RealField> Residual<'a, T> {
    fn probability(&self) -> T {
        T::one()
            / (self.sigma_sqrt.clone()
                * (nalgebra::convert::<_, T>(-0.5) * self.distance_to_plane_squared.clone()
                    / self.sigma.clone()))
    }
}
