use std::cmp::Ordering;

use nalgebra::{Point3, RealField, Scalar, Vector3};

use crate::utils::ToVoxelIndex;
use crate::voxel_map::uncertain::UncertainWorldPoint;

use super::VoxelMap;

pub struct Residual<T: Scalar> {
    pub plane_normal: Vector3<T>,
    pub dis_to_plane: T,
    pub sigma: T,
}

impl<'alloc, T> VoxelMap<'alloc, T>
where
    T: RealField + Default,
    Point3<T>: ToVoxelIndex,
{
    pub fn get_residual(&'alloc self, point: UncertainWorldPoint<T>) -> Option<Residual<T>> {
        let index = point
            .map(|x| x / self.config.voxel_size.clone())
            .to_voxel_index();
        let root = &self.roots[&index];
        root.iter_planes()
            .map(|plane| {
                let normal = &plane.normal;
                let dis_to_plane = normal.dot(&point.coords) - normal.dot(&plane.center.coords);
                let sigma = plane.sigma_to(&point);
                Residual {
                    plane_normal: normal.clone(),
                    dis_to_plane,
                    sigma,
                }
            })
            .min_by(|a, b| a.sigma.partial_cmp(&b.sigma).unwrap_or(Ordering::Equal))
    }
}
