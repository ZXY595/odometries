use std::cmp::Ordering;
use std::ops::Deref;

use nalgebra::{ComplexField, RealField, Scalar};

use super::{
    VoxelMap,
    index::{ToVoxelIndex, VoxelIndex},
    oct_tree::OctTreeRoot,
    uncertain::{UncertainWorldPoint, plane::Plane},
};

pub struct Residual<'a, T: Scalar> {
    pub plane: &'a Plane<T>,
    pub distance_to_plane: T,
    pub sigma: T,
    distance_to_plane_squared: T,
    sigma_sqrt: T,
}

pub struct NoValidResidual<'a, T: ComplexField> {
    /// the root of the oct tree where the residual of the given point was not found
    voxel_root: &'a OctTreeRoot<T>,
    voxel_index: VoxelIndex<T>,
}

impl<T> VoxelMap<T>
where
    T: RealField,
{
    pub fn get_residual(
        &self,
        point: &UncertainWorldPoint<T>,
    ) -> Result<Residual<'_, T>, Option<NoValidResidual<'_, T>>> {
        let voxel_index = point.as_voxel_index(self.config.voxel_size.clone());
        self.get_residual_by_coord(voxel_index, point)
    }

    pub fn get_residual_or_nearest(
        &self,
        point: &UncertainWorldPoint<T>,
    ) -> Option<Residual<'_, T>> {
        self.get_residual(point)
            .or_else(|err| {
                let NoValidResidual {
                    voxel_root,
                    voxel_index,
                } = err.ok_or(())?;

                let nearest_coord = voxel_root.nearest_voxel(point, voxel_index);

                self.get_residual_by_coord(nearest_coord, point)
                    .map_err(drop)
            })
            .ok()
    }

    fn get_residual_by_coord(
        &self,
        voxel_index: VoxelIndex<T>,
        point: &UncertainWorldPoint<T>,
    ) -> Result<Residual<'_, T>, Option<NoValidResidual<'_, T>>> {
        let voxel_root = self.roots.get(&voxel_index).ok_or(None)?;
        let radius_factor: T = nalgebra::convert(3.0);

        // TODO: could be optimized by using `rayon`?
        voxel_root
            .iter_planes()
            .map(|plane| {
                let normal = &plane.normal;
                let distance_to_plane =
                    normal.dot(&point.coords) - normal.dot(&plane.center.coords);
                (
                    plane,
                    distance_to_plane.clone(),
                    distance_to_plane.clone() * distance_to_plane,
                )
            })
            .filter(|(plane, _, distance_to_plane_squared)| {
                let distance_to_center_squared = (point.deref() - &plane.center).norm_squared();
                let range_distance =
                    (distance_to_center_squared - distance_to_plane_squared.clone()).sqrt();
                range_distance <= radius_factor.clone() * plane.radius.clone()
            })
            .map(|(plane, distance_to_plane, distance_to_plane_squared)| {
                let sigma = plane.sigma_to(point);
                Residual {
                    plane: plane.deref(),
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
            .max_by(|a, b| {
                a.probability()
                    .partial_cmp(&b.probability())
                    // TODO: this might be bad
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or(Some(NoValidResidual {
                voxel_root,
                voxel_index,
            }))
    }
}

impl<'a, T: RealField> Residual<'a, T> {
    pub fn plane_normal(&self) -> &nalgebra::Vector3<T> {
        &self.plane.normal
    }

    fn probability(&self) -> T {
        T::one()
            / (self.sigma_sqrt.clone()
                * (self.distance_to_plane_squared.clone() * nalgebra::convert(-0.5)
                    / self.sigma.clone()))
    }
}
