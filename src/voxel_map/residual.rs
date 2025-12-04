use std::cmp::Ordering;
use std::ops::Deref;

use nalgebra::{ComplexField, RealField, Scalar, U1};

use crate::{eskf::state::KFState, voxel_map::MapIndex};

use super::{
    VoxelMap,
    index::ToVoxelIndex,
    oct_tree::OctTreeRoot,
    uncertain::{UncertainPlane, UncertainWorldPoint},
};

pub struct Residual<'a, T: Scalar> {
    pub plane: &'a UncertainPlane<T>,
    pub distance_to_plane: T,
    sigma: T,
    distance_to_plane_squared: T,
    sigma_sqrt: T,
}

pub struct NoValidResidual<'a, T: ComplexField> {
    /// the root of the oct tree where the residual of the given point was not found
    voxel_root: &'a OctTreeRoot<T>,
    voxel_index: MapIndex<T>,
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

    pub fn get_or_nearest_residual(
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
        voxel_index: MapIndex<T>,
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
                (plane, distance_to_plane.clone(), distance_to_plane.powi(2))
            })
            .filter(|(plane, _, distance_to_plane_squared)| {
                let distance_to_center = point.deref() - &plane.center;
                let range_distance =
                    (distance_to_center.norm_squared() - distance_to_plane_squared.clone()).sqrt();
                range_distance <= radius_factor.clone() * plane.radius.clone()
            })
            .map(|(plane, distance_to_plane, distance_to_plane_squared)| {
                let sigma = plane.sigma_to(point).to_scalar();
                Residual {
                    plane,
                    distance_to_plane,
                    distance_to_plane_squared,
                    sigma_sqrt: sigma.clone().sqrt(),
                    sigma,
                }
            })
            .filter(|residual| {
                residual.distance_to_plane.clone().abs()
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
    #[inline]
    pub fn plane_normal(&self) -> &nalgebra::Vector3<T> {
        &self.plane.normal
    }

    fn probability(&self) -> T {
        self.sigma_sqrt.clone().recip()
            * (self.distance_to_plane_squared.clone() * nalgebra::convert(-0.5)
                / self.sigma.clone())
            .exp()
    }
}

impl<'a, T: Scalar> KFState for Residual<'a, T> {
    type Element = T;
    type Dim = U1;
}
