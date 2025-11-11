mod oct_tree;
mod residual;
pub mod uncertain;

use std::ops::Deref;

use nalgebra::{Point3, RealField, Scalar};
use nohash_hasher::IntMap;
pub use residual::Residual;

use crate::{
    frame::{IsometryFramed, frames},
    utils::ToVoxelIndex,
    voxel_map::{
        oct_tree::OctTreeRoot,
        uncertain::{UncertainWorldPoint, plane::PlaneConfig},
    },
};

type VoxelIndex<T> = <Point3<T> as ToVoxelIndex<T>>::Index;

pub struct VoxelMap<T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex<T>,
{
    roots: IntMap<VoxelIndex<T>, OctTreeRoot<T>>,
    config: Config<T>,
}

pub struct Config<T> {
    plane: PlaneConfig<T>,
    /// residual sigma factor, larger value means more uncertain
    sigma_ratio: T,

    /// downsample leaf size
    #[expect(unused)]
    voxel_grid_resolution: T,

    /// voxel size in the voxel grid
    voxel_size: T,

    /// map size for map sliding window
    #[expect(unused)]
    map_size: usize,

    /// delta pose change threshold to update map sliding window
    #[expect(unused)]
    sliding_thresh: T,

    /// the transform from body(lidar) frame to IMU frame
    #[expect(unused)]
    extrinsic_transform: IsometryFramed<T, fn(frames::Body) -> frames::Imu>,
}

impl<T> VoxelMap<T>
where
    T: Scalar + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    pub fn new(config: Config<T>) -> Self {
        Self {
            roots: IntMap::default(),
            config,
        }
    }
}

impl<T> VoxelMap<T>
where
    T: RealField + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    pub fn insert(&mut self, point: UncertainWorldPoint<T>) {
        let voxel_size = &self.config.voxel_size;
        let index = point.to_voxel_index(voxel_size.clone());
        let root = self
            .roots
            .entry(index)
            .or_insert_with(|| OctTreeRoot::new(point.deref(), voxel_size.clone()));
        root.insert(&self.config.plane, point);
    }
}
impl<T> Extend<UncertainWorldPoint<T>> for VoxelMap<T>
where
    T: RealField + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = UncertainWorldPoint<T>>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter().for_each(|point| self.insert(point));
    }
}
