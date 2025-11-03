mod oct_tree;
mod residual;
pub mod uncertain;

use nalgebra::{Point3, Scalar};
use nohash_hasher::IntMap;
use typed_arena::Arena;

use crate::{
    frame::{IsometryFramed, frames},
    utils::ToVoxelIndex,
    voxel_map::{oct_tree::OctTreeNode, uncertain::plane::PlaneConfig},
};

type VoxelIndex<T> = <Point3<T> as ToVoxelIndex>::Index;

/// A boxed VoxelMap, use this to avoid lifetime propagation
pub type BoxedVoxelMap<T> = Box<VoxelMap<'static, T>>;

pub struct VoxelMap<'this, T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex,
{
    roots: IntMap<VoxelIndex<T>, OctTreeNode<'this, T>>,
    allocator: Arena<OctTreeNode<'this, T>>,
    config: Config<T>,
}

pub struct Config<T> {
    plane: PlaneConfig<T>,
    /// downsample leaf size
    voxel_grid_resolution: T,
    /// voxel size in the voxel grid
    voxel_size: T,
    /// map size for map sliding window
    map_size: usize,
    /// delta pose change threshold to update map sliding window
    sliding_thresh: T,
    /// the transform from body(lidar) frame to IMU frame
    extrinsic_transform: IsometryFramed<T, fn(frames::Body) -> frames::Imu>,
}

impl<'alloc, T> VoxelMap<'alloc, T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex,
{
    pub fn new(config: Config<T>) -> Self {
        Self {
            roots: IntMap::default(),
            allocator: Arena::new(),
            config,
        }
    }
}

impl<T> VoxelMap<'static, T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex,
{
    pub fn new_boxed(config: Config<T>) -> BoxedVoxelMap<T> {
        Box::new(Self::new(config))
    }
}

// impl<'alloc, T: RealField> Index<WorldPoint<T>> for VoxelMap<'alloc, T> {
//     type Output = Plane<'alloc, T>;
//
//     fn index(&self, point: WorldPoint<T>) -> &Self::Output {
//         let index = point.map(|x| x.floor());
//     }
// }
