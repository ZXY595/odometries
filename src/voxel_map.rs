mod index;
mod oct_tree;
mod residual;
pub mod uncertain;

use nalgebra::{ComplexField, RealField};
use nohash_hasher::IntMap;
pub use residual::Residual;
use simba::scalar::SupersetOf;

use crate::voxel_map::{
    index::{ToVoxelIndex, VoxelIndex},
    oct_tree::OctTreeRoot,
    uncertain::{UncertainWorldPoint, plane::PlaneConfig},
};

pub struct VoxelMap<T>
where
    T: ComplexField,
{
    roots: IntMap<VoxelIndex<T>, OctTreeRoot<T>>,
    config: Config<T>,
}

pub struct Config<T> {
    pub plane: PlaneConfig<T>,
    /// residual sigma factor, larger value means more uncertain
    pub sigma_ratio: T,

    /// voxel size in the voxel grid
    pub voxel_size: T,

    /// map size for map sliding window
    pub map_size: usize,

    /// delta pose change threshold to update map sliding window
    pub sliding_thresh: T,
}

impl<T> Default for Config<T>
where
    T: Clone + SupersetOf<f64>,
{
    fn default() -> Self {
        let voxel_size: T = nalgebra::convert(0.5);
        Self {
            plane: Default::default(),
            sigma_ratio: nalgebra::convert(3.0),
            voxel_size,
            map_size: 200,
            sliding_thresh: nalgebra::convert(8.0),
        }
    }
}

impl<T> VoxelMap<T>
where
    T: ComplexField,
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
    T: RealField,
{
    pub fn insert(&mut self, point: UncertainWorldPoint<T>) {
        let voxel_size = &self.config.voxel_size;
        let index = point.as_voxel_index(voxel_size.clone());
        let root = self
            .roots
            .entry(index)
            .or_insert_with(|| OctTreeRoot::new(&point, voxel_size.clone()));
        root.insert(&self.config.plane, point);
    }
}
impl<T> Extend<UncertainWorldPoint<T>> for VoxelMap<T>
where
    T: RealField,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = UncertainWorldPoint<T>>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter().for_each(|point| self.insert(point));
    }
}
