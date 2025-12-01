use nalgebra::{ComplexField, Point3};

use std::hash::Hash;

pub type VoxelCoord<T> = <Point3<T> as ToVoxelCoord<T>>::Coord;
pub type VoxelIndex<T> = <VoxelCoord<T> as ToVoxelIndex>::Index;

pub trait ToVoxelIndex {
    type Index: nohash_hasher::IsEnabled + Eq + Hash;
    fn to_voxel_index(self) -> Self::Index;
}

pub trait ToVoxelCoord<S> {
    type Coord: ToVoxelIndex;
    fn as_voxel_coord(&self, voxel_size: S) -> Self::Coord;
    fn to_voxel_coord(self) -> Self::Coord;

    fn as_voxel_index(&self, voxel_size: S) -> <Self::Coord as ToVoxelIndex>::Index;
}

impl<T: ComplexField> ToVoxelCoord<T> for Point3<T> {
    type Coord = Point3<i64>;
    #[inline]
    fn as_voxel_coord(&self, voxel_size: T) -> Self::Coord {
        (self / voxel_size).to_voxel_coord()
    }
    #[inline]
    fn to_voxel_coord(self) -> Self::Coord {
        self.map(|x| x.floor().to_subset_unchecked())
            .map(|x: f64| x as i64)
    }
    #[inline]
    fn as_voxel_index(&self, voxel_size: T) -> <Self::Coord as ToVoxelIndex>::Index {
        self.as_voxel_coord(voxel_size).to_voxel_index()
    }
}

impl ToVoxelIndex for Point3<i64> {
    type Index = i64;
    /// see also Optimized Spatial Hashing for Collision Detection of Deformable Objects, Matthias Teschner et. al., VMV 2003
    #[inline]
    fn to_voxel_index(self) -> Self::Index {
        ((self.x * 73856093) ^ (self.y * 471944) ^ (self.z * 83492791)) % 10000000
    }
}
