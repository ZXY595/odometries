use nalgebra::ComplexField;

use std::{
    hash::{Hash, Hasher},
    ops::Deref,
};

use crate::frame::FramedPoint;

pub type VoxelIndex<T, F> = <FramedPoint<T, F> as ToVoxelIndex<T>>::Index;

pub trait ToVoxelIndex<S> {
    type Index: nohash_hasher::IsEnabled + Eq + Hash;

    fn as_voxel_index(&self, voxel_size: S) -> Self::Index;

    fn to_voxel_index(self) -> Self::Index;
}

impl<T: ComplexField, F> ToVoxelIndex<T> for FramedPoint<T, F> {
    type Index = FramedPoint<i64, F>;

    #[inline]
    fn as_voxel_index(&self, voxel_size: T) -> Self::Index {
        (self / voxel_size).to_voxel_index()
    }

    #[inline]
    fn to_voxel_index(self) -> Self::Index {
        self.map_framed_point(|x| x.floor().to_subset_unchecked())
            .map_framed_point(|x: f64| x as i64)
    }
}

impl<F> Hash for FramedPoint<i64, F> {
    /// see also Optimized Spatial Hashing for Collision Detection of Deformable Objects, Matthias Teschner et. al., VMV 2003
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        hasher.write_i64(((self.x * 73856093) ^ (self.y * 471943) ^ (self.z * 83492791)) % 10000000)
    }
}

/// The [`Hash`] implementation of [`WorldPoint<i64>`] invokes [`write_i64`](Hasher::write_i64)
/// method exactly once.
impl<F> nohash_hasher::IsEnabled for FramedPoint<i64, F> {}

impl<F> PartialEq for FramedPoint<i64, F> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other)
    }
}

impl<F> Eq for FramedPoint<i64, F> {}
