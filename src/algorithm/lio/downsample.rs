use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use nalgebra::ComplexField;

use crate::{
    frame::{BodyPoint, frames},
    voxel_map::index::{ToVoxelIndex, VoxelIndex},
};

type GridIndex<T> = VoxelIndex<T, frames::Body>;
type VoxelGrid<T> = HashMap<GridIndex<T>, (usize, BodyPoint<T>)>;

pub struct Downsampler<T: ComplexField> {
    pub resolution: T,
    pub grid: VoxelGrid<T>,
}

impl<T: ComplexField> Downsampler<T> {
    pub fn new(resolution: T) -> Self {
        Self {
            resolution,
            grid: HashMap::new(),
        }
    }
}

pub trait Downsample<T: ComplexField>: Iterator<Item = BodyPoint<T>> + Sized {
    /// Downsample the points by consuming the [`Iterator`] and collect into the `&mut` [`VoxelGrid<T>`].
    /// After downsampling, a new [`Iterator`] is returned.
    ///
    /// Note that this will not clear the `&mut` [`VoxelGrid<T>`] before downsampling,
    /// instead, it drain the grid after downsampling is done.
    /// If this is not desired, you can [`VoxelGrid<T>::clear`] the grid by yourself
    /// before calling this method.
    fn voxel_grid_downsample(
        self,
        resolution: &T,
        grid: &mut VoxelGrid<T>,
    ) -> impl Iterator<Item = BodyPoint<T>> {
        // TODO: parallel optimizable
        self.for_each(|point| {
            let index = point.as_voxel_index(resolution.clone());

            grid.entry(index)
                .and_modify(|(count, barycenter)| {
                    *count += 1;
                    *barycenter.deref_mut() =
                        barycenter.lerp(point.deref(), nalgebra::convert((*count as f64).recip()));
                })
                .or_insert((1, point));
        });
        grid.drain().map(|(_, (_, point))| point)
    }
}

impl<T, I> Downsample<T> for I
where
    T: ComplexField,
    I: Iterator<Item = BodyPoint<T>>,
{
}
