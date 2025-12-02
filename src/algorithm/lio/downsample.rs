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
type Grid<T> = HashMap<GridIndex<T>, (usize, BodyPoint<T>)>;

pub struct Downsampler<T: ComplexField> {
    pub resolution: T,
    pub grid: Grid<T>,
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
    fn voxel_grid_downsample<'map>(
        self,
        resolution: &T,
        grid: &'map mut Grid<T>,
    ) -> impl Iterator<Item = &'map BodyPoint<T>> + Clone {
        grid.clear();

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
        grid.values().map(|(_, point)| point)
    }
}

impl<T, I> Downsample<T> for I
where
    T: ComplexField,
    I: Iterator<Item = BodyPoint<T>>,
{
}
