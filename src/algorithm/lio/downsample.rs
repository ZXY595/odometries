use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use nalgebra::ComplexField;

use crate::{
    frame::{FramedPoint, frames},
    voxel_map::index::{ToVoxelIndex, VoxelIndex},
};

type GridIndex<T, F> = VoxelIndex<T, F>;
type VoxelGrid<T, F> = HashMap<GridIndex<T, F>, (usize, FramedPoint<T, F>)>;

pub type ScanDownsampler<T> = Downsampler<T, frames::Body>;

pub struct Downsampler<T: ComplexField, F> {
    pub resolution: T,
    pub grid: VoxelGrid<T, F>,
}

impl<T: ComplexField, F> Downsampler<T, F> {
    pub fn new(resolution: T) -> Self {
        Self {
            resolution,
            grid: HashMap::new(),
        }
    }
}

pub trait Downsample<T: ComplexField, F>: Iterator<Item = FramedPoint<T, F>> + Sized {
    /// Downsample the points by consuming the [`Iterator`] and collect into the `&mut` [`VoxelGrid<T>`].
    /// After downsampling, a new [`Iterator`] is returned.
    ///
    /// Note that this will not clear the `&mut` [`VoxelGrid<T>`] before downsampling,
    /// instead, it drain the grid after downsampling is done.
    /// If this is not desired, you can [`VoxelGrid<T>::clear`] the grid by yourself
    /// before calling this method.
    ///
    /// # Example
    /// ```rust
    /// use nalgebra::{Point3, vector};
    /// use odometries::{
    ///     algorithm::lio::downsample::{Downsample, Downsampler},
    ///     frame::BodyPoint
    /// };
    ///
    /// let mut downsampler = Downsampler::new(0.5);
    ///
    /// let points = [[3.0, 3.0, 0.0], [3.2, 3.2, 0.0], [3.4, 3.4, 0.0]];
    ///
    /// let points = points
    ///     .into_iter()
    ///     .map(|p| Point3::from(p))
    ///     .map(BodyPoint::new);
    ///
    /// let points = points
    ///     .voxel_grid_downsample(&downsampler.resolution, &mut downsampler.grid)
    ///     .collect::<Vec<_>>();
    ///
    /// assert_eq!(points.len(), 1);
    /// assert_eq!(points[0].coords, vector![3.2, 3.2, 0.0]);
    /// ```
    fn voxel_grid_downsample(
        self,
        resolution: &T,
        grid: &mut VoxelGrid<T, F>,
    ) -> impl Iterator<Item = FramedPoint<T, F>> {
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

impl<T, F, I> Downsample<T, F> for I
where
    T: ComplexField,
    I: Iterator<Item = FramedPoint<T, F>>,
{
}
