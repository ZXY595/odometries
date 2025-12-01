use crate::eskf::state::{KFState, SubStateOf};
use std::ops::{Deref, DerefMut};

use nalgebra::{
    CStride, DefaultAllocator, DimName, MatrixView, MatrixViewMut, OMatrix, RStride,
    allocator::Allocator,
};
use num_traits::Zero;

type OwnedSquareMatrix<T, D> = OMatrix<T, D, D>;
type SquareMatrixViewMut<'a, T, R, C, DS> =
    MatrixViewMut<'a, T, R, C, RStride<T, DS, DS>, CStride<T, DS, DS>>;
type SquareMatrixView<'a, T, R, C, DS> =
    MatrixView<'a, T, R, C, RStride<T, DS, DS>, CStride<T, DS, DS>>;

/// # Overview
/// ```text
///     ├──────────  S  ─────────┤
///      ├──x──┤ ├──y──┤ ├──z──┤
/// ┬   ╭────────────────────────╮
/// │ ┬ │                        │
/// │ x │   xx      xy      xz   │
/// │ ┴ │                        │
///   ┬ │                        │
/// S y │   yx      yy      yz   │
///   ┴ │                        │
/// │ ┬ │                        │
/// │ z │   zx      zy      zz   │
/// │ ┴ │                        │
/// ┴   ╰────────────────────────╯
/// ```
#[derive(Debug, Clone)]
pub struct Covariance<S>(pub OwnedSquareMatrix<S::Element, S::Dim>)
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>;

impl<S> Deref for Covariance<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    type Target = OwnedSquareMatrix<S::Element, S::Dim>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S> DerefMut for Covariance<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S> Covariance<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    #[inline]
    pub fn sensitivity_mut<Src, Dst>(
        &mut self,
    ) -> SquareMatrixViewMut<'_, S::Element, Dst::Dim, Src::Dim, S::Dim>
    where
        Src: SubStateOf<S> + ?Sized,
        Dst: SubStateOf<S> + ?Sized,
    {
        self.generic_view_mut(
            (Dst::Offset::DIM, Src::Offset::DIM),
            (Dst::Dim::name(), Src::Dim::name()),
        )
    }

    #[inline]
    pub fn sensitivity<Src, Dst>(
        &self,
    ) -> SquareMatrixView<'_, S::Element, Dst::Dim, Src::Dim, S::Dim>
    where
        Src: SubStateOf<S> + ?Sized,
        Dst: SubStateOf<S> + ?Sized,
    {
        self.generic_view(
            (Dst::Offset::DIM, Src::Offset::DIM),
            (Dst::Dim::name(), Src::Dim::name()),
        )
    }

    #[inline]
    pub fn sub_covariance<Sub>(
        &self,
    ) -> SquareMatrixView<'_, S::Element, Sub::Dim, Sub::Dim, S::Dim>
    where
        Sub: SubStateOf<S> + ?Sized,
    {
        self.sensitivity::<Sub, Sub>()
    }

    #[inline]
    pub fn sub_covariance_mut<Sub>(
        &mut self,
    ) -> SquareMatrixViewMut<'_, S::Element, Sub::Dim, Sub::Dim, S::Dim>
    where
        Sub: SubStateOf<S> + ?Sized,
    {
        self.sensitivity_mut::<Sub, Sub>()
    }
}

impl<S> Default for Covariance<S>
where
    S: KFState<Element: Zero>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    #[inline]
    fn default() -> Self {
        Self(OMatrix::zeros())
    }
}
