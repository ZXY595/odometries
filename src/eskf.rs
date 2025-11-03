//! Error‑State Kalman Filter.

use std::ops::{Deref, DerefMut};

use nalgebra::{
    CStride, DefaultAllocator, DimName, MatrixView, MatrixViewMut, OMatrix, RStride,
    allocator::Allocator,
};

pub mod state;
use state::KFState;

use crate::{
    eskf::state::{SubStateOf},
    uncertain::Uncertained,
};

pub struct Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    uncertain: Uncertained<S>,
    pub process_cov: Covariance<S>,
}

impl<S> Deref for Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    type Target = Uncertained<S>;

    fn deref(&self) -> &Self::Target {
        &self.uncertain
    }
}

impl<S> DerefMut for Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uncertain
    }
}

impl<S> Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim, Buffer<S::Element>: Default>,
{
    pub fn new(process_cov: Covariance<S>) -> Self
    where
        S: Default,
    {
        let state = S::default();
        Self::new_with_state(state, process_cov)
    }

    pub fn new_with_state(state: S, process_cov: Covariance<S>) -> Self {
        let uncertain = Uncertained::new(state);
        Self {
            uncertain,
            process_cov,
        }
    }
}

type OwnedSquareMatrix<T, D> = OMatrix<T, D, D>;
type SquareViewMut<'a, T, D, DS> =
    MatrixViewMut<'a, T, D, D, RStride<T, DS, DS>, CStride<T, DS, DS>>;
type SquareView<'a, T, D, DS> = MatrixView<'a, T, D, D, RStride<T, DS, DS>, CStride<T, DS, DS>>;

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
    pub fn sensitivity_mut<Src, Dst>(&mut self) -> SquareViewMut<'_, S::Element, Src::Dim, S::Dim>
    where
        Src: SubStateOf<S>,
        Dst: SubStateOf<S, Dim = Src::Dim>,
    {
        self.generic_view_mut(
            (Dst::Offset::DIM, Src::Offset::DIM),
            (Dst::Dim::name(), Src::Dim::name()),
        )
    }

    pub fn sensitivity<Src, Dst>(&self) -> SquareView<'_, S::Element, Src::Dim, S::Dim>
    where
        Src: SubStateOf<S>,
        Dst: SubStateOf<S, Dim = Src::Dim>,
    {
        self.generic_view(
            (Dst::Offset::DIM, Src::Offset::DIM),
            (Dst::Dim::name(), Src::Dim::name()),
        )
    }

    pub fn sub_covariance<Sub>(&self) -> SquareView<'_, S::Element, Sub::Dim, S::Dim>
    where
        Sub: SubStateOf<S>,
    {
        self.sensitivity::<Sub, Sub>()
    }

    pub fn sub_covariance_mut<Sub>(&mut self) -> SquareViewMut<'_, S::Element, Sub::Dim, S::Dim>
    where
        Sub: SubStateOf<S>,
    {
        self.sensitivity_mut::<Sub, Sub>()
    }
}

pub struct DeltaTime<T> {
    /// The time difference between the current and previous `prediction`.
    pub predict: T,
    /// The time difference between the current and previous `measurement`.
    pub observe: T,
}

pub trait StatePredictor<T> {
    fn predict(&mut self, dt: T);
}

pub trait StateObserver<T> {
    fn observe(&mut self, measurement: T);
}
