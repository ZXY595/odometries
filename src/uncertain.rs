use std::ops::{Deref, DerefMut};

use nalgebra::{DefaultAllocator, OMatrix, allocator::Allocator};

use crate::{
    eskf::{Covariance, state::KFState},
    frame::FramedPoint,
};

pub struct Uncertained<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub state: S,
    /// The covariance matrix of the state.
    pub cov: Covariance<S>,
}

impl<S> Deref for Uncertained<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<S> DerefMut for Uncertained<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

pub type UncertainPoint<T, F> = Uncertained<FramedPoint<T, F>>;

#[derive(Debug, Clone)]
pub struct ProcessCovConfig<T> {
    pub distance: T,
    pub direction: T,
}

impl<S> Uncertained<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim, Buffer<S::Element>: Default>,
{
    pub fn new(state: S) -> Self {
        let cov = OMatrix::default();
        Self::new_with_cov(state, cov)
    }
    pub const fn new_with_cov(state: S, cov: OMatrix<S::Element, S::Dim, S::Dim>) -> Self {
        let cov = Covariance(cov);
        Self { state, cov }
    }
}

impl<S> Default for Uncertained<S>
where
    S: KFState + Default,
    DefaultAllocator: Allocator<S::Dim, S::Dim, Buffer<S::Element>: Default>,
{
    fn default() -> Self {
        let cov = Covariance(OMatrix::default());
        Self {
            state: Default::default(),
            cov,
        }
    }
}
