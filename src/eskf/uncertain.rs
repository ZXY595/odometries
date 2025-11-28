use std::ops::{Deref, DerefMut};

use nalgebra::{DefaultAllocator, OMatrix, allocator::Allocator};
use num_traits::{One, Zero};
use simba::scalar::SupersetOf;

use crate::{
    eskf::{Covariance, state::KFState},
    frame::FramedPoint,
};

#[derive(Debug)]
pub struct Uncertained<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub state: S,
    /// The covariance matrix of the state.
    pub cov: Covariance<S>,
}

pub type UncertainPoint<T, F> = Uncertained<FramedPoint<T, F>>;

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

impl<S> Uncertained<S>
where
    S: KFState<Element: One + Zero + SupersetOf<f64>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub fn new(state: S) -> Self {
        let cov = OMatrix::from_diagonal_element(nalgebra::convert(1e-6));
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

impl<S> Uncertained<&S>
where
    for<'s> &'s S: KFState,
    DefaultAllocator: for<'s> Allocator<<&'s S as KFState>::Dim, <&'s S as KFState>::Dim>,
{
    pub fn as_deref_ref(&self) -> &S {
        self.state
    }
}
