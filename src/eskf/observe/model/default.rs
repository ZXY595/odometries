use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    AnyStorageMatrix,
    eskf::{
        observe::Observation,
        state::{KFState, correlation::CorrelateTo},
    },
};
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, DefaultAllocator, Dim, DimName, Dyn, Matrix, OMatrix,
    OVector, RealField, Scalar, allocator::Allocator,
};

use super::ObserveModel;
use num_traits::{One, Zero};

/// The default observation model, which can be [`collect`](std::iter::Iterator::collect) from [`Iterator`].
pub struct DefaultModel<S, Super: KFState, D: Dim>
where
    S: CorrelateTo<Super>,
    DefaultAllocator: Allocator<S::CorDim, D>,
{
    /// Make dimension `D` be the last dimension, this is useful for matrix [`Extend`](std::iter::Extend) operations.
    inner: OMatrix<S::Element, S::CorDim, D>,
}

impl<S, Super: KFState, D: Dim> ObserveModel<S, Super, D> for DefaultModel<S, Super, D>
where
    Super::Element: Scalar + Zero + One + ClosedMulAssign + ClosedAddAssign,
    S: CorrelateTo<Super, Element = Super::Element>,
    DefaultAllocator: Allocator<D, S::CorDim> + Allocator<S::CorDim, D>,
{
    #[inline(always)]
    fn new_with_dim(dim: D) -> Self {
        Self {
            inner: OMatrix::zeros_generic(S::CorDim::name(), dim),
        }
    }
    #[inline(always)]
    fn mul<D2: Dim>(
        &self,
        rhs: AnyStorageMatrix!(Super::Element, S::CorDim, D2),
    ) -> AnyStorageMatrix!(Super::Element, D, D2)
    where
        DefaultAllocator: Allocator<D, D2>,
    {
        Matrix::tr_mul(self, &rhs)
    }

    #[inline(always)]
    fn tr_mul<D2: Dim>(
        &self,
        lhs: AnyStorageMatrix!(Super::Element, D2, S::CorDim),
    ) -> AnyStorageMatrix!(Super::Element, D2, D)
    where
        DefaultAllocator: Allocator<D2, D>,
    {
        lhs * self.deref()
    }
}

impl<S, Super: KFState, D: Dim> Deref for DefaultModel<S, Super, D>
where
    S: CorrelateTo<Super>,
    DefaultAllocator: Allocator<S::CorDim, D>,
{
    type Target = OMatrix<S::Element, S::CorDim, D>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<S, Super: KFState, D: Dim> DerefMut for DefaultModel<S, Super, D>
where
    S: CorrelateTo<Super>,
    DefaultAllocator: Allocator<S::CorDim, D>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<S, Super, D: Dim> Observation<S, Super, D>
where
    Super: KFState<Element: RealField>,
    S: CorrelateTo<Super, Element = Super::Element>,
    DefaultAllocator: Allocator<D> + Allocator<D, S::CorDim> + Allocator<S::CorDim, D>,
{
    pub fn new(
        measurement: OVector<S::Element, D>,
        model: OMatrix<S::Element, S::CorDim, D>,
        noise: OVector<S::Element, D>,
    ) -> Self {
        Self {
            measurement,
            model: DefaultModel { inner: model },
            noise,
            _marker: PhantomData,
        }
    }
}

type ObservationIterItem<T, D> = (T, OVector<T, D>, T);

impl<S, Super: KFState> FromIterator<ObservationIterItem<S::Element, S::CorDim>>
    for Observation<S, Super, Dyn>
where
    Super: KFState<Element: RealField>,
    S: CorrelateTo<Super, Element = Super::Element>,
    DefaultAllocator: Allocator<S::CorDim>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ObservationIterItem<S::Element, S::CorDim>>,
    {
        let (measurement, model, noise) = iter.into_iter().collect();
        Self::new(measurement, model, noise)
    }
}
