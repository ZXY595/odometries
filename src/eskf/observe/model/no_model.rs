use std::marker::PhantomData;

use nalgebra::{DefaultAllocator, Dim, OVector, RealField, allocator::Allocator};

use super::ObserveModel;
use crate::{
    AnyStorageMatrix,
    eskf::{
        observe::NoModelObservation,
        state::{KFState, correlation::CorrelateTo},
    },
};

/// A observation model that does not apply any transform to the observation.
pub struct NoModel;

impl<S, Super: KFState> ObserveModel<S, Super, S::CorDim> for NoModel
where
    S: CorrelateTo<Super>,
{
    #[inline(always)]
    fn new_with_dim(_: S::CorDim) -> Self {
        Self
    }

    #[inline(always)]
    fn mul<D2: Dim>(
        &self,
        rhs: AnyStorageMatrix!(Super::Element, S::CorDim, D2),
    ) -> AnyStorageMatrix!(Super::Element, S::CorDim, D2)
    where
        DefaultAllocator: Allocator<S::CorDim, D2>,
    {
        rhs
    }

    #[inline(always)]
    fn tr_mul<D2: Dim>(
        &self,
        lhs: AnyStorageMatrix!(Super::Element, D2, S::CorDim),
    ) -> AnyStorageMatrix!(Super::Element, D2, S::CorDim)
    where
        DefaultAllocator: Allocator<D2, S::CorDim>,
    {
        lhs
    }
}

impl<S, Super> NoModelObservation<S, Super>
where
    Super: KFState<Element: RealField>,
    S: CorrelateTo<Super, Element = Super::Element>,
    DefaultAllocator: Allocator<S::CorDim>,
{
    pub fn new_no_model(
        measurement: OVector<S::Element, S::CorDim>,
        noise: OVector<S::Element, S::CorDim>,
    ) -> Self {
        Self {
            measurement,
            model: NoModel,
            noise,
            _marker: PhantomData,
        }
    }
}
