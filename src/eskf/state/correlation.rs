use std::marker::PhantomData;

use crate::AnyStorageMatrix;
use nalgebra::{DefaultAllocator, Dim, DimName, allocator::Allocator};

use super::{KFState, SubStateOf};

/// A trait for a sub-state that can be correlated to a super-state.
pub trait CorrelateTo<Super: KFState>: SubStateOf<Super> {
    type SensiDim: DimName;

    fn correlate_to<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, D, Super::Dim),
    ) -> AnyStorageMatrix!(Super::Element, D, Self::SensiDim)
    where
        DefaultAllocator: Allocator<D, Self::SensiDim>;

    fn correlate_from<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, Super::Dim, D),
    ) -> AnyStorageMatrix!(Super::Element, Self::SensiDim, D)
    where
        DefaultAllocator: Allocator<Self::SensiDim, D>;
}

pub(crate) type SensitivityDim<S, Super> = <S as CorrelateTo<Super>>::SensiDim;

/// A [`KFState`] with no inner bias sub-state.
/// but the [`KFState`] that all sub-states are [`Unbiased`] is also considered to be [`Unbiased`].
pub trait Unbiased {}

pub struct UnbiasedState<S: Unbiased>(PhantomData<S>);

impl<S> KFState for UnbiasedState<S>
where
    S: KFState + Unbiased,
{
    type Element = S::Element;
    type Dim = S::Dim;
}

impl<S, Super> SubStateOf<Super> for UnbiasedState<S>
where
    S: SubStateOf<Super> + Unbiased,
    Super: KFState,
{
    type Offset = S::Offset;
}

impl<S, Super: KFState> CorrelateTo<Super> for UnbiasedState<S>
where
    S: SubStateOf<Super> + Unbiased,
{
    type SensiDim = S::Dim;

    #[inline(always)]
    fn correlate_to<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, D, Super::Dim),
    ) -> AnyStorageMatrix!(Super::Element, D, Self::SensiDim) {
        s.columns_generic(S::Offset::DIM, Self::SensiDim::name())
    }

    #[inline(always)]
    fn correlate_from<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, Super::Dim, D),
    ) -> AnyStorageMatrix!(Super::Element, Self::SensiDim, D) {
        s.rows_generic(S::Offset::DIM, Self::SensiDim::name())
    }
}
