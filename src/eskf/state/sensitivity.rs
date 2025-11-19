use std::marker::PhantomData;

use crate::AnyStorageMatrix;
use nalgebra::{DefaultAllocator, Dim, DimName, allocator::Allocator};

use super::{KFState, SubStateOf};

/// A trait for a sub-state that is sensitive to a super-state.
pub trait SensitiveTo<Super: KFState>: SubStateOf<Super> {
    type SensiDim: DimName;

    fn sensitivity_to_super<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, D, Super::Dim),
    ) -> AnyStorageMatrix!(Super::Element, D, Self::SensiDim)
    where
        DefaultAllocator: Allocator<D, Self::SensiDim>;

    fn sensitivity_from_super<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, Super::Dim, D),
    ) -> AnyStorageMatrix!(Super::Element, Self::SensiDim, D)
    where
        DefaultAllocator: Allocator<Self::SensiDim, D>;
}

pub(crate) type SensitivityDim<S, Super> = <S as SensitiveTo<Super>>::SensiDim;

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

impl<S, Super: KFState> SensitiveTo<Super> for UnbiasedState<S>
where
    S: SubStateOf<Super> + Unbiased,
{
    type SensiDim = S::Dim;

    #[inline(always)]
    fn sensitivity_to_super<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, D, Super::Dim),
    ) -> AnyStorageMatrix!(Super::Element, D, Self::SensiDim) {
        s.columns_generic(S::Offset::DIM, Self::SensiDim::name())
    }

    #[inline(always)]
    fn sensitivity_from_super<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, Super::Dim, D),
    ) -> AnyStorageMatrix!(Super::Element, Self::SensiDim, D) {
        s.rows_generic(S::Offset::DIM, Self::SensiDim::name())
    }
}
