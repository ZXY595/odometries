use std::marker::PhantomData;

use crate::eskf::Covariance;

use nalgebra::{DefaultAllocator, DimName, Matrix, Storage, allocator::Allocator};
use odometries_macros::AnyStorageMatrix;

use super::{KFState, SubStateOf};

/// A trait for a sub-state that is sensitive to a super-state.
pub trait SensitiveTo<Super: KFState>: SubStateOf<Super> {
    type SensiDim: DimName;

    fn sensitivity_to_super(
        cov: &Covariance<Super>,
    ) -> AnyStorageMatrix!(Super::Element, Super::Dim, Self::SensiDim)
    where
        DefaultAllocator: Allocator<Super::Dim, Super::Dim>;

    fn sensitivity_from_super(
        cov: &Covariance<Super>,
    ) -> AnyStorageMatrix!(Super::Element, Self::SensiDim, Super::Dim)
    where
        DefaultAllocator: Allocator<Super::Dim, Super::Dim>;
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
    type EndOffset = S::EndOffset;
}

impl<S, Super> SensitiveTo<Super> for UnbiasedState<S>
where
    S: SubStateOf<Super> + Unbiased,
    Super: SubStateOf<Super>,
    DefaultAllocator: Allocator<Super::Dim, Super::Dim>,
{
    type SensiDim = S::Dim;

    fn sensitivity_to_super(
        cov: &Covariance<Super>,
    ) -> Matrix<
        Super::Element,
        Super::Dim,
        Self::SensiDim,
        impl Storage<Super::Element, Super::Dim, Self::SensiDim>,
    > {
        cov.sensitivity::<Self, Super>()
    }

    fn sensitivity_from_super(
        cov: &Covariance<Super>,
    ) -> Matrix<
        Super::Element,
        Self::SensiDim,
        Super::Dim,
        impl Storage<Super::Element, Self::SensiDim, Super::Dim>,
    > {
        cov.sensitivity::<Super, Self>()
    }
}
