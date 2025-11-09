use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, DefaultAllocator, Dim, OMatrix, Scalar, allocator::Allocator,
};
use num_traits::{One, Zero};

use odometries_macros::AnyStorageMatrix;

use crate::eskf::state::{
    KFState,
    sensitivity::{SensitiveTo, SensitivityDim},
};

pub trait ObserveModel<S, Super: KFState, D: Dim>
where
    S: SensitiveTo<Super>,
{
    fn mul<D2: Dim>(
        &self,
        rhs: AnyStorageMatrix!(Super::Element, S::SensiDim, D2),
    ) -> AnyStorageMatrix!(Super::Element, D, D2)
    where
        DefaultAllocator: Allocator<D, D2>;

    fn tr_mul<D2: Dim>(
        &self,
        lhs: AnyStorageMatrix!(Super::Element, D2, S::SensiDim),
    ) -> AnyStorageMatrix!(Super::Element, D2, D)
    where
        DefaultAllocator: Allocator<D2, D>;
}

pub struct DefaultObserveModel<S, Super: KFState, D: Dim>(
    pub(super) OMatrix<S::Element, D, SensitivityDim<S, Super>>,
)
where
    S: SensitiveTo<Super>,
    DefaultAllocator: Allocator<D, SensitivityDim<S, Super>>;

impl<S, Super: KFState, D: Dim> ObserveModel<S, Super, D> for DefaultObserveModel<S, Super, D>
where
    Super::Element: Scalar + Zero + One + ClosedMulAssign + ClosedAddAssign,
    S: SensitiveTo<Super, Element = Super::Element>,
    DefaultAllocator:
        Allocator<D, SensitivityDim<S, Super>> + Allocator<SensitivityDim<S, Super>, D>,
{
    #[inline(always)]
    fn mul<D2: Dim>(
        &self,
        rhs: AnyStorageMatrix!(Super::Element, S::SensiDim, D2),
    ) -> AnyStorageMatrix!(Super::Element, D, D2)
    where
        DefaultAllocator: Allocator<D, D2>,
    {
        &self.0 * rhs
    }

    #[inline(always)]
    fn tr_mul<D2: Dim>(
        &self,
        lhs: AnyStorageMatrix!(Super::Element, D2, S::SensiDim),
    ) -> AnyStorageMatrix!(Super::Element, D2, D)
    where
        DefaultAllocator: Allocator<D2, D>,
    {
        lhs * self.0.transpose()
    }
}

pub struct NoObserveModel;

impl<S, Super: KFState> ObserveModel<S, Super, S::SensiDim> for NoObserveModel
where
    S: SensitiveTo<Super>,
{
    #[inline(always)]
    fn mul<D2: Dim>(
        &self,
        rhs: AnyStorageMatrix!(Super::Element, S::SensiDim, D2),
    ) -> AnyStorageMatrix!(Super::Element, S::SensiDim, D2)
    where
        DefaultAllocator: Allocator<S::SensiDim, D2>,
    {
        rhs
    }

    #[inline(always)]
    fn tr_mul<D2: Dim>(
        &self,
        lhs: AnyStorageMatrix!(Super::Element, D2, S::SensiDim),
    ) -> AnyStorageMatrix!(Super::Element, D2, S::SensiDim)
    where
        DefaultAllocator: Allocator<D2, S::SensiDim>,
    {
        lhs
    }
}
