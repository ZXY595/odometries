mod default;
mod no_model;
mod transposed;

use nalgebra::{DefaultAllocator, Dim, allocator::Allocator};

use crate::AnyStorageMatrix;

use crate::eskf::state::{KFState, correlation::CorrelateTo};

pub use default::DefaultModel;
pub use no_model::NoModel;

pub trait ObserveModel<S, Super: KFState, D: Dim>
where
    S: CorrelateTo<Super>,
{
    fn new_with_dim(dim: D) -> Self;

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
