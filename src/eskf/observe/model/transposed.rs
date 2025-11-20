use nalgebra::{DefaultAllocator, Dim, OMatrix, allocator::Allocator};

use crate::eskf::state::{KFState, correlation::CorrelateTo};

/// The transposed observation model which is almost the same as the [`DefaultModel`](super::DefaultModel),
/// but can't be collect from [`Iterator`]
#[expect(unused)]
pub struct TransposedModel<S, Super: KFState, D: Dim>
where
    S: CorrelateTo<Super>,
    DefaultAllocator: Allocator<D, S::SensiDim>,
{
    inner: OMatrix<S::Element, D, S::SensiDim>,
}
