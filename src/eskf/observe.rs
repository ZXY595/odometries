mod model;

use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref},
};

pub use model::ObserveModel;
use nalgebra::{
    ClosedMulAssign, DefaultAllocator, Dim, DimAdd, DimMin, OVector, U1, allocator::Allocator,
};
use num_traits::Zero;

use crate::{
    eskf::{
        observe::model::{DefaultModel, NoModel},
        state::{
            KFState,
            correlation::{CorrelateTo, SensitivityDim, UnbiasedState},
        },
    },
    utils::{InverseWithSubstitute, Substitutive, ViewDiagonalMut},
};

use super::{Eskf, StateObserver};

/// The most generic eskf observation.
pub struct Observation<S, Super: KFState, D: Dim, M = DefaultModel<S, Super, D>>
where
    S: CorrelateTo<Super>,
    M: ObserveModel<S, Super, D>,
    DefaultAllocator: Allocator<D>,
{
    /// The measurement vector
    pub measurement: OVector<S::Element, D>,
    /// The measurement matrix, which map state to measurement frame
    pub model: M,
    /// The measurement noise, larger `noise` means more uncertain.
    pub noise: OVector<S::Element, D>,

    _marker: PhantomData<Super>,
}

pub type NoModelObservation<S, Super> = Observation<S, Super, SensitivityDim<S, Super>, NoModel>;

pub type UnbiasedObservation<S, Super, D> = Observation<UnbiasedState<S>, Super, D>;

impl<S, Super: KFState, D: Dim, M> Observation<S, Super, D, M>
where
    S: CorrelateTo<Super, Element: Zero>,
    M: ObserveModel<S, Super, D>,
    DefaultAllocator: Allocator<D>,
{
    pub fn new_with_dim(dim: D) -> Self {
        Self {
            measurement: OVector::zeros_generic(dim, U1),
            model: M::new_with_dim(dim),
            noise: OVector::zeros_generic(dim, U1),
            _marker: PhantomData,
        }
    }
    pub fn get_dim(&self) -> D {
        self.measurement.shape_generic().0
    }
}

impl<S, Super, D: Dim, M> StateObserver<Observation<S, Super, D, M>> for Eskf<Super>
where
    Super: KFState<Element: Substitutive + ClosedMulAssign>
        + AddAssign<OVector<Super::Element, Super::Dim>>,
    S: CorrelateTo<Super, Element = Super::Element>,
    M: ObserveModel<S, Super, D>,
    // for diagonal view
    D: DimMin<D, Output = D> + DimAdd<U1>,
    DefaultAllocator: Allocator<Super::Dim, Super::Dim>
        + Allocator<D, D>
        + Allocator<Super::Dim>
        + Allocator<D>
        + Allocator<D, S::SensiDim>
        + Allocator<S::SensiDim, D>
        + Allocator<Super::Dim, D>
        + Allocator<D, Super::Dim>
        + Allocator<Super::Dim, S::SensiDim>
        + Allocator<S::SensiDim, Super::Dim>,
{
    fn observe(
        &mut self,
        Observation {
            measurement,
            model,
            noise,
            ..
        }: Observation<S, Super, D, M>,
    ) {
        let cross_cov = model.tr_mul(S::correlate_to(&self.cov));

        let innovation_cov = model
            .mul(S::correlate_from(&cross_cov))
            .into_owned()
            .diagonal_add(noise);

        let kalman_gain = cross_cov * innovation_cov.cholesky_inverse_with_substitute();

        self.state += &kalman_gain * measurement;

        *self.cov = self.cov.deref() - kalman_gain * model.mul(S::correlate_from(&self.cov));
    }
}
