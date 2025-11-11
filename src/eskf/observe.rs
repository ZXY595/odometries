mod model;

use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref},
};

pub use model::ObserveModel;
use nalgebra::{
    ClosedMulAssign, DefaultAllocator, Dim, DimAdd, DimMin, DimName, OVector, U1,
    allocator::Allocator,
};
use num_traits::Zero;

use crate::{
    eskf::{
        observe::model::{DefaultModel, NoModel},
        state::{
            KFState,
            sensitivity::{SensitiveTo, SensitivityDim, UnbiasedState},
        },
    },
    utils::{InverseWithSubstitute, Substitutive, ViewDiagonalMut},
};

use super::{Eskf, StateObserver};

/// The most generic eskf observation.
pub struct Observation<S, Super: KFState, D: Dim, M = DefaultModel<S, Super, D>>
where
    S: SensitiveTo<Super>,
    M: ObserveModel<S, Super, D>,
    DefaultAllocator: Allocator<D>,
{
    /// The observed measurement.
    pub measurement: OVector<S::Element, D>,
    /// The measurement function, also known as the observation model or jacobian matrix,
    /// which map state to measurement frame
    pub model: M,
    /// The measurement noise, larger `noise` means more uncertain.
    pub noise: OVector<S::Element, D>,

    _marker: PhantomData<Super>,
}

pub type NoModelObservation<S, Super> = Observation<S, Super, SensitivityDim<S, Super>, NoModel>;

pub type UnbiasedObservation<S, Super, D> = Observation<UnbiasedState<S>, Super, D>;

impl<S, Super: KFState, D: Dim, M> Observation<S, Super, D, M>
where
    S: SensitiveTo<Super, Element: Zero>,
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
    S: SensitiveTo<Super, Element = Super::Element>,
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
        + Allocator<D, Super::Dim>,
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
        let pht = model.tr_mul(S::sensitivity_to_super(&self.cov));

        let mut hpht = model
            .mul(pht.rows_generic(0, S::SensiDim::name()))
            .into_owned();
        hpht.view_diagonal_mut().add_assign(noise);
        let hpht_r = hpht;

        let hpht_r_inv = hpht_r.cholesky_inverse_with_substitute();

        let kalman_gain = pht * hpht_r_inv;

        self.state += &kalman_gain * measurement;

        let hp = model.mul(S::sensitivity_from_super(&self.cov));

        *self.cov = self.cov.deref() - kalman_gain * hp;
    }
}
