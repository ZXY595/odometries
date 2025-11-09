mod model;

use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref},
};

pub use model::ObserveModel;
use nalgebra::{
    ClosedMulAssign, DefaultAllocator, Dim, DimAdd, DimMin, DimName, OMatrix, OVector, RealField,
    U1, allocator::Allocator,
};

use crate::{
    eskf::{
        observe::model::{DefaultObserveModel, NoObserveModel},
        state::{
            KFState,
            sensitivity::{SensitiveTo, SensitivityDim},
        },
    },
    utils::{InverseWithSubstitute, Substitutive, ViewDiagonalMut},
};

use super::{Eskf, StateObserver};

/// The most generic observation model.
pub struct Observation<S, Super, D, M = DefaultObserveModel<S, Super, D>>
where
    S: SensitiveTo<Super>,
    Super: KFState,
    D: Dim,
    M: ObserveModel<S, Super, D>,
    DefaultAllocator: Allocator<D>,
{
    /// The observed measurement.
    measurement: OVector<S::Element, D>,
    /// The measurement function, also known as the observation model or jacobian matrix,
    /// which map state to measurement frame
    model: M,
    /// The measurement noise, larger `noise` means more uncertain.
    noise: OVector<S::Element, D>,

    _marker: PhantomData<Super>,
}

pub type ObservationNoModel<S, Super> =
    Observation<S, Super, SensitivityDim<S, Super>, NoObserveModel>;

impl<S, Super, D: Dim> Observation<S, Super, D>
where
    S::Element: RealField,
    S: SensitiveTo<Super, Element = Super::Element>,
    Super: KFState,
    DefaultAllocator: Allocator<D>
        + Allocator<D, SensitivityDim<S, Super>>
        + Allocator<SensitivityDim<S, Super>, D>,
{
    pub fn new(
        measurement: OVector<S::Element, D>,
        model: OMatrix<S::Element, D, SensitivityDim<S, Super>>,
        noise: OVector<S::Element, D>,
    ) -> Self {
        Self {
            measurement,
            model: DefaultObserveModel(model),
            noise,
            _marker: PhantomData,
        }
    }
}

impl<S, Super> ObservationNoModel<S, Super>
where
    S::Element: RealField,
    S: SensitiveTo<Super, Element = Super::Element>,
    Super: KFState,
    DefaultAllocator: Allocator<SensitivityDim<S, Super>>,
{
    pub fn new_no_model(
        measurement: OVector<S::Element, SensitivityDim<S, Super>>,
        noise: OVector<S::Element, SensitivityDim<S, Super>>,
    ) -> Self {
        Self {
            measurement,
            model: NoObserveModel,
            noise,
            _marker: PhantomData,
        }
    }
}

impl<S, Super, D: Dim, M> StateObserver<Observation<S, Super, D, M>> for Eskf<Super>
where
    Super::Element: Substitutive + ClosedMulAssign,
    Super: KFState + AddAssign<OVector<Super::Element, Super::Dim>>,
    S: SensitiveTo<Super, Element = Super::Element>,
    M: ObserveModel<S, Super, D>,
    // for diagonal view
    D: DimMin<D, Output = D> + DimAdd<U1>,
    DefaultAllocator: Allocator<Super::Dim, Super::Dim>
        + Allocator<D, D>
        + Allocator<Super::Dim>
        + Allocator<D>
        + Allocator<D, SensitivityDim<S, Super>>
        + Allocator<SensitivityDim<S, Super>, D>
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
        let cov = &self.cov;

        let pht = model.tr_mul(S::sensitivity_to_super(cov));

        let hp = model.mul(S::sensitivity_from_super(cov)).into_owned();

        let mut hpht = model
            .mul(pht.rows_generic(0, S::SensiDim::name()))
            .into_owned();
        hpht.view_diagonal_mut().add_assign(noise);
        let hpht_r = hpht;

        let hpht_r_inv = hpht_r.cholesky_inverse_with_substitute();

        let kalman_gain = pht * hpht_r_inv;

        self.state += &kalman_gain * measurement;
        *self.cov = self.cov.deref() - kalman_gain * hp;
    }
}
