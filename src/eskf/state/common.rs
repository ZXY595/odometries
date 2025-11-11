pub(crate) mod marker;
use crate::frame::{FramedIsometry, frames};

use super::macro_export::*;
use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref, DerefMut},
};

use nalgebra::{
    ClosedAddAssign, DimNameSum, IsometryMatrix3, RealField, Rotation3, Scalar, Storage,
    Translation3, U0, U3, U6, Vector, Vector3,
};
use odometries_macros::{KFState, Unbiased, VectorAddAssign};

pub struct MarkedState<S, M>(pub S, PhantomData<M>);

impl<S: Default, M> Default for MarkedState<S, M> {
    fn default() -> Self {
        Self(S::default(), PhantomData)
    }
}

impl<S, M> Deref for MarkedState<S, M> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S, M> DerefMut for MarkedState<S, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub type Vector3State<T, S> = MarkedState<Vector3<T>, S>;
pub type IsometryState<T, S> = MarkedState<FramedIsometry<T, fn(frames::Imu) -> frames::World>, S>;

impl<T, S> Unbiased for Vector3State<T, S> {}
impl<T, S> Unbiased for IsometryState<T, S> {}

pub type PoseState<T> = IsometryState<T, marker::Pose>;
pub type RotationState<T> = Vector3State<T, marker::Rotation>;
pub type PositionState<T> = Vector3State<T, marker::Position>;
pub type VelocityState<T> = Vector3State<T, marker::Velocity>;
pub type GravityState<T> = Vector3State<T, marker::Gravity>;
pub type LinearAccState<T> = Vector3State<T, marker::LinearAccel>;
pub type LinearAccBiasState<T> = Vector3State<T, marker::AccelBias>;
pub type AngularAccState<T> = Vector3State<T, marker::AngularAccel>;
pub type AngularAccBiasState<T> = Vector3State<T, marker::GyroBias>;

#[derive(KFState, VectorAddAssign, Default)]
#[element(T)]
#[vector_add_assign(predicates(ClosedAddAssign))]
pub struct AccWithBiasState<T: Scalar> {
    pub acc: AccState<T>,
    pub bias: BiasState<T>,
}

#[derive(KFState, VectorAddAssign, Unbiased, Default)]
#[element(T)]
#[vector_add_assign(predicates(ClosedAddAssign))]
pub struct AccState<T: Scalar> {
    pub linear: LinearAccState<T>,
    pub angular: AngularAccState<T>,
}

#[derive(KFState, VectorAddAssign, Unbiased, Default)]
#[element(T)]
#[vector_add_assign(predicates(ClosedAddAssign))]
pub struct BiasState<T: Scalar> {
    pub linear: LinearAccBiasState<T>,
    pub angular: AngularAccBiasState<T>,
}

impl<T: Scalar, M> super::KFState for Vector3State<T, M> {
    type Element = T;
    type Dim = U3;
}

impl<T, S, M> AddAssign<Vector<T, U3, S>> for Vector3State<T, M>
where
    T: Scalar + ClosedAddAssign,
    S: Storage<T, U3>,
{
    fn add_assign(&mut self, rhs: Vector<T, U3, S>) {
        self.0 += rhs;
    }
}

impl<T: Scalar, M> super::KFState for IsometryState<T, M> {
    type Element = T;
    type Dim = U6;
}

impl<T, S, M> AddAssign<Vector<T, U6, S>> for IsometryState<T, M>
where
    T: RealField,
    S: Storage<T, U6>,
{
    fn add_assign(&mut self, rhs: Vector<T, U6, S>) {
        let pose: &mut IsometryMatrix3<T> = self.deref_mut();
        *pose *= Rotation3::new(rhs.fixed_rows(0));
        *pose *= Translation3::from(rhs.fixed_rows(3).into_owned());
    }
}

impl<T: Scalar> super::SubStateOf<PoseState<T>> for RotationState<T> {
    type Offset = U0;
    type EndOffset = DimNameSum<Self::Offset, Self::Dim>;
}

impl<T: Scalar> super::SubStateOf<PoseState<T>> for PositionState<T> {
    type Offset = U3;
    type EndOffset = DimNameSum<Self::Offset, Self::Dim>;
}
