pub(crate) mod marker;
use super::macro_support::*;
use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref, DerefMut},
};

use nalgebra::{
    ClosedAddAssign, DimNameSum, IsometryMatrix3, RealField, Rotation3, Scalar, Storage,
    Translation3, U0, U3, U6, Vector, Vector3,
};
use odometries_macros::{AddAssignVector, KFState};

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
pub type IsometryState<T, S> = MarkedState<IsometryMatrix3<T>, S>;

pub type PoseState<T> = IsometryState<T, marker::Pose>;
pub type RotationState<T> = Vector3State<T, marker::Rotation>;
pub type PositionState<T> = Vector3State<T, marker::Position>;
pub type VelocityState<T> = Vector3State<T, marker::Velocity>;
pub type AccelBiasState<T> = Vector3State<T, marker::AccelBias>;
pub type GyroBiasState<T> = Vector3State<T, marker::GyroBias>;
pub type GravityState<T> = Vector3State<T, marker::Gravity>;
pub type LinearAccelState<T> = Vector3State<T, marker::LinearAccel>;
pub type AngularAccelState<T> = Vector3State<T, marker::AngularAccel>;

#[derive(KFState, AddAssignVector, Default)]
#[Element(T)]
#[Predicates(ClosedAddAssign)]
pub struct BiasState<T: Scalar> {
    pub accel: AccelBiasState<T>,
    pub gyro: GyroBiasState<T>,
}

#[derive(KFState, AddAssignVector, Default)]
#[Element(T)]
#[Predicates(ClosedAddAssign)]
pub struct AccelState<T: Scalar> {
    pub linear: LinearAccelState<T>,
    pub angular: AngularAccelState<T>,
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
        self.0 *= Rotation3::new(rhs.fixed_rows(0));
        self.0 *= Translation3::from(rhs.fixed_rows(3).into_owned());
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
