pub(crate) mod marker;
use crate::{
    AnyStorageMatrix,
    frame::{IsometryFramed, frames},
};

use super::{
    StateDim,
    macro_export::*,
    correlation::{CorrelateTo, UnbiasedState},
};
use std::{
    marker::PhantomData,
    ops::{AddAssign, Deref, DerefMut},
};

use nalgebra::{
    ClosedAddAssign, DefaultAllocator, Dim, IsometryMatrix3, RealField, Rotation3, Scalar, Storage,
    Translation3, U0, U3, U6, Vector, Vector3, allocator::Allocator,
};
use odometries_macros::{KFState, Unbiased, VectorAddAssign};

#[derive(Debug)]
pub struct MarkedState<S, M>(pub S, PhantomData<M>);

pub type Vector3State<T, S> = MarkedState<Vector3<T>, S>;
pub type IsometryState<T, F, S> = MarkedState<IsometryFramed<T, F>, S>;

impl<T, S> Unbiased for Vector3State<T, S> {}
impl<T, F, S> Unbiased for IsometryState<T, F, S> {}

pub type PoseState<T> = IsometryState<T, fn(frames::Imu) -> frames::World, marker::Pose>;
pub type RotationState<T> = Vector3State<T, marker::Rotation>;
pub type PositionState<T> = Vector3State<T, marker::Position>;
pub type VelocityState<T> = Vector3State<T, marker::Velocity>;
pub type GravityState<T> = Vector3State<T, marker::Gravity>;
pub type LinearAccState<T> = Vector3State<T, marker::LinearAcc>;
pub type LinearAccBiasState<T> = Vector3State<T, marker::AccBias>;
pub type AngularAccState<T> = Vector3State<T, marker::AngularAcc>;
pub type AngularAccBiasState<T> = Vector3State<T, marker::GyroBias>;

#[derive(KFState, VectorAddAssign, Default)]
#[element(T)]
#[vector_add_assign(predicates(ClosedAddAssign))]
pub struct AccWithBiasState<T: Scalar> {
    pub acc: AccState<T>,
    pub bias: BiasState<T>,
}

#[derive(Debug, KFState, VectorAddAssign, Unbiased, Default)]
#[element(T)]
#[vector_add_assign(predicates(ClosedAddAssign))]
pub struct AccState<T: Scalar> {
    /// Unit: m/s^2
    pub linear: LinearAccState<T>,
    /// Unit: rad/s
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

impl<T: Scalar, F, M> super::KFState for IsometryState<T, F, M> {
    type Element = T;
    type Dim = U6;
}

impl<T, F, S, M> AddAssign<Vector<T, U6, S>> for IsometryState<T, F, M>
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
}

impl<T: Scalar> super::SubStateOf<PoseState<T>> for PositionState<T> {
    type Offset = U3;
}

impl<T: Scalar, S> Vector3State<T, S> {
    pub fn new_marked(x: T, y: T, z: T) -> Self {
        Self::new(Vector3::new(x, y, z))
    }
}

impl<T: Scalar> AccState<T> {
    pub fn new(acc_x: T, acc_y: T, acc_z: T, gyro_x: T, gyro_y: T, gyro_z: T) -> Self {
        Self {
            linear: Vector3State::new_marked(acc_x, acc_y, acc_z),
            angular: Vector3State::new_marked(gyro_x, gyro_y, gyro_z),
        }
    }
}

impl<Super> CorrelateTo<Super> for AccWithBiasState<Super::Element>
where
    Super: KFState<Element: Scalar + ClosedAddAssign>,
    Self: SubStateOf<Super, Element = Super::Element>,
    AccState<Super::Element>: SubStateOf<Super>,
    BiasState<Super::Element>: SubStateOf<Super, Dim = StateDim<AccState<Super::Element>>>,
{
    type SensiDim = StateDim<AccState<Super::Element>>;

    #[inline]
    fn correlate_to<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, D, Super::Dim),
    ) -> AnyStorageMatrix!(Super::Element, D, Self::SensiDim)
    where
        DefaultAllocator: Allocator<D, Self::SensiDim>,
    {
        UnbiasedState::<AccState<Super::Element>>::correlate_to(s)
            + UnbiasedState::<BiasState<Super::Element>>::correlate_to(s)
    }

    #[inline]
    fn correlate_from<D: Dim>(
        s: &AnyStorageMatrix!(Super::Element, Super::Dim, D),
    ) -> AnyStorageMatrix!(Super::Element, Self::SensiDim, D)
    where
        DefaultAllocator: Allocator<Self::SensiDim, D>,
    {
        UnbiasedState::<AccState<Super::Element>>::correlate_from(s)
            + UnbiasedState::<BiasState<Super::Element>>::correlate_from(s)
    }
}

impl<S, M> MarkedState<S, M> {
    #[inline]
    pub fn new(state: S) -> Self {
        MarkedState(state, PhantomData)
    }
    #[inline]
    pub fn map_state_marker<N>(self) -> MarkedState<S, N> {
        MarkedState(self.0, PhantomData)
    }
}

impl<S: Default, M> Default for MarkedState<S, M> {
    #[inline]
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

impl<S: Clone, M> Clone for MarkedState<S, M> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}
