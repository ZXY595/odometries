use crate::eskf::state::{common::Vector3State, macro_support::*};
use nalgebra::{DimNameSum, Scalar};
use odometries_macros::KFState;

use super::State;

#[derive(KFState)]
#[Element(T)]
pub struct KState<T>
where
    T: Scalar,
{
    state: State<T>,
    kinematic_velocity_bias: KinVelocityBiasState<T>,
    contact_foot_pos: ContactFootPosState<T>,
}

pub struct KinVelocityBias;
type KinVelocityBiasState<T> = Vector3State<T, KinVelocityBias>;

pub struct ContactFootPos;
type ContactFootPosState<T> = Vector3State<T, ContactFootPos>;
