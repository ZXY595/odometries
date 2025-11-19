use crate::{
    algorithm::lio,
    eskf::state::{common::Vector3State, macro_export::*},
};
use nalgebra::{RealField, Scalar};
use odometries_macros::{KFState, VectorAddAssign};

#[derive(KFState, VectorAddAssign)]
#[element(T)]
#[vector_add_assign(predicates(RealField))]
pub struct State<T>
where
    T: Scalar,
{
    state: lio::state::State<T>,
    kinematic_velocity_bias: KinVelocityBiasState<T>,
    contact_foot_pos: ContactFootPosState<T>,
}

pub struct KinVelocityBias;
type KinVelocityBiasState<T> = Vector3State<T, KinVelocityBias>;

pub struct ContactFootPos;
type ContactFootPosState<T> = Vector3State<T, ContactFootPos>;
