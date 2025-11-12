//! An inertial–LiDAR tightly‑coupled error‑state Kalman filter odometry system.
//! Most of the ideas comes from Leg-Kilo

pub mod estimate;
pub mod state;

use nalgebra::ComplexField;
use state::State;

use crate::{eskf::Eskf, voxel_map::VoxelMap};

#[expect(unused)]
pub struct Kilo<T: ComplexField> {
    eskf: Eskf<State<T>>,
    map: VoxelMap<T>,
}
