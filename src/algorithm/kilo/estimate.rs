use super::State;
use crate::eskf::{observe::UnbiasedObservation, state::common::*};

use nalgebra::Dyn;

pub type KinImuObserved<T> = UnbiasedObservation<PoseState<T>, State<T>, Dyn>;
