use crate::{
    eskf::uncertain::Uncertained,
    frame::{BodyPoint, ImuPoint, WorldPoint},
};

pub mod body_point;
pub mod plane;
mod residual;
pub mod world_point;
use super::Residual;
use plane::Plane;

pub type UncertainBodyPoint<T> = Uncertained<BodyPoint<T>>;
pub type UncertainImuPoint<T> = Uncertained<ImuPoint<T>>;
pub type UncertainWorldPoint<T> = Uncertained<WorldPoint<T>>;

pub type UncertainPlane<T> = Uncertained<Plane<T>>;
pub type UncertainResidual<'a, T> = Uncertained<Residual<'a, T>>;
