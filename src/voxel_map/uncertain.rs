use crate::{
    eskf::uncertain::Uncertained,
    frame::{BodyPoint, ImuPoint, WorldPoint},
};

pub mod body_point;
pub mod plane;
pub mod world_point;
use plane::Plane;

pub type UncertainBodyPoint<'a, T> = Uncertained<&'a BodyPoint<T>>;
pub type UncertainImuPoint<T> = Uncertained<ImuPoint<T>>;
pub type UncertainWorldPoint<T> = Uncertained<WorldPoint<T>>;

pub type UncertainPlane<T> = Uncertained<Plane<T>>;
