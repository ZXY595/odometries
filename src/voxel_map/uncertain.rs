use crate::{
    frame::{BodyPoint, ImuPoint, WorldPoint},
    uncertain::Uncertained,
};

pub mod body_point;
pub mod plane;
pub mod world_point;
use plane::Plane;

pub type UncertainBodyPoint<T> = Uncertained<BodyPoint<T>>;
pub type UncertainImuPoint<T> = Uncertained<ImuPoint<T>>;
pub type UncertainWorldPoint<T> = Uncertained<WorldPoint<T>>;

pub type UncertainPlane<T> = Uncertained<Plane<T>>;
