// use nalgebra::Scalar;
//
// use crate::eskf::state::StateMarker;

// macro_rules! impl_state_marker {
//     ($(pub struct $marker:ident => $state:ident;)*) => {
//     $(
//         pub struct $marker;
//         impl StateMarker for $marker {
//             type State<T: Scalar> = super::$state<T>;
//     })*
//     };
// }

pub struct Pose;
pub struct Rotation;
pub struct Position;
pub struct Velocity;
// pub struct Bias;
pub struct AccBias;
pub struct GyroBias;
pub struct Gravity;
// pub struct Accel;
pub struct LinearAcc;
pub struct AngularAcc;
