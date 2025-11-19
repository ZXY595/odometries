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

#[derive(Debug)]
pub struct Pose;

#[derive(Debug)]
pub struct Rotation;

#[derive(Debug)]
pub struct Position;

#[derive(Debug)]
pub struct Velocity;
// pub struct Bias;
pub struct AccBias;

#[derive(Debug)]
pub struct GyroBias;

#[derive(Debug)]
pub struct Gravity;

// pub struct Accel;
#[derive(Debug)]
pub struct LinearAcc;

#[derive(Debug)]
pub struct AngularAcc;
