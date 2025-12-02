use nalgebra::{IsometryMatrix3, Matrix3, Point3, Scalar, U3};

use crate::eskf::state::KFState;

use super::Framed;

#[derive(Debug)]
pub struct Body;

#[derive(Debug)]
pub struct Imu;

#[derive(Debug)]
pub struct World;

pub type BodyFramed<T> = Framed<T, Body>;
pub type ImuFramed<T> = Framed<T, Imu>;
pub type WorldFramed<T> = Framed<T, World>;

pub type IsometryFramed<T, F> = Framed<IsometryMatrix3<T>, F>;
pub type CrossMatrixFramed<T, F> = Framed<Matrix3<T>, F>;

pub type FramedPoint<T, F> = Framed<Point3<T>, F>;
pub type BodyPoint<T> = BodyFramed<Point3<T>>;
pub type ImuPoint<T> = ImuFramed<Point3<T>>;
pub type WorldPoint<T> = WorldFramed<Point3<T>>;

impl<T: Scalar, F> KFState for FramedPoint<T, F> {
    type Element = T;
    type Dim = U3;
}

impl<T: Scalar, F> KFState for &Framed<Point3<T>, F> {
    type Element = T;
    type Dim = U3;
}
