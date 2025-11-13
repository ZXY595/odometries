use nalgebra::{Point3, Scalar};

use crate::{eskf::state::common::AccState, frame::BodyPoint};

pub type PointChunkStamped<'a, T, P> = (T, &'a [P]);
pub type ImuMeasured<T> = AccState<T>;
pub type ImuMeasuredStamped<'a, T> = (T, &'a ImuMeasured<T>);

pub trait LidarPoint<T: Scalar>: Clone {
    fn to_body_point(self) -> BodyPoint<T>;
}

impl<T: Scalar> LidarPoint<T> for [T; 3] {
    #[inline]
    fn to_body_point(self) -> BodyPoint<T> {
        let point = Point3::from(self);
        BodyPoint::new(point)
    }
}

impl<T: Scalar> LidarPoint<T> for BodyPoint<T> {
    #[inline(always)]
    fn to_body_point(self) -> BodyPoint<T> {
        self
    }
}

impl<T: Scalar, P: LidarPoint<T>> LidarPoint<T> for (P, T) {
    #[inline(always)]
    fn to_body_point(self) -> BodyPoint<T> {
        self.0.to_body_point()
    }
}
