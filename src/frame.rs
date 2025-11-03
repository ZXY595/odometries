pub mod frames;
use std::{
    marker::PhantomData,
    ops::{Add, Deref, DerefMut, Mul, Sub},
};

pub use frames::*;
use nalgebra::{ClosedAddAssign, Scalar, Vector3};

pub struct Framed<T, F> {
    inner: T,
    frame: PhantomData<F>,
}

impl<T: Clone, F> Clone for Framed<T, F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            frame: PhantomData,
        }
    }
}

impl<T: Default, F> Default for Framed<T, F> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            frame: PhantomData,
        }
    }
}

impl<T, F> Framed<T, F> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            frame: PhantomData,
        }
    }
    pub fn new_with_frame(inner: T, frame: F) -> Self {
        let _ = frame;
        Self::new(inner)
    }
    /// Apply a function to the inner value of the `Framed` type.
    pub fn map_framed(self, f: impl FnOnce(T) -> T) -> Self {
        Self {
            inner: f(self.inner),
            frame: PhantomData,
        }
    }
    pub const fn as_ref(&self) -> Framed<&T, F> {
        let Framed { inner, frame: _ } = self;
        Framed {
            inner,
            frame: PhantomData,
        }
    }
}

impl<T, F1, F2> Framed<T, fn(F1) -> F2> {
    pub fn new_transform(inner: T, from: F1, to: F2) -> Self {
        let _ = (from, to);
        Self {
            inner,
            frame: PhantomData,
        }
    }
}

impl<T, F> Deref for Framed<T, F> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, F> DerefMut for Framed<T, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T1, T2, F> Sub<Framed<T2, F>> for Framed<T1, F>
where
    T1: Sub<T2>,
{
    type Output = Framed<<T1 as Sub<T2>>::Output, F>;
    fn sub(self, transform: Framed<T2, F>) -> Self::Output {
        Framed {
            inner: self.inner - transform.inner,
            frame: PhantomData,
        }
    }
}

impl<'a, T1, T2, F> Sub<&'a Framed<T2, F>> for &'a Framed<T1, F>
where
    &'a T1: Sub<&'a T2>,
{
    type Output = Framed<<&'a T1 as Sub<&'a T2>>::Output, F>;
    fn sub(self, transform: &'a Framed<T2, F>) -> Self::Output {
        Framed {
            inner: self.deref() - transform.deref(),
            frame: PhantomData,
        }
    }
}

impl<T1, T2, F1, F2> Mul<Framed<T2, fn(F1) -> F2>> for Framed<T1, F1>
where
    T2: Mul<T1>,
{
    type Output = Framed<<T2 as Mul<T1>>::Output, F2>;
    fn mul(self, transform: Framed<T2, fn(F1) -> F2>) -> Self::Output {
        Framed {
            inner: transform.inner * self.inner,
            frame: PhantomData,
        }
    }
}

impl<'a, T1, T2, F1, F2> Mul<&'a Framed<T2, fn(F1) -> F2>> for &'a Framed<T1, F1>
where
    &'a T2: Mul<&'a T1>,
{
    type Output = Framed<<&'a T2 as Mul<&'a T1>>::Output, F2>;
    fn mul(self, transform: &'a Framed<T2, fn(F1) -> F2>) -> Self::Output {
        Framed {
            inner: transform.deref() * self.deref(),
            frame: PhantomData,
        }
    }
}

impl<'a, T1, T2, F1, F2, F3> Mul<&'a Framed<T2, fn(F2) -> F3>> for &'a Framed<T1, fn(F1) -> F2>
where
    &'a T1: Mul<&'a T2>,
{
    type Output = Framed<<&'a T1 as Mul<&'a T2>>::Output, fn(F1) -> F3>;
    fn mul(self, transform: &'a Framed<T2, fn(F2) -> F3>) -> Self::Output {
        Framed {
            inner: self.deref() * transform.deref(),
            frame: PhantomData,
        }
    }
}

impl<T1, T2, F1, F2, F3> Mul<Framed<T2, fn(F2) -> F3>> for Framed<T1, fn(F1) -> F2>
where
    T1: Mul<T2>,
{
    type Output = Framed<<T1 as Mul<T2>>::Output, fn(F1) -> F3>;
    fn mul(self, transform: Framed<T2, fn(F2) -> F3>) -> Self::Output {
        Framed {
            inner: self.inner * transform.inner,
            frame: PhantomData,
        }
    }
}

impl<T, F> Add<Vector3<T>> for &FramedPoint<T, F>
where
    T: Scalar + ClosedAddAssign,
{
    type Output = FramedPoint<T, F>;

    fn add(self, rhs: Vector3<T>) -> Self::Output {
        FramedPoint {
            inner: &self.inner + rhs,
            frame: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{IsometryMatrix3, Point3, Vector3};

    use super::*;

    #[test]
    fn test_framed_transform() {
        let p = Point3::new(1.0, 0.0, 0.0);

        let t1 = IsometryMatrix3::new(
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::z() * std::f64::consts::PI,
        );
        let body_2_imu = Framed::new_transform(t1, frames::Body, frames::Imu);

        let t2 = IsometryMatrix3::new(Vector3::new(1.0, 1.0, 0.0), Vector3::zeros());
        let imu_2_world = Framed::new_transform(t2, frames::Imu, frames::World);

        let body_2_world = body_2_imu * imu_2_world;

        let p = Framed::new_with_frame(p, frames::Body) * body_2_world;

        let distance = nalgebra::distance(&p, &Point3::new(-1.0, 0.0, 0.0));
        assert!(distance < 1e-6);
    }
}
