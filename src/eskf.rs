//! Error‑State Kalman Filter.

use std::ops::{Deref, DerefMut, Sub};

use nalgebra::{DefaultAllocator, allocator::Allocator};

mod covariance;
pub mod observe;
pub mod state;
pub use covariance::Covariance;
use num_traits::{One, Zero};
use simba::scalar::SupersetOf;
use state::KFState;

use crate::uncertain::Uncertained;

pub struct Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    uncertainty: Uncertained<S>,
    pub process_cov: Covariance<S>,
}

#[derive(Default, Clone)]
pub struct KFTime<T> {
    pub predict: T,
    pub observe: T,
}

pub type DeltaTime<T> = KFTime<T>;

impl<T: Clone> KFTime<T> {
    pub fn all(t: T) -> Self {
        Self {
            predict: t.clone(),
            observe: t,
        }
    }
}

impl<T> Sub for KFTime<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            predict: self.predict - rhs.predict,
            observe: self.observe - rhs.observe,
        }
    }
}

pub trait StatePredictor<T> {
    fn predict(&mut self, dt: T);
}

pub trait StateObserver<T> {
    fn observe(&mut self, measurement: T);
}

impl<S> Deref for Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    type Target = Uncertained<S>;

    fn deref(&self) -> &Self::Target {
        &self.uncertainty
    }
}

impl<S> DerefMut for Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.uncertainty
    }
}

impl<S> Eskf<S>
where
    S: KFState<Element: One + Zero + SupersetOf<f64>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub fn new(process_cov: Covariance<S>) -> Self
    where
        S: Default,
    {
        let state = S::default();
        Self::new_with_state(state, process_cov)
    }

    pub fn new_with_state(state: S, process_cov: Covariance<S>) -> Self {
        let uncertain = Uncertained::new(state);
        Self {
            uncertainty: uncertain,
            process_cov,
        }
    }
}
