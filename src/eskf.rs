//! Error‑State Kalman Filter.

use std::ops::{Deref, DerefMut};

use nalgebra::{DefaultAllocator, allocator::Allocator};

mod covariance;
pub mod observe;
pub mod state;
pub use covariance::Covariance;
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

pub struct DeltaTime<T> {
    /// The time difference between the current and previous `prediction`.
    pub predict: T,
    /// The time difference between the current and previous `measurement`.
    pub observe: T,
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
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim, Buffer<S::Element>: Default>,
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
