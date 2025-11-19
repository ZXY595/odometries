//! Error‑State Kalman Filter.

use core::{
    borrow::BorrowMut,
    ops::{Deref, DerefMut, Sub},
};

use nalgebra::{DefaultAllocator, allocator::Allocator};

mod covariance;
pub mod observe;
pub mod state;
pub use covariance::Covariance;
pub mod uncertain;

use num_traits::{One, Zero};
use simba::scalar::SupersetOf;
use state::KFState;

use uncertain::Uncertained;

pub struct Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    uncertainty: Uncertained<S>,
    pub process_cov: Covariance<S>,
    pub last_update_time: KFTime<S::Element>,
}

#[derive(Debug, Default, Clone)]
pub struct KFTime<T> {
    pub predict: T,
    pub observe: T,
}

pub type DeltaTime<T> = KFTime<T>;

pub trait StatePredictor<T> {
    fn predict(&mut self, dt: T);
}

pub trait StateObserver<T> {
    fn observe(&mut self, measurement: T);
}

pub trait StateFilter<S: KFState>: BorrowMut<Eskf<S>>
where
    Eskf<S>: StatePredictor<DeltaTime<S::Element>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn update<OB>(&mut self, timestamp: S::Element, f: impl FnOnce(&Self) -> Option<OB>)
    where
        S::Element: Sub<Output = S::Element> + Clone,
        Eskf<S>: StateObserver<OB>,
    {
        let eskf = self.borrow_mut();
        let dt = KFTime::all(timestamp.clone()) - eskf.last_update_time.clone();
        eskf.predict(dt);
        eskf.last_update_time.predict = timestamp.clone();

        let observation = f(self);

        if let Some(ob) = observation {
            let eskf = self.borrow_mut();
            eskf.observe(ob);
            eskf.last_update_time.observe = timestamp;
        }
    }
}

impl<S: KFState, T> StateFilter<S> for T
where
    Eskf<S>: StatePredictor<DeltaTime<S::Element>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
    T: BorrowMut<Eskf<S>>,
{
}

impl<S> Eskf<S>
where
    S: KFState<Element: One + Zero + SupersetOf<f64>>,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    pub fn new(process_cov: Covariance<S>, timestamp_init: <S as KFState>::Element) -> Self
    where
        S: Default,
    {
        let state = S::default();
        Self::new_with_state(state, process_cov, timestamp_init)
    }

    pub fn new_with_state(
        state: S,
        process_cov: Covariance<S>,
        timestamp_init: S::Element,
    ) -> Self {
        let uncertain = Uncertained::new(state);
        Self {
            uncertainty: uncertain,
            process_cov,
            last_update_time: KFTime::all(timestamp_init),
        }
    }
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

impl<S> core::borrow::Borrow<KFTime<S::Element>> for Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn borrow(&self) -> &KFTime<S::Element> {
        &self.last_update_time
    }
}

impl<S> BorrowMut<KFTime<S::Element>> for Eskf<S>
where
    S: KFState,
    DefaultAllocator: Allocator<S::Dim, S::Dim>,
{
    fn borrow_mut(&mut self) -> &mut KFTime<S::Element> {
        &mut self.last_update_time
    }
}

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
