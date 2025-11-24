use std::borrow::{Borrow, BorrowMut};

use odometries_macros::{KFState, VectorAddAssign, sub_state_of};

use crate::eskf::{
    Eskf,
    state::{common::*, macro_export::*},
};

use nalgebra::{ComplexField, RealField, Scalar};

use super::LIO;

#[derive(KFState, VectorAddAssign)]
#[element(T)]
#[vector_add_assign(predicates(RealField))]
pub struct State<T: Scalar> {
    /// The pose of the body in the world frame.
    /// use to transform from body to world frame.
    pub pose: PoseState<T>,

    pub velocity: VelocityState<T>,

    pub gravity: GravityState<T>,

    pub acc_with_bias: AccWithBiasState<T>,
}

#[sub_state_of(State)]
struct PoseState<T: Scalar>(RotationState<T>, PositionState<T>);

#[sub_state_of(State)]
struct AccWithBiasState<T: Scalar>(AccState<T>, BiasState<T>);

#[sub_state_of(State)]
struct AccState<T: Scalar>(LinearAccState<T>, AngularAccState<T>);

#[sub_state_of(State)]
struct BiasState<T: Scalar>(LinearAccBiasState<T>, AngularAccBiasState<T>);

impl<T> Default for State<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            pose: Default::default(),
            velocity: Default::default(),
            gravity: Default::default(),
            acc_with_bias: Default::default(),
        }
    }
}

impl<T> Borrow<Eskf<State<T>>> for LIO<T>
where
    T: ComplexField,
{
    fn borrow(&self) -> &Eskf<State<T>> {
        &self.eskf
    }
}

impl<T> BorrowMut<Eskf<State<T>>> for LIO<T>
where
    T: ComplexField,
{
    fn borrow_mut(&mut self) -> &mut Eskf<State<T>> {
        &mut self.eskf
    }
}

#[cfg(test)]
mod tests {
    use crate::eskf::state::StateDim;

    use super::*;
    use nalgebra::DimName;

    type TestT = f64;

    type State = super::State<f64>;
    type SubStateOffset<S> = super::SubStateOffset<S, State>;
    type SubStateEndOffset<S> = super::SubStateEndOffset<S, State>;

    #[test]
    fn test_error_state_impl() {
        assert_eq!(StateDim::<State>::DIM, 24);

        assert_eq!(SubStateOffset::<PoseState<TestT>>::DIM, 0);
        assert_eq!(SubStateEndOffset::<PoseState<TestT>>::DIM, 6);

        assert_eq!(SubStateOffset::<RotationState<TestT>>::DIM, 0);
        assert_eq!(SubStateEndOffset::<RotationState<TestT>>::DIM, 3);

        assert_eq!(SubStateOffset::<PositionState<TestT>>::DIM, 3);
        assert_eq!(SubStateEndOffset::<PositionState<TestT>>::DIM, 6);

        assert_eq!(SubStateOffset::<VelocityState<TestT>>::DIM, 6);
        assert_eq!(SubStateEndOffset::<VelocityState<TestT>>::DIM, 9);

        assert_eq!(SubStateOffset::<GravityState<TestT>>::DIM, 9);
        assert_eq!(SubStateEndOffset::<GravityState<TestT>>::DIM, 12);

        assert_eq!(SubStateOffset::<AccWithBiasState<TestT>>::DIM, 12);

        assert_eq!(SubStateOffset::<AccState<TestT>>::DIM, 12);

        assert_eq!(SubStateOffset::<LinearAccState<TestT>>::DIM, 12);
        assert_eq!(SubStateEndOffset::<LinearAccState<TestT>>::DIM, 15);
        assert_eq!(SubStateOffset::<AngularAccState<TestT>>::DIM, 15);
        assert_eq!(SubStateEndOffset::<AngularAccState<TestT>>::DIM, 18);

        assert_eq!(SubStateEndOffset::<AccState<TestT>>::DIM, 18);

        assert_eq!(SubStateOffset::<BiasState<TestT>>::DIM, 18);

        assert_eq!(SubStateOffset::<LinearAccBiasState<TestT>>::DIM, 18);
        assert_eq!(SubStateEndOffset::<LinearAccBiasState<TestT>>::DIM, 21);
        assert_eq!(SubStateOffset::<AngularAccBiasState<TestT>>::DIM, 21);
        assert_eq!(SubStateEndOffset::<AngularAccBiasState<TestT>>::DIM, 24);

        assert_eq!(SubStateEndOffset::<BiasState<TestT>>::DIM, 24);

        assert_eq!(SubStateEndOffset::<AccWithBiasState<TestT>>::DIM, 24);
    }
}
