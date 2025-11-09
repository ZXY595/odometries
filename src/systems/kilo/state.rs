mod kinematic;

use odometries_macros::{AnyStorageMatrix, KFState, VectorAddAssign, sub_state_of};

use crate::eskf::{
    Covariance,
    state::{
        StateDim,
        common::*,
        macro_export::*,
        sensitivity::{SensitiveTo, UnbiasedState},
    },
};

use nalgebra::{ClosedAddAssign, OMatrix, RealField, Scalar, SimdRealField};

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

impl<T> SensitiveTo<State<T>> for AccWithBiasState<T>
where
    T: Scalar + ClosedAddAssign,
{
    type SensiDim = StateDim<AccState<T>>;

    fn sensitivity_to_super(
        cov: &Covariance<State<T>>,
    ) -> AnyStorageMatrix!(T, StateDim<State<T>>, Self::SensiDim) {
        UnbiasedState::<AccState<T>>::sensitivity_to_super(cov)
            + UnbiasedState::<BiasState<T>>::sensitivity_to_super(cov)
    }

    fn sensitivity_from_super(
        cov: &Covariance<State<T>>,
    ) -> AnyStorageMatrix!(T, Self::SensiDim, StateDim<State<T>>) {
        UnbiasedState::<AccState<T>>::sensitivity_from_super(cov)
            + UnbiasedState::<BiasState<T>>::sensitivity_from_super(cov)
    }
}

pub struct ProcessCovConfig<T: Scalar> {
    pub velocity: T,
    pub accel_bias: T,
    pub gyro_bias: T,
    pub gravity: T,
    pub linear: T,
    pub angular: T,
}

impl<T> From<ProcessCovConfig<T>> for Covariance<State<T>>
where
    T: Scalar + Default,
{
    fn from(value: ProcessCovConfig<T>) -> Self {
        let mut cov = Covariance::<State<T>>(OMatrix::default());
        cov.sub_covariance_mut::<VelocityState<T>>()
            .fill_diagonal(value.velocity);
        cov.sub_covariance_mut::<LinearAccBiasState<T>>()
            .fill_diagonal(value.accel_bias);
        cov.sub_covariance_mut::<AngularAccBiasState<T>>()
            .fill_diagonal(value.gyro_bias);
        cov.sub_covariance_mut::<GravityState<T>>()
            .fill_diagonal(value.gravity);
        cov.sub_covariance_mut::<LinearAccState<T>>()
            .fill_diagonal(value.linear);
        cov.sub_covariance_mut::<AngularAccState<T>>()
            .fill_diagonal(value.angular);
        cov
    }
}

impl<T> Default for State<T>
where
    T: Default + SimdRealField<Element: SimdRealField>,
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

        assert_eq!(SubStateOffset::<GravityState<TestT>>::DIM, 15);
        assert_eq!(SubStateEndOffset::<GravityState<TestT>>::DIM, 18);

        assert_eq!(SubStateOffset::<AccState<TestT>>::DIM, 9);
        assert_eq!(SubStateEndOffset::<AccState<TestT>>::DIM, 15);
        assert_eq!(SubStateOffset::<LinearAccState<TestT>>::DIM, 9);
        assert_eq!(SubStateEndOffset::<LinearAccState<TestT>>::DIM, 12);
        assert_eq!(SubStateOffset::<AngularAccState<TestT>>::DIM, 12);
        assert_eq!(SubStateEndOffset::<AngularAccState<TestT>>::DIM, 15);

        assert_eq!(SubStateOffset::<BiasState<TestT>>::DIM, 18);
        assert_eq!(SubStateEndOffset::<BiasState<TestT>>::DIM, 24);
        assert_eq!(SubStateOffset::<LinearAccBiasState<TestT>>::DIM, 18);
        assert_eq!(SubStateEndOffset::<LinearAccBiasState<TestT>>::DIM, 21);
        assert_eq!(SubStateOffset::<AngularAccBiasState<TestT>>::DIM, 21);
        assert_eq!(SubStateEndOffset::<AngularAccBiasState<TestT>>::DIM, 24);
    }
}
