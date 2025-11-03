mod kinematic;

use odometries_macros::{AddAssignVector, KFState};

use crate::eskf::{
    Covariance,
    state::{KFState, common::*, macro_support::*},
};

use nalgebra::{OMatrix, RealField, Scalar, SimdRealField};

#[derive(KFState, AddAssignVector)]
#[Element(T)]
#[Predicates(RealField)]
pub struct State<T: Scalar> {
    /// The pose of the body in the world frame.
    /// use to transform from body to world frame.
    #[SubStates(RotationState<T>, PositionState<T>)]
    pub pose: PoseState<T>,

    pub velocity: VelocityState<T>,

    #[SubStates(AccelBiasState<T>, GyroBiasState<T>)]
    pub bias: BiasState<T>,

    pub gravity: GravityState<T>,

    #[SubStates(LinearAccelState<T>, AngularAccelState<T>)]
    pub acc: AccelState<T>,
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
        cov.sub_covariance_mut::<AccelBiasState<T>>()
            .fill_diagonal(value.accel_bias);
        cov.sub_covariance_mut::<GyroBiasState<T>>()
            .fill_diagonal(value.gyro_bias);
        cov.sub_covariance_mut::<GravityState<T>>()
            .fill_diagonal(value.gravity);
        cov.sub_covariance_mut::<LinearAccelState<T>>()
            .fill_diagonal(value.linear);
        cov.sub_covariance_mut::<AngularAccelState<T>>()
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
            bias: Default::default(),
            gravity: Default::default(),
            acc: Default::default(),
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

        assert_eq!(SubStateOffset::<BiasState<TestT>>::DIM, 9);
        assert_eq!(SubStateEndOffset::<BiasState<TestT>>::DIM, 15);

        assert_eq!(SubStateOffset::<AccelBiasState<TestT>>::DIM, 9);
        assert_eq!(SubStateEndOffset::<AccelBiasState<TestT>>::DIM, 12);

        assert_eq!(SubStateOffset::<GyroBiasState<TestT>>::DIM, 12);
        assert_eq!(SubStateEndOffset::<GyroBiasState<TestT>>::DIM, 15);

        assert_eq!(SubStateOffset::<GravityState<TestT>>::DIM, 15);
        assert_eq!(SubStateEndOffset::<GravityState<TestT>>::DIM, 18);

        assert_eq!(SubStateOffset::<AccelState<TestT>>::DIM, 18);
        assert_eq!(SubStateEndOffset::<AccelState<TestT>>::DIM, 24);

        assert_eq!(SubStateOffset::<LinearAccelState<TestT>>::DIM, 18);
        assert_eq!(SubStateEndOffset::<LinearAccelState<TestT>>::DIM, 21);

        assert_eq!(SubStateOffset::<AngularAccelState<TestT>>::DIM, 21);
        assert_eq!(SubStateEndOffset::<AngularAccelState<TestT>>::DIM, 24);
    }
}
