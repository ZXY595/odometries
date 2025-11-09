use std::ops::Deref;

use super::State;
use crate::{
    eskf::{
        Covariance, DeltaTime, Eskf, StateObserver, StatePredictor,
        observe::{Observation, ObservationNoModel},
        state::{StateDim, common::*, sensitivity::UnbiasedState},
    },
    utils::Substitutive,
};

use nalgebra::{
    DVector, Dyn, OMatrix, OVector, RealField, Rotation3, SMatrix, Scalar, Translation3,
};

pub struct PointsObserved<T: Scalar> {
    /// The observed measurement.
    pub z: DVector<T>,
    /// The measurement function, also known as the observation model or jacobian matrix,
    /// which map state to measurement frame
    pub h: OMatrix<T, Dyn, StateDim<PoseState<T>>>,
    /// The measurement noise,
    /// larger r means more uncertain.
    pub r: DVector<T>,
}

pub struct ImuObserved<T: Scalar> {
    /// The observed measurement.
    pub z: OVector<T, StateDim<AccState<T>>>,
    /// The measurement noise,
    pub r: OVector<T, StateDim<AccState<T>>>,
}

#[expect(unused)]
pub struct KinImuObserved<T: Scalar> {
    pub zr: ImuObserved<T>,
    pub h: OMatrix<T, StateDim<PoseState<T>>, StateDim<State<T>>>,
}

impl<T> StatePredictor<T> for State<T>
where
    T: RealField,
{
    fn predict(&mut self, dt: T) {
        let acc = &self.acc_with_bias.acc;
        *self.pose *= Rotation3::new(acc.angular.deref() * dt.clone());
        *self.pose *= Translation3::from(self.velocity.deref() * dt.clone());
        *self.velocity += (self.pose.deref() * acc.linear.deref() + self.gravity.deref()) * dt;
    }
}

impl<T> Eskf<State<T>>
where
    T: RealField,
{
    pub fn predict_cov(&mut self, dt: T) {
        let state = &self.state;
        let acc = &self.acc_with_bias.acc;

        let mut fx = Covariance::<State<T>>(SMatrix::identity());

        fx.sub_covariance_mut::<RotationState<T>>()
            .copy_from(Rotation3::new(acc.angular.deref() * -dt.clone()).matrix());

        fx.sensitivity_mut::<AngularAccState<T>, RotationState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<VelocityState<T>, PositionState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<RotationState<T>, VelocityState<T>>()
            .copy_from(&(&state.pose.rotation * acc.linear.cross_matrix() * -dt.clone()));

        fx.sensitivity_mut::<GravityState<T>, VelocityState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<LinearAccState<T>, VelocityState<T>>()
            .copy_from(&(state.pose.rotation.matrix() * -dt.clone()));

        // X = Fx.X.FX' + dt^2 * Q
        let mut cov = self.process_cov.deref() * dt.powi(2);
        cov.quadform_tr(T::one(), &fx, self.cov.deref(), T::one());
        *self.cov = cov;
    }
}

impl<T> StatePredictor<DeltaTime<T>> for Eskf<State<T>>
where
    T: RealField,
{
    fn predict(&mut self, DeltaTime { predict, observe }: DeltaTime<T>) {
        self.state.predict(predict);
        self.predict_cov(observe);
    }
}

impl<T> StateObserver<PointsObserved<T>> for Eskf<State<T>>
where
    T: RealField + Substitutive,
{
    fn observe(&mut self, PointsObserved { z, h, r }: PointsObserved<T>) {
        self.observe(Observation::<UnbiasedState<PoseState<T>>, State<T>, Dyn>::new(z, h, r))
    }
}

impl<T> StateObserver<ImuObserved<T>> for Eskf<State<T>>
where
    T: RealField + Substitutive,
{
    fn observe(&mut self, ImuObserved { z, r }: ImuObserved<T>) {
        self.observe(ObservationNoModel::<AccWithBiasState<T>, State<T>>::new_no_model(z, r));
    }
}

// TODO: add kinematic imu observation.
