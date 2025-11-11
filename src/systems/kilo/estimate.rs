use std::ops::{Deref, DerefMut};

use super::State;
use crate::eskf::{
    Covariance, DeltaTime, Eskf, StatePredictor,
    observe::{NoModelObservation, UnbiasedObservation},
    state::{StateDim, common::*},
};

use nalgebra::{Dyn, IsometryMatrix3, OMatrix, RealField, Rotation3, SMatrix, Translation3};

pub type PointsObserved<T> = UnbiasedObservation<PoseState<T>, State<T>, Dyn>;
pub type ImuObserved<T> = NoModelObservation<AccWithBiasState<T>, State<T>>;

#[expect(unused)]
pub struct KinImuObserved<T: RealField> {
    pub zr: ImuObserved<T>,
    pub h: OMatrix<T, StateDim<PoseState<T>>, StateDim<State<T>>>,
}

impl<T> StatePredictor<T> for State<T>
where
    T: RealField,
{
    fn predict(&mut self, dt: T) {
        let acc = &self.acc_with_bias.acc;
        let pose: &mut IsometryMatrix3<T> = self.pose.deref_mut();

        *pose *= Rotation3::new(acc.angular.deref() * dt.clone());
        *pose *= Translation3::from(self.velocity.deref() * dt.clone());
        *self.velocity += (pose.deref() * acc.linear.deref() + self.gravity.deref()) * dt;
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
