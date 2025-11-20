use std::ops::{Deref, DerefMut};

use super::State;
use crate::{
    algorithm::lio::LIO,
    eskf::{Covariance, DeltaTime, Eskf, StatePredictor, state::common::*},
};

use nalgebra::{IsometryMatrix3, RealField, Rotation3, SMatrix, Scalar, Translation3};
use simba::scalar::SupersetOf;

pub struct ProcessCovConfig<T> {
    pub velocity: T,
    pub linear_acc_bias: T,
    pub angular_acc_bias: T,
    pub linear_acc: T,
    pub angular_acc: T,
}

impl<T> StatePredictor<T> for State<T>
where
    T: RealField,
{
    fn predict(&mut self, dt: T) {
        let acc = &self.acc_with_bias.acc;
        let pose: &mut IsometryMatrix3<T> = self.pose.deref_mut();

        let delta_rotation = Rotation3::new(acc.angular.deref() * dt.clone());
        let delta_translation = Translation3::from(self.velocity.deref() * dt.clone());
        let delta_velocity = (pose.deref() * acc.linear.deref() + self.gravity.deref()) * dt;

        *pose *= delta_rotation;
        *pose *= delta_translation;
        *self.velocity += delta_velocity;
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

        fx.sensitivity_mut::<RotationState<T>, VelocityState<T>>()
            .copy_from(&(&state.pose.rotation * acc.linear.cross_matrix() * -dt.clone()));

        fx.sensitivity_mut::<VelocityState<T>, PositionState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<GravityState<T>, VelocityState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<LinearAccState<T>, VelocityState<T>>()
            .copy_from(&(state.pose.rotation.matrix() * -dt.clone()));

        fx.sensitivity_mut::<AngularAccState<T>, RotationState<T>>()
            .fill_diagonal(dt.clone());

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
    fn predict(&mut self, dt: DeltaTime<T>) {
        self.state.predict(dt.predict);
        self.predict_cov(dt.observe);
    }
}

impl<T> StatePredictor<DeltaTime<T>> for LIO<T>
where
    T: RealField,
{
    #[inline]
    fn predict(&mut self, dt: DeltaTime<T>) {
        self.eskf.predict(dt)
    }
}

impl<T: SupersetOf<f64>> Default for ProcessCovConfig<T> {
    fn default() -> Self {
        Self {
            velocity: nalgebra::convert(20.0),
            linear_acc: nalgebra::convert(500.0),
            linear_acc_bias: nalgebra::convert(0.01),
            angular_acc: nalgebra::convert(1000.0),
            angular_acc_bias: nalgebra::convert(0.01),
        }
    }
}

impl<T> From<ProcessCovConfig<T>> for Covariance<State<T>>
where
    T: Scalar + Default,
{
    fn from(value: ProcessCovConfig<T>) -> Self {
        let mut cov = Self::default();

        cov.sub_covariance_mut::<VelocityState<T>>()
            .fill_diagonal(value.velocity);

        cov.sub_covariance_mut::<LinearAccState<T>>()
            .fill_diagonal(value.linear_acc);

        cov.sub_covariance_mut::<LinearAccBiasState<T>>()
            .fill_diagonal(value.linear_acc_bias);

        cov.sub_covariance_mut::<AngularAccState<T>>()
            .fill_diagonal(value.angular_acc);

        cov.sub_covariance_mut::<AngularAccBiasState<T>>()
            .fill_diagonal(value.angular_acc_bias);

        cov
    }
}
