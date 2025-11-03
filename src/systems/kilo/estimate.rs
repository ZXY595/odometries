use std::ops::{AddAssign, Deref};

use super::State;
use crate::{
    eskf::{
        Covariance, DeltaTime, Eskf, StateObserver, StatePredictor,
        state::{KFState, StateDim, SubStateOf, SubStateOffset, common::*},
    },
    utils::{InverseWithSubstitute, Substitutive, ViewDiagonalMut},
};

use nalgebra::{
    Const, DVector, DimAdd, DimMin, DimName, Dyn, OMatrix, OVector, RealField, Rotation3, SMatrix,
    Scalar, Translation3, U1,
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
    pub z: OVector<T, StateDim<PoseState<T>>>,
    /// The measurement noise,
    pub r: OVector<T, StateDim<PoseState<T>>>,
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
        *self.pose *= Rotation3::new(self.acc.angular.deref() * dt.clone());
        *self.pose *= Translation3::from(self.velocity.deref() * dt.clone());
        *self.velocity += (self.pose.deref() * self.acc.linear.deref() + self.gravity.deref()) * dt;
    }
}

impl<T> Eskf<State<T>>
where
    T: RealField,
{
    pub fn predict_cov(&mut self, dt: T) {
        let state = &self.state;

        let mut fx = Covariance::<State<T>>(SMatrix::identity());

        fx.sub_covariance_mut::<RotationState<T>>()
            .copy_from(Rotation3::new(state.acc.angular.deref() * -dt.clone()).matrix());

        fx.sensitivity_mut::<AngularAccelState<T>, RotationState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<VelocityState<T>, PositionState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<RotationState<T>, VelocityState<T>>()
            .copy_from(&(&state.pose.rotation * state.acc.linear.cross_matrix() * -dt.clone()));

        fx.sensitivity_mut::<GravityState<T>, VelocityState<T>>()
            .fill_diagonal(dt.clone());

        fx.sensitivity_mut::<LinearAccelState<T>, VelocityState<T>>()
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

impl<T, const D: usize, const POSE: usize> StateObserver<PointsObserved<T>> for Eskf<State<T>>
where
    T: RealField + Substitutive,
    State<T>: KFState<Element = T, Dim = Const<D>>,
    PoseState<T>: SubStateOf<State<T>, Dim = Const<POSE>>,
{
    fn observe(&mut self, PointsObserved { z, h, r }: PointsObserved<T>) {
        let cov = self.cov.deref();

        let pose_dim = StateDim::<PoseState<T>>::name();
        let pht = cov.columns_generic(0, pose_dim) * h.transpose();
        let hp = &h * cov.rows_generic(0, pose_dim);

        // two Dyn dim all come from the h above,
        // so hpht_r is symmetric.
        let mut hpht_r = h * pht.rows_generic(0, pose_dim);
        hpht_r.view_diagonal_mut().add_assign(r);

        let hpht_r_inv = hpht_r.cholesky_inverse_with_substitute();

        // X*Hx'*SI
        let kalman_gain = pht * hpht_r_inv;

        self.state += &kalman_gain * z;
        *self.cov = self.cov.deref() - kalman_gain * hp;
    }
}

impl<T, const D: usize, const POSE: usize, const ACC_BIAS: usize, const LINEAR_ACC: usize>
    StateObserver<ImuObserved<T>> for Eskf<State<T>>
where
    T: RealField + Substitutive,
    State<T>: KFState<Element = T, Dim = Const<D>>,
    PoseState<T>: SubStateOf<State<T>, Dim = Const<POSE>>,
    AccelBiasState<T>: SubStateOf<State<T>, Dim = Const<ACC_BIAS>>,
    LinearAccelState<T>: SubStateOf<State<T>, Dim = Const<LINEAR_ACC>>,
    BiasState<T>: KFState<Dim = Const<POSE>>,
    AccelState<T>: KFState<Dim = Const<POSE>>,
    // for diagnoal view.
    Const<POSE>: DimAdd<U1> + DimMin<StateDim<PoseState<T>>, Output = StateDim<PoseState<T>>>,
{
    fn observe(&mut self, ImuObserved { z, r }: ImuObserved<T>) {
        let cov = self.cov.deref();

        let pose_dim = StateDim::<PoseState<T>>::name();
        let accel_bias_offset = SubStateOffset::<AccelBiasState<T>, State<T>>::DIM;
        let accel_offset = SubStateOffset::<LinearAccelState<T>, State<T>>::DIM;

        let pht = cov.columns_generic(accel_bias_offset, pose_dim)
            + cov.columns_generic(accel_offset, pose_dim);
        let hp = cov.rows_generic(accel_bias_offset, pose_dim)
            + cov.rows_generic(accel_offset, pose_dim);

        let mut hpht = self.cov.sensitivity::<PoseState<T>, BiasState<T>>()
            + self.cov.sensitivity::<PoseState<T>, AccelState<T>>();
        hpht.view_diagonal_mut().add_assign(r);

        let hpht_r_inv = hpht.cholesky_inverse_with_substitute();

        // Kalman gain, which is X*Hx'*SI
        let kalman_gain = pht * hpht_r_inv;

        self.state += &kalman_gain * z;
        *self.cov = self.cov.deref() - kalman_gain * hp;
    }
}

// TODO: add kinematic imu observation.
