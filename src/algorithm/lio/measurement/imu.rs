mod init;
use std::ops::{Deref, DerefMut};

use nalgebra::{RealField, Scalar, stack};

use crate::{
    algorithm::lio::state::State,
    eskf::{
        Eskf,
        observe::NoModelObservation,
        state::common::{AccState, AccWithBiasState},
    },
    utils::ToRadians,
};

use super::LIO;
pub use init::ImuInit;

pub type ImuObserved<T> = NoModelObservation<AccWithBiasState<T>, State<T>>;
pub type ImuMeasured<T> = AccState<T>;

#[derive(Debug)]
pub struct ImuMeasuredStamped<T: Scalar> {
    pub measured: ImuMeasured<T>,
    pub timestamp: T,
}

impl<T> Eskf<State<T>>
where
    T: RealField + ToRadians,
{
    fn observe_imu(
        &self,
        gravity_factor: T,
        measure_noise: &AccState<T>,
        imu_acc: &ImuMeasured<T>,
    ) -> ImuObserved<T> {
        let AccWithBiasState {
            acc: state_acc,
            bias: state_acc_bias,
        } = &self.state.acc_with_bias;

        let measured_linear_acc = imu_acc.linear.deref() * gravity_factor
            - state_acc.linear.deref()
            - state_acc_bias.linear.deref();

        let measured_angular_acc =
            imu_acc.angular.deref() - state_acc.angular.deref() - state_acc_bias.angular.deref();

        #[expect(clippy::toplevel_ref_arg)]
        let measurement = stack![measured_linear_acc; measured_angular_acc];

        #[expect(clippy::toplevel_ref_arg)]
        let noise = stack![measure_noise.linear; measure_noise.angular];

        ImuObserved::new_no_model(measurement, noise)
    }
}

impl<T: Scalar> ImuMeasuredStamped<T> {
    pub fn from_tuple((timesstamp, acc_state): (T, ImuMeasured<T>)) -> Self {
        Self {
            timestamp: timesstamp,
            measured: acc_state,
        }
    }
    pub fn new(timestamp: T, acc_state: ImuMeasured<T>) -> Self {
        Self {
            timestamp,
            measured: acc_state,
        }
    }
}

impl<T: Scalar> Deref for ImuMeasuredStamped<T> {
    type Target = ImuMeasured<T>;

    fn deref(&self) -> &Self::Target {
        &self.measured
    }
}

impl<T: Scalar> DerefMut for ImuMeasuredStamped<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.measured
    }
}

impl<T> Extend<ImuMeasuredStamped<T>> for LIO<T>
where
    T: RealField + ToRadians,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ImuMeasuredStamped<T>>,
    {
        iter.into_iter().for_each(
            |ImuMeasuredStamped {
                 timestamp,
                 measured,
             }| {
                self.eskf.update(timestamp, |eskf| {
                    Some(eskf.observe_imu(
                        self.gravity_factor.clone(),
                        &self.measure_noise.imu_acc,
                        &measured,
                    ))
                });
            },
        )
    }
}
