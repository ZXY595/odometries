mod init;
use std::ops::{Deref, DerefMut};

use nalgebra::{RealField, Scalar, stack};

use crate::{
    algorithm::lio::state::State,
    eskf::{
        StateFilter,
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

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    fn observe_imu(&self, imu_acc: &ImuMeasured<T>) -> ImuObserved<T> {
        let linear_acc = imu_acc.linear.deref();
        let angular_acc = imu_acc.angular.deref();

        let AccWithBiasState {
            acc: state_acc,
            bias: state_acc_bias,
        } = &self.eskf.state.acc_with_bias;

        let linear_measured_acc = linear_acc * self.gravity_factor.clone()
            - state_acc.linear.deref()
            - state_acc_bias.linear.deref();

        let angular_measured_acc =
            angular_acc - state_acc.angular.deref() - state_acc_bias.angular.deref();

        #[expect(clippy::toplevel_ref_arg)]
        let measurement = stack![linear_measured_acc; angular_measured_acc];

        let noise = &self.measure_noise.imu_acc;
        #[expect(clippy::toplevel_ref_arg)]
        let noise = stack![noise.linear; noise.angular];

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

impl<T> Extend<ImuMeasuredStamped<T>> for Option<ImuInit<T>>
where
    T: RealField + Default,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ImuMeasuredStamped<T>>,
    {
        *self = iter.into_iter().collect();
    }
}

impl<T> Extend<ImuMeasuredStamped<T>> for LIO<T>
where
    T: RealField + ToRadians + Default,
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
                self.update(timestamp, |ilo| Some(ilo.observe_imu(&measured)));
            },
        )
    }
}
