mod init;
use std::ops::Deref;

use nalgebra::{RealField, Scalar, stack};
use num_traits::Zero;

use crate::{
    algorithm::lio::state::State,
    eskf::{
        Eskf,
        observe::NoModelObservation,
        state::common::{AccState, AccWithBiasState},
    },
    utils::ToRadians,
};

use super::{LIO, StampedMeasurement};
pub use init::ImuInit;

pub type ImuObserved<T> = NoModelObservation<AccWithBiasState<T>, State<T>>;
pub type ImuMeasured<T> = AccState<T>;
pub type StampedImu<T> = StampedMeasurement<T, ImuMeasured<T>>;

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

impl<T: Scalar + Zero> StampedImu<T> {
    pub fn zeros(timestamp: T) -> Self {
        Self {
            timestamp,
            measured: ImuMeasured::default(),
        }
    }
}

impl<T> Extend<StampedImu<T>> for LIO<T>
where
    T: RealField + ToRadians,
{
    fn extend<I>(&mut self, imus: I)
    where
        I: IntoIterator<Item = StampedImu<T>>,
    {
        imus.into_iter().for_each(|imu| {
            self.eskf.update(imu.timestamp, |eskf| {
                Some(eskf.observe_imu(
                    self.gravity_factor.clone(),
                    &self.measure_noise.imu_acc,
                    &imu.measured,
                ))
            });
        })
    }
}
