use std::ops::{Deref, DerefMut};

use nalgebra::{RealField, Scalar, stack};

use crate::{
    algorithm::lio::{ImuInit, state::State},
    eskf::{
        StateFilter,
        observe::NoModelObservation,
        state::common::{AccState, AccWithBiasState},
    },
    utils::ToRadians,
};

pub type ImuObserved<T> = NoModelObservation<AccWithBiasState<T>, State<T>>;

use super::LIO;

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

        let linear_measured_acc = linear_acc * self.gravity_norm_factor.clone()
            - state_acc.linear.deref()
            - state_acc_bias.linear.deref();

        let angular_measured_acc =
            angular_acc - state_acc.angular.deref() - state_acc_bias.angular.deref();

        #[expect(clippy::toplevel_ref_arg)]
        let measurement = stack![linear_measured_acc; angular_measured_acc];

        let noise = &self.measure_noise.imu_acc;
        #[expect(clippy::toplevel_ref_arg)]
        let noise = stack![noise.linear; noise.angular];

        ImuObserved::new_no_model(dbg!(measurement), noise)
    }
}

impl<T: Scalar> ImuMeasuredStamped<T> {
    pub fn from_tuple((timesstamp, acc_state): (T, ImuMeasured<T>)) -> Self {
        Self {
            timestamp: timesstamp,
            measured: acc_state,
        }
    }
    pub fn new(timesstamp: T, acc_state: ImuMeasured<T>) -> Self {
        Self {
            timestamp: timesstamp,
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

impl<T> FromIterator<ImuMeasuredStamped<T>> for Option<ImuInit<T>>
where
    T: RealField + Default,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ImuMeasuredStamped<T>>,
    {
        let mut iter = iter.into_iter().enumerate();

        let (_, first) = iter.next()?;
        let ImuMeasuredStamped {
            timestamp,
            measured: acc_mean,
        } = iter.fold(first, |mut mean_acc, (i, current_acc)| {
            let n: T = nalgebra::convert(i as f64);
            mean_acc.measured.linear +=
                (current_acc.linear.deref() - mean_acc.linear.deref()) / n.clone();
            mean_acc.measured.angular +=
                (current_acc.angular.deref() - mean_acc.angular.deref()) / n;
            mean_acc
        });

        let linear_acc_norm = acc_mean.linear.norm();

        Some(ImuInit {
            linear_acc_norm,
            linear_acc_mean: acc_mean.linear,
            angular_acc_bias: acc_mean.angular.map_state_marker(),
            timestamp_init: timestamp,
        })
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
