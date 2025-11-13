use std::ops::Deref;

use nalgebra::{RealField, stack};

use crate::{
    algorithm::lio::{ImuInit, estimate::ImuObserved},
    eskf::state::common::{AccState, AccWithBiasState, GravityState},
    utils::ToRadians,
};

use super::LIO;

pub type ImuMeasured<T> = AccState<T>;
pub type ImuMeasuredStamped<'a, T> = (T, &'a ImuMeasured<T>);

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

        ImuObserved::new_no_model(measurement, noise)
    }
}

impl<T> FromIterator<ImuMeasured<T>> for Option<ImuInit<T>>
where
    T: RealField + Default,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ImuMeasured<T>>,
    {
        let mut iter = iter.into_iter().enumerate();

        let (_, first) = iter.next()?;
        let acc_mean = iter.fold(first, |mut mean_acc, (i, current_acc)| {
            let n: T = nalgebra::convert(i as f64);
            mean_acc.linear += (current_acc.linear.deref() - mean_acc.linear.deref()) / n.clone();
            mean_acc.angular += (current_acc.angular.deref() - mean_acc.angular.deref()) / n;
            mean_acc
        });

        let linear_acc_norm = acc_mean.linear.norm();

        Some(ImuInit {
            linear_acc_norm,
            linear_acc_mean: acc_mean.linear,
            angular_acc_bias: acc_mean.angular.map_state_marker(),
        })
    }
}

impl<'a, T> Extend<ImuMeasuredStamped<'a, T>> for LIO<T>
where
    T: RealField + ToRadians + Default,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ImuMeasuredStamped<'a, T>>,
    {
        iter.into_iter().for_each(|(timestamp, acc)| {
            self.eskf_update(timestamp, |ilo| Some(ilo.observe_imu(acc)));
        })
    }
}
