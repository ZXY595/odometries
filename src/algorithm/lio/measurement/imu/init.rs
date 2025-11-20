use std::ops::Deref;

use nalgebra::RealField;

use crate::eskf::state::common::{AngularAccBiasState, LinearAccState};

use super::ImuMeasuredStamped;

/// The imu initialization, which can be created by
/// [`impl Iterator<Item = ImuMeasured<T>>::collect`](std::iter::Iterator::collect).
///
/// Can be used to create odometries that need imu observation.
#[derive(Debug, Clone)]
pub struct ImuInit<T> {
    pub linear_acc_norm: T,
    pub linear_acc_mean: LinearAccState<T>,
    pub angular_acc_bias: AngularAccBiasState<T>,
    pub timestamp_init: T,
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

        // yield and skip index 0 here
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

        Some(ImuInit {
            linear_acc_norm: acc_mean.linear.norm(),
            linear_acc_mean: acc_mean.linear,
            angular_acc_bias: acc_mean.angular.map_state_marker(),
            timestamp_init: timestamp,
        })
    }
}
