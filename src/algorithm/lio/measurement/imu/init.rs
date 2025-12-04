use std::ops::{Deref, DerefMut};

use nalgebra::{ComplexField, RealField, Vector3};

use crate::eskf::state::common::{AngularAccBiasState, LinearAccState};

use super::ImuMeasuredStamped;

/// The imu initialization, which can be [`collect`](std::iter::Iterator::collect) by
/// [`impl Iterator<Item = ImuMeasured<T>>`](super::ImuMeasured).
///
/// Can be used to create odometries that need imu observation.
///
/// If you need to initialize the odometries without imu observation, don't use this with all zeros.
/// Use methods like [`LIO::new_with_gravity_factor`](crate::algorithm::lio::LIO::new_with_gravity_factor) instead.
#[derive(Debug, Clone)]
pub struct ImuInit<T> {
    pub linear_acc_norm: T,
    pub linear_acc_mean: LinearAccState<T>,
    pub angular_acc_bias: AngularAccBiasState<T>,
    pub timestamp_init: T,
}

impl<T> ImuInit<T>
where
    T: ComplexField<RealField = T>,
{
    pub fn from_gravity(gravity: Vector3<T>) -> Self {
        Self::from_stamped_gravity(T::zero(), gravity)
    }

    pub fn from_stamped_gravity(timestamp: T, gravity: Vector3<T>) -> Self {
        Self {
            linear_acc_norm: gravity.norm(),
            linear_acc_mean: LinearAccState::new(gravity),
            angular_acc_bias: Default::default(),
            timestamp_init: timestamp,
        }
    }
}

impl<T> FromIterator<ImuMeasuredStamped<T>> for Option<ImuInit<T>>
where
    T: RealField,
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

            *mean_acc.linear.deref_mut() = mean_acc
                .linear
                .lerp(current_acc.linear.deref(), n.clone().recip());

            *mean_acc.angular.deref_mut() = mean_acc
                .angular
                .lerp(current_acc.angular.deref(), n.recip());

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

impl<T> Extend<ImuMeasuredStamped<T>> for Option<ImuInit<T>>
where
    T: RealField,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ImuMeasuredStamped<T>>,
    {
        *self = iter.into_iter().collect();
    }
}
