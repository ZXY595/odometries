//! Most of ideas comes from Leg-Kilo

pub mod estimate;
pub mod measurement;
pub mod state;

use std::ops::Deref;

use state::State;

use crate::{
    eskf::{
        Eskf, KFTime, StateObserver, StatePredictor,
        state::common::{AccState, AngularAccBiasState, GravityState, LinearAccState},
    },
    frame::{FramedIsometry, frames},
    utils::ToRadians,
    voxel_map::{self, VoxelMap, uncertain::body_point},
};

use nalgebra::{ComplexField, RealField, Scalar};

pub use body_point::ProcessCovConfig as BodyPointProcessCovConfig;
pub use state::ProcessCovConfig as StateProcessCovConfig;

/// # Input
/// ```text
/// ├───┬─>>─┬─── timestamp ───>>──┬───┬───┤
///     │    │        │            │   │
///     │    │        ┴            │   │
///     │    │   LiDAR points      │   │
///     │    │                     │   │
///     │    │               IMU ├─╯   │
///     │    │                         │
///     │    ╰─┤ LiDAR points          │
///     │                              │
///     ╰─┤ IMU                        │
///                                    │
///                      LiDAR point ├─╯
/// ```
pub struct LIO<T>
where
    T: ComplexField,
{
    eskf: Eskf<State<T>>,
    last_update_time: KFTime<T>,
    map: VoxelMap<T>,
    // configs
    body_point_process_cov: BodyPointProcessCovConfig<T>,
    extrinsics: FramedIsometry<T, fn(frames::Body) -> frames::Imu>,
    measure_noise: MeasureNoiseConfig<T>,
    gravity_norm_factor: T,
}

pub struct Config<T: Scalar> {
    pub process_cov: ProcessCovConfig<T>,
    pub voxel_map: voxel_map::Config<T>,
    pub extrinsics: FramedIsometry<T, fn(frames::Body) -> frames::Imu>,
    pub measure_noise: MeasureNoiseConfig<T>,
    pub gravity: T,
}

pub struct MeasureNoiseConfig<T: Scalar> {
    pub imu_acc: AccState<T>,
    lidar_point: T,
}

pub struct ProcessCovConfig<T: Scalar> {
    state: StateProcessCovConfig<T>,
    body_point: BodyPointProcessCovConfig<T>,
}

/// The imu initialization returned by
/// [`impl Iterator<Item = ImuMeasured<T>>::collect`](std::iter::Iterator::collect)
///
/// Can be used to create odometries that need imu observation.
pub struct ImuInit<T> {
    linear_acc_norm: T,
    linear_acc_mean: LinearAccState<T>,
    angular_acc_bias: AngularAccBiasState<T>,
}

impl<T> ImuInit<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn new_lio(self, config: Config<T>) -> LIO<T> {
        LIO::new(self, config)
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn new(imu_init: ImuInit<T>, config: Config<T>) -> Self {
        let mut eskf = Eskf::new(config.process_cov.state.into());

        let gravity_norm_factor = config.gravity / imu_init.linear_acc_norm.clone();

        eskf.gravity =
            GravityState::new(-imu_init.linear_acc_mean.deref() * gravity_norm_factor.clone());
        eskf.acc_with_bias.bias.angular = imu_init.angular_acc_bias;

        Self {
            eskf,
            last_update_time: Default::default(),
            map: VoxelMap::new(config.voxel_map),
            body_point_process_cov: config.process_cov.body_point,
            extrinsics: config.extrinsics,
            gravity_norm_factor,
            measure_noise: config.measure_noise,
        }
    }

    pub fn get_pose(&self) -> &FramedIsometry<T, fn(frames::Imu) -> frames::World> {
        &self.eskf.pose.0
    }

    fn eskf_update<OB>(&mut self, timestamp: T, f: impl FnOnce(&Self) -> Option<OB>)
    where
        Eskf<State<T>>: StateObserver<OB>,
    {
        self.eskf
            .predict(KFTime::all(timestamp.clone()) - self.last_update_time.clone());
        self.last_update_time.predict = timestamp.clone();

        let observation = f(self);

        if let Some(ob) = observation {
            self.eskf.observe(ob);
            self.last_update_time.observe = timestamp;
        }
    }
}
