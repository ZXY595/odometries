//! Most of ideas comes from Leg-Kilo

pub mod measurement;
pub mod predict;
pub mod state;

use std::ops::Deref;

use simba::scalar::SupersetOf;
use state::State;

use crate::{
    eskf::{
        Eskf,
        state::common::{GravityState, LinearAccState},
    },
    frame::{IsometryFramed, frames},
    utils::ToRadians,
    voxel_map::{self, VoxelMap, uncertain::body_point},
};

use nalgebra::{ComplexField, RealField, Scalar};

pub use body_point::ProcessCov as BodyPointProcessCov;
pub use measurement::MeasureNoiseConfig;
pub use predict::ProcessCovConfig as StateProcessCovConfig;

pub use measurement::{ImuInit, ImuMeasured, ImuMeasuredStamped};

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
    map: VoxelMap<T>,
    // configs
    body_point_process_cov: BodyPointProcessCov<T>,
    extrinsics: IsometryFramed<T, fn(frames::Body) -> frames::Imu>,
    measure_noise: MeasureNoiseConfig<T>,
    gravity_factor: T,
}

pub struct Config<T: Scalar> {
    pub process_cov: ProcessCovConfig<T>,
    pub measure_noise: MeasureNoiseConfig<T>,
    pub extrinsics: IsometryFramed<T, fn(frames::Body) -> frames::Imu>,
    pub gravity: T,
    pub voxel_map: voxel_map::Config<T>,
}

pub struct ProcessCovConfig<T> {
    pub state: StateProcessCovConfig<T>,
    pub body_point: BodyPointProcessCov<T>,
}

impl<T> ImuInit<T>
where
    T: RealField + ToRadians
{
    pub fn new_lio(self, lio_config: Config<T>) -> LIO<T> {
        LIO::new(lio_config, self)
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians
{
    pub fn new(config: Config<T>, imu_init: ImuInit<T>) -> Self {
        let mut eskf = Eskf::new(config.process_cov.state.into(), imu_init.timestamp_init);

        let gravity_factor = config.gravity / imu_init.linear_acc_norm.clone();

        let gravity = imu_init.linear_acc_mean.deref() * gravity_factor.clone();

        eskf.acc_with_bias.acc.linear = LinearAccState::new(gravity.clone());
        eskf.gravity = GravityState::new(-gravity);
        eskf.acc_with_bias.bias.angular = imu_init.angular_acc_bias;

        Self {
            eskf,
            map: VoxelMap::new(config.voxel_map),
            body_point_process_cov: config.process_cov.body_point,
            extrinsics: config.extrinsics,
            gravity_factor,
            measure_noise: config.measure_noise,
        }
    }

    #[inline]
    pub fn get_pose(&self) -> &IsometryFramed<T, fn(frames::Imu) -> frames::World> {
        &self.eskf.pose.0
    }
}

impl<T: RealField> Default for Config<T> {
    fn default() -> Self {
        Self {
            process_cov: Default::default(),
            measure_noise: Default::default(),
            voxel_map: Default::default(),
            extrinsics: Default::default(),
            gravity: nalgebra::convert(9.81_f64),
        }
    }
}

impl<T: SupersetOf<f64>> Default for ProcessCovConfig<T> {
    fn default() -> Self {
        Self {
            state: Default::default(),
            body_point: Default::default(),
        }
    }
}
