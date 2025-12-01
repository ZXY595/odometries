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
    frame::{BodyPoint, Framed, IsometryFramed, frames},
    utils::ToRadians,
    voxel_map::{
        self, VoxelMap,
        uncertain::{UncertainWorldPoint, body_point},
    },
};

use nalgebra::{ComplexField, Matrix3, RealField, Scalar};

pub use body_point::ProcessCov as BodyPointProcessCov;
pub use measurement::MeasureNoiseConfig;
pub use predict::ProcessCovConfig as StateProcessCovConfig;

pub use measurement::{ImuInit, ImuMeasured, ImuMeasuredStamped};

pub type PointsProcessBuffer<T> = Vec<(
    BodyPoint<T>,
    UncertainWorldPoint<T>,
    Framed<Matrix3<T>, frames::Imu>,
)>;

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
    points_process_buffer: PointsProcessBuffer<T>,
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

    /// downsample leaf size
    #[expect(unused)]
    voxel_grid_resolution: T,

    /// The size of the processing buffer used to store the temporary transformed points.
    pub buffer_init_size: usize,
}

pub struct ProcessCovConfig<T> {
    pub state: StateProcessCovConfig<T>,
    pub body_point: BodyPointProcessCov<T>,
}

impl<T> ImuInit<T>
where
    T: RealField + ToRadians,
{
    pub fn new_lio(self, lio_config: Config<T>) -> LIO<T> {
        LIO::new(lio_config, self)
    }
}

impl<T> LIO<T>
where
    T: RealField + ToRadians,
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
            measure_noise: config.measure_noise,
            gravity_factor,
            points_process_buffer: Vec::with_capacity(config.buffer_init_size),
        }
    }

    /// Create a new LIO instance with a given gravity factor.
    ///
    /// This will ignore the `gravity` in [`Config<T>`]
    pub fn new_with_gravity_factor(
        config: Config<T>,
        timestamp_init: T,
        gravity_factor: T,
    ) -> Self {
        let eskf = Eskf::new(config.process_cov.state.into(), timestamp_init);

        Self {
            eskf,
            map: VoxelMap::new(config.voxel_map),
            body_point_process_cov: config.process_cov.body_point,
            extrinsics: config.extrinsics,
            measure_noise: config.measure_noise,
            gravity_factor,
            points_process_buffer: Vec::with_capacity(config.buffer_init_size),
        }
    }

    #[inline]
    pub fn get_pose(&self) -> &IsometryFramed<T, fn(frames::Imu) -> frames::World> {
        &self.eskf.pose.0
    }
}

impl<T: RealField> Default for Config<T> {
    fn default() -> Self {
        let voxel_map_config: voxel_map::Config<T> = Default::default();
        Self {
            process_cov: Default::default(),
            measure_noise: Default::default(),
            extrinsics: Default::default(),
            voxel_grid_resolution: voxel_map_config.voxel_size.clone(),
            voxel_map: voxel_map_config,
            gravity: nalgebra::convert(9.81),
            buffer_init_size: 96,
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
