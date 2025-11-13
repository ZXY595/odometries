pub mod estimate;
pub mod measurement;
pub mod state;

use state::State;

use crate::{
    eskf::{
        Eskf, KFTime, StateObserver, StatePredictor,
        state::common::{AccState, AngularAccBiasState, GravityState},
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
    // TODO: refactor these config into a config struct
    body_point_process_cov: BodyPointProcessCovConfig<T>,
    extrinsics: FramedIsometry<T, fn(frames::Body) -> frames::Imu>,
    linear_acc_norm: T,
    // TODO: refactor these fields into a noise config struct
    imu_acc_measure_noise: AccState<T>,
    lidar_point_measure_noise: T,
}

pub struct Config<T: Scalar> {
    process_cov: ProcessCovConfig<T>,
    voxel_map: voxel_map::Config<T>,
    extrinsics: FramedIsometry<T, fn(frames::Body) -> frames::Imu>,
    // TODO: refactor these fields into a noise config struct
    imu_acc_measure_noise: AccState<T>,
    lidar_point_measure_noise: T,
}

pub struct ProcessCovConfig<T: Scalar> {
    state: StateProcessCovConfig<T>,
    body_point: BodyPointProcessCovConfig<T>,
}

pub struct ImuInit<T> {
    linear_acc_norm: T,
    angular_acc_bias: AngularAccBiasState<T>,
    gravity: GravityState<T>,
}

impl<T> LIO<T>
where
    T: RealField + ToRadians + Default,
{
    pub fn new(imu_init: ImuInit<T>, config: Config<T>) -> Self {
        let process_cov_config = config.process_cov;
        let mut eskf = Eskf::new(process_cov_config.state.into());

        eskf.gravity = imu_init.gravity;
        eskf.acc_with_bias.bias.angular = imu_init.angular_acc_bias;

        Self {
            eskf,
            last_update_time: Default::default(),
            map: VoxelMap::new(config.voxel_map),
            body_point_process_cov: process_cov_config.body_point,
            extrinsics: config.extrinsics,
            linear_acc_norm: imu_init.linear_acc_norm,
            imu_acc_measure_noise: config.imu_acc_measure_noise,
            lidar_point_measure_noise: config.lidar_point_measure_noise,
        }
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
