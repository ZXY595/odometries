//! An inertial–LiDAR tightly‑coupled error‑state Kalman filter odometry system.
//! Most of the ideas come from Leg-Kilo

mod estimate;
mod state;

use state::State;

use crate::{
    eskf::Eskf,
    utils::ToVoxelIndex,
    voxel_map::{self, BoxedVoxelMap, VoxelMap},
};

use nalgebra::{Point3, Scalar, SimdRealField};

pub use state::ProcessCovConfig;

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
pub struct ILO<T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex,
{
    eskf: Eskf<State<T>>,
    map: BoxedVoxelMap<T>,
}

pub struct Config<T: Scalar> {
    process_cov: ProcessCovConfig<T>,
    voxel_map: voxel_map::Config<T>,
}

impl<T> ILO<T>
where
    T: Scalar + Default + SimdRealField<Element: SimdRealField>,
    Point3<T>: ToVoxelIndex,
{
    pub fn new(config: Config<T>) -> Self {
        let eskf = Eskf::new(config.process_cov.into());
        Self {
            eskf,
            map: VoxelMap::new_boxed(config.voxel_map),
        }
    }
    // TODO: add lidar points and imu measurements to the filter
}
