//! An inertial–LiDAR tightly‑coupled error‑state Kalman filter odometry system.
//! Most of the ideas come from Leg-Kilo

mod estimate;
mod state;

use state::State;

use crate::{
    eskf::Eskf,
    frame::BodyPoint,
    utils::ToVoxelIndex,
    voxel_map::{self, VoxelMap},
};

use nalgebra::{Point3, RealField, Scalar};

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
    Point3<T>: ToVoxelIndex<T>,
{
    eskf: Eskf<State<T>>,
    map: VoxelMap<T>,
}

pub struct Config<T: Scalar> {
    process_cov: ProcessCovConfig<T>,
    voxel_map: voxel_map::Config<T>,
}

impl<T> ILO<T>
where
    T: RealField + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    pub fn new(config: Config<T>) -> Self {
        let eskf = Eskf::new(config.process_cov.into());
        Self {
            eskf,
            map: VoxelMap::new(config.voxel_map),
        }
    }
    // TODO: add lidar points and imu measurements to the filter
}

impl<T> Extend<BodyPoint<T>> for ILO<T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex<T>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = BodyPoint<T>>,
    {
        todo!()
    }
}
