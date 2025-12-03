use crate::{
    frame::{Framed, IsometryFramed, frames},
    voxel_map,
};

use super::measurement::MeasureNoiseConfig;
pub use super::predict::ProcessCovConfig as StateProcessCovConfig;
pub use crate::voxel_map::uncertain::body_point::ProcessCov as BodyPointProcessCov;

use nalgebra::{IsometryMatrix3, RealField, Scalar, Translation3};
use simba::scalar::SupersetOf;

/// The configuration of the LIO algorithm with no need to provide the gravity.
pub type NoGravityConfig<T> = Config<T, NoGravity>;
pub struct NoGravity;

pub struct Config<T: Scalar, G = T> {
    /// The process noise configuration of the state.
    pub process_cov: ProcessCovConfig<T>,

    /// The measurement noise configuration of the IMU and the lidar point(body frame).
    pub measure_noise: MeasureNoiseConfig<T>,

    /// The extrinsics of the IMU to the body frame.
    pub extrinsics: IsometryFramed<T, fn(frames::Body) -> frames::Imu>,

    /// The gravity norm. Used to calculate the gravity factor (also known as gravity compensation)
    ///
    /// Note that this is optional, you can provide the gravity factor directly,
    /// see also [`LIO::new_with_gravity_factor`](super::LIO::new_with_gravity_factor).
    pub gravity: G,

    /// The voxel map configuration.
    pub voxel_map: voxel_map::Config<T>,

    /// downsample leaf size
    pub downsample_resolution: T,

    /// The size of the processing buffer used to store the temporary transformed points.
    pub buffer_init_size: usize,
}

pub struct ProcessCovConfig<T> {
    pub state: StateProcessCovConfig<T>,
    pub body_point: BodyPointProcessCov<T>,
}

impl<T: RealField> Default for Config<T> {
    fn default() -> Self {
        let voxel_map_config: voxel_map::Config<T> = Default::default();
        Self {
            process_cov: Default::default(),
            measure_noise: Default::default(),
            extrinsics: Default::default(),
            downsample_resolution: voxel_map_config.voxel_size.clone(),
            voxel_map: voxel_map_config,
            gravity: nalgebra::convert(9.81),
            buffer_init_size: 80,
        }
    }
}

impl<T: RealField> Default for NoGravityConfig<T> {
    #[inline]
    fn default() -> Self {
        Config::<T>::default().take_gravity().1
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

impl<T: Scalar, G> Config<T, G> {
    pub fn take_gravity(self) -> (G, NoGravityConfig<T>) {
        let Self {
            process_cov,
            measure_noise,
            extrinsics,
            gravity,
            voxel_map,
            downsample_resolution,
            buffer_init_size,
        } = self;
        (
            gravity,
            NoGravityConfig {
                gravity: NoGravity,
                process_cov,
                measure_noise,
                extrinsics,
                voxel_map,
                downsample_resolution,
                buffer_init_size,
            },
        )
    }
}

impl<T: RealField> Config<T> {
    #[inline]
    pub fn with_mid360_extrinsics(self) -> Self {
        let extrinsics = IsometryMatrix3::from_parts(
            Translation3::new(-0.011, -0.02329, 0.04412).cast(),
            Default::default(),
        );
        Self {
            extrinsics: Framed::new(extrinsics),
            ..self
        }
    }
}
