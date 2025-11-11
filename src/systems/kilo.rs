//! An inertial–LiDAR tightly‑coupled error‑state Kalman filter odometry system.
//! Most of the ideas come from Leg-Kilo

mod estimate;
mod state;

use std::ops::Deref;

use state::State;

use crate::{
    eskf::{
        DeltaTime, Eskf, StateObserver, StatePredictor,
        state::common::{AccState, AccWithBiasState},
    },
    frame::{BodyPoint, Framed, FramedIsometry, frames},
    systems::kilo::estimate::{ImuObserved, PointsObserved},
    utils::{ToRadians, ToVoxelIndex},
    voxel_map::{
        self, VoxelMap,
        uncertain::{UncertainBodyPoint, UncertainWorldPoint, body_point},
    },
};

use nalgebra::{Point3, RealField, Scalar, stack};

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
pub struct ILO<T>
where
    T: Scalar,
    Point3<T>: ToVoxelIndex<T>,
{
    eskf: Eskf<State<T>>,
    map: VoxelMap<T>,
    body_point_process_cov: BodyPointProcessCovConfig<T>,
    extrinsics: FramedIsometry<T, fn(frames::Body) -> frames::Imu>,
    last_predict_time: T,
    last_observe_time: T,
}

pub struct Config<T: Scalar> {
    process_cov: ProcessCovConfig<T>,
    voxel_map: voxel_map::Config<T>,
    extrinsics: FramedIsometry<T, fn(frames::Body) -> frames::Imu>,
}

pub struct ProcessCovConfig<T: Scalar> {
    state: StateProcessCovConfig<T>,
    body_point: BodyPointProcessCovConfig<T>,
}

pub type ImuMeasurement<T> = AccState<T>;

impl<T> ILO<T>
where
    T: RealField + ToRadians + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    pub fn new(config: Config<T>) -> Self {
        let process_cov_config = config.process_cov;
        let eskf = Eskf::new(process_cov_config.state.into());
        Self {
            eskf,
            map: VoxelMap::new(config.voxel_map),
            body_point_process_cov: process_cov_config.body_point,
            extrinsics: config.extrinsics,
            last_predict_time: T::default(),
            last_observe_time: T::default(),
        }
    }
    fn eskf_update<OB>(&mut self, timestamp: T, f: impl FnOnce(&mut Self) -> Option<OB>)
    where
        Eskf<State<T>>: StateObserver<OB>,
    {
        self.eskf.predict(DeltaTime {
            predict: timestamp.clone() - self.last_predict_time.clone(),
            observe: timestamp.clone() - self.last_observe_time.clone(),
        });
        self.last_predict_time = timestamp.clone();

        let observation = f(self);

        if let Some(ob) = observation {
            self.eskf.observe(ob);
            self.last_observe_time = timestamp;
        }
    }
    pub fn extend_points(
        &mut self,
        points_with_timestamp: (impl IntoIterator<Item = BodyPoint<T>, IntoIter: Clone>, T),
    ) {
        let (points, timestamp) = points_with_timestamp;
        let points = points.into_iter();

        self.eskf_update(timestamp, |ilo| {
            let body_to_imu = &ilo.extrinsics;
            let imu_to_world = ilo.eskf.pose.deref();
            let body_to_world = body_to_imu * imu_to_world;

            // TODO: could this be optimized by using `rayon`?
            let observation = points
                .clone()
                .filter_map(|point| {
                    let config = ilo.body_point_process_cov.clone();

                    let body_point = UncertainBodyPoint::<T>::from_body_point(point, config);

                    let imu_point = body_point.deref() * body_to_imu;
                    let cross_matrix_imu = imu_point.coords.cross_matrix();

                    let world_point = UncertainWorldPoint::from_uncertain_body_point(
                        body_point,
                        imu_to_world,
                        &body_to_world,
                        Framed::new(&cross_matrix_imu),
                        &ilo.eskf.cov,
                    );
                    let residual = ilo.map.get_residual(world_point).or_else(|| {
                        // TODO: search for one nearest voxel in the map
                        todo!()
                    })?;
                    let plane_normal = residual.plane_normal;
                    let crossmatrix_rotation_t_normal =
                        cross_matrix_imu * ilo.eskf.pose.rotation.transpose() * plane_normal;

                    #[expect(clippy::toplevel_ref_arg)]
                    let model = stack![crossmatrix_rotation_t_normal; plane_normal];
                    // TODO: do we really need to [`neg`](std::ops::Neg) this measurement?
                    let measurement = -residual.distance_to_plane;
                    // TODO: add a config to adjust the noise
                    let noise = residual.sigma;

                    Some((measurement, model, noise))
                })
                .collect::<PointsObserved<T>>();

            if observation.get_dim().0 == 0 {
                return None;
            }
            Some(observation)
        });

        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        let new_world_points = points.map(|point| {
            let config = self.body_point_process_cov.clone();
            UncertainWorldPoint::from_body_point(
                point,
                config,
                body_to_imu,
                imu_to_world,
                &body_to_world,
                &self.eskf.cov,
            )
        });
        self.map.extend(new_world_points);
    }
}

impl<T> FromIterator<ImuMeasurement<T>> for ILO<T>
where
    T: RealField + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ImuMeasurement<T>>,
    {
        let iter = iter.into_iter();
        todo!()
    }
}

impl<T> Extend<(ImuMeasurement<T>, T)> for ILO<T>
where
    T: RealField + ToRadians + Default,
    Point3<T>: ToVoxelIndex<T>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (ImuMeasurement<T>, T)>,
    {
        iter.into_iter().for_each(|(acc, timestamp)| {
            let linear_acc = acc.linear.deref();
            let angular_acc = acc.angular.deref();
            self.eskf_update(timestamp, |ilo| {
                let gravity: T = nalgebra::convert(9.81);
                let AccWithBiasState {
                    acc: state_acc,
                    bias: state_acc_bias,
                } = &ilo.eskf.state.acc_with_bias;
                let linear_measured_acc=
                    linear_acc * gravity /* TODO: div acc_norm here */ - state_acc.linear.deref() - state_acc_bias.linear.deref();
                let angular_measured_acc =  angular_acc - state_acc.angular.deref() - state_acc_bias.angular.deref();

                #[expect(clippy::toplevel_ref_arg)]
                let measurement = stack![linear_measured_acc; angular_measured_acc];
                // TODO: add a config to adjust the noise.
                let noise = nalgebra::Vector6::zeros();
                Some(ImuObserved::new_no_model(measurement, noise))
            });
        })
    }
}

impl<P, T> Extend<(P, T)> for ILO<T>
where
    T: RealField + ToRadians + Default,
    Point3<T>: ToVoxelIndex<T>,
    P: IntoIterator<Item = BodyPoint<T>, IntoIter: Clone>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (P, T)>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter().for_each(|points_with_timestamp| {
            self.extend_points(points_with_timestamp);
        });
    }
}
