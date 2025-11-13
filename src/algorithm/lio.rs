pub mod estimate;
pub mod measurement;
pub mod state;

use std::ops::Deref;

use state::State;

use crate::{
    algorithm::lio::{
        estimate::{ImuObserved, PointsObserved},
        measurement::{ImuMeasured, ImuMeasuredStamped, LidarPoint, PointChunkStamped},
    },
    eskf::{
        Eskf, KFTime, StateObserver, StatePredictor,
        state::common::{AccState, AccWithBiasState, AngularAccBiasState, GravityState},
    },
    frame::{BodyPoint, Framed, FramedIsometry, frames},
    utils::ToRadians,
    voxel_map::{
        self, VoxelMap,
        uncertain::{UncertainBodyPoint, UncertainWorldPoint, body_point},
    },
};

use nalgebra::{ComplexField, RealField, Scalar, stack};

pub use body_point::ProcessCovConfig as BodyPointProcessCovConfig;
pub use state::ProcessCovConfig as StateProcessCovConfig;

pub const GRAVITY: f64 = 9.81;

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

    pub fn extend_measurements<'a, P: LidarPoint<T> + 'a>(
        &mut self,
        points: impl IntoIterator<Item = PointChunkStamped<'a, T, P>>,
        imus: impl IntoIterator<Item = ImuMeasuredStamped<'a, T>>,
    ) {
        let points = points.into_iter();
        let mut imus = imus.into_iter();

        // TODO: could this be optimized by using `rayon`?
        points.for_each(|(points_time, point_chunk)| {
            let imus_before_points = imus
                .by_ref()
                .take_while(|(imu_time, _)| imu_time < &points_time);
            self.extend(imus_before_points);
            self.extend_point_chunk(points_time, point_chunk);
        });
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

    /// The reason why `&'a [impl LidarPoint<T>]` instead of `impl IntoIterator<Item = LidarPoint<T>>`
    /// is that we need to iterate the point chunk twice, once for observation and once for updating the map.
    pub fn extend_point_chunk(&mut self, timestamp: T, points: &[impl LidarPoint<T>]) {
        self.eskf_update(timestamp, |ilo| {
            ilo.observe_points(points.iter().cloned().map(LidarPoint::to_body_point))
        });

        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        let new_world_points = points
            .iter()
            .cloned()
            .map(LidarPoint::to_body_point)
            .map(|point| {
                UncertainWorldPoint::from_body_point(
                    point,
                    self.body_point_process_cov.clone(),
                    body_to_imu,
                    imu_to_world,
                    &body_to_world,
                    &self.eskf.cov,
                )
            });
        self.map.extend(new_world_points);
    }

    fn observe_points(
        &self,
        points: impl Iterator<Item = BodyPoint<T>>,
    ) -> Option<PointsObserved<T>> {
        let body_to_imu = &self.extrinsics;
        let imu_to_world = self.eskf.pose.deref();
        let body_to_world = body_to_imu * imu_to_world;

        // TODO: could this be optimized by using `rayon`?
        let observation = points
            .filter_map(|point| {
                let body_point = UncertainBodyPoint::<T>::from_body_point(
                    point,
                    self.body_point_process_cov.clone(),
                );

                let imu_point = body_point.deref() * body_to_imu;
                let cross_matrix_imu = imu_point.coords.cross_matrix();

                let world_point = UncertainWorldPoint::from_uncertain_body_point(
                    body_point,
                    imu_to_world,
                    &body_to_world,
                    Framed::new(&cross_matrix_imu),
                    &self.eskf.cov,
                );
                let residual = self.map.get_residual(world_point).or_else(|| {
                    // TODO: search for one nearest voxel in the map
                    todo!()
                })?;
                let plane_normal = residual.plane_normal;
                let crossmatrix_rotation_t_normal =
                    cross_matrix_imu * self.eskf.pose.rotation.transpose() * plane_normal;

                #[expect(clippy::toplevel_ref_arg)]
                let model = stack![crossmatrix_rotation_t_normal; plane_normal];

                // TODO: do we really need to [`neg`](std::ops::Neg) this measurement?
                let measurement = -residual.distance_to_plane;

                let noise = self.lidar_point_measure_noise.clone() * residual.sigma;

                Some((measurement, model, noise))
            })
            .collect::<PointsObserved<T>>();

        if observation.get_dim().0 == 0 {
            return None;
        }
        Some(observation)
    }

    fn observe_imu(&self, imu_acc: &ImuMeasured<T>) -> ImuObserved<T> {
        let linear_acc = imu_acc.linear.deref();
        let angular_acc = imu_acc.angular.deref();
        let gravity: T = nalgebra::convert(GRAVITY);

        let AccWithBiasState {
            acc: state_acc,
            bias: state_acc_bias,
        } = &self.eskf.state.acc_with_bias;

        let linear_measured_acc = linear_acc * (gravity / self.linear_acc_norm.clone())
            - state_acc.linear.deref()
            - state_acc_bias.linear.deref();

        let angular_measured_acc =
            angular_acc - state_acc.angular.deref() - state_acc_bias.angular.deref();

        #[expect(clippy::toplevel_ref_arg)]
        let measurement = stack![linear_measured_acc; angular_measured_acc];

        let noise = &self.imu_acc_measure_noise;
        #[expect(clippy::toplevel_ref_arg)]
        let noise = stack![noise.linear; noise.angular];

        ImuObserved::new_no_model(measurement, noise)
    }
}

pub struct ImuInit<T> {
    linear_acc_norm: T,
    angular_acc_bias: AngularAccBiasState<T>,
    gravity: GravityState<T>,
}

impl<T> FromIterator<ImuMeasured<T>> for Option<ImuInit<T>>
where
    T: RealField + Default,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ImuMeasured<T>>,
    {
        let mut iter = iter.into_iter().enumerate();

        let (_, first) = iter.next()?;
        let acc_mean = iter.fold(first, |mut mean_acc, (i, current_acc)| {
            let n: T = nalgebra::convert(i as f64);
            mean_acc.linear += (current_acc.linear.deref() - mean_acc.linear.deref()) / n.clone();
            mean_acc.angular += (current_acc.angular.deref() - mean_acc.angular.deref()) / n;
            mean_acc
        });

        let linear_acc_norm = acc_mean.linear.norm();
        let gravity: T = nalgebra::convert(GRAVITY);

        Some(ImuInit {
            gravity: GravityState::new(
                -acc_mean.linear.deref() / linear_acc_norm.clone() * gravity,
            ),
            linear_acc_norm,
            angular_acc_bias: acc_mean.angular.map_state_marker(),
        })
    }
}

impl<'a, T> Extend<ImuMeasuredStamped<'a, T>> for LIO<T>
where
    T: RealField + ToRadians + Default,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ImuMeasuredStamped<'a, T>>,
    {
        iter.into_iter().for_each(|(timestamp, acc)| {
            self.eskf_update(timestamp, |ilo| Some(ilo.observe_imu(acc)));
        })
    }
}

impl<'a, P, T> Extend<PointChunkStamped<'a, T, P>> for LIO<T>
where
    T: RealField + ToRadians + Default,
    P: LidarPoint<T>,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (T, &'a [P])>,
    {
        // TODO: could this be optimized by using `rayon`?
        iter.into_iter().for_each(|(timestamp, points)| {
            self.extend_point_chunk(timestamp, points);
        });
    }
}
