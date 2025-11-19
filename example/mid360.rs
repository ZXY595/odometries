use std::pin::pin;

use livox2::{
    lidar_port::{IpConfig, point_data::CoordinateDataRef},
    types::ethernet::{CartesianHighPoint, ImuData as LivoxImuData},
};
use odometries::algorithm::lio::{self, ImuInit, ImuMeasured, ImuMeasuredStamped};
use smol::stream::{self, StreamExt};

fn main() -> std::io::Result<()> {
    let lidar_ip = IpConfig::new([192, 168, 1, 10], [192, 168, 1, 190]);
    smol::block_on(async {
        let mut point_cloud = lidar_ip.new_default_point_data_port().await?;

        let lio_config = lio::Config::default();
        let gravity = lio_config.gravity;

        let imu_stream = lidar_ip
            .new_default_imu_port()
            .await?
            .into_stream(|packet| {
                ImuMeasuredStamped::new(
                    packet.header.timestamp as f64 / 1e9,
                    livox_imu_to_mesurement(packet.data, gravity),
                )
            });

        let mut imu_stream = pin!(imu_stream);
        let imu_init = (&mut imu_stream)
            .take(200)
            .collect::<Option<ImuInit<_>>>()
            .await
            .unwrap();

        dbg!(&imu_init);

        let mut lio = imu_init.new_lio(lio_config);

        loop {
            let point_cloud = point_cloud.next_packet_ref().await?;
            if let CoordinateDataRef::CartesianHigh(points) = point_cloud.data {
                let point_start_timestamp = point_cloud.header.timestamp_sec();
                let point_end_timestamp = point_cloud.header.end_timestamp_sec();

                let imu_measurments = stream::block_on(
                    imu_stream
                        .drain()
                        .skip_while(|measurment| measurment.timestamp < point_start_timestamp)
                        .take_while(|measurment| measurment.timestamp < point_end_timestamp),
                );
                lio.extend(imu_measurments);

                lio.extend_points(
                    point_start_timestamp,
                    points.iter().map(livox_point_to_mesurement),
                );
                dbg!(lio.get_pose().translation);
            }
        }
    })
}

fn livox_imu_to_mesurement(
    &LivoxImuData {
        gyro_x,
        gyro_y,
        gyro_z,
        acc_x,
        acc_y,
        acc_z,
    }: &LivoxImuData,
    gravity: f64,
) -> ImuMeasured<f64> {
    ImuMeasured::new(
        acc_x as f64 * gravity,
        acc_y as f64 * gravity,
        acc_z as f64 * gravity,
        gyro_x as f64,
        gyro_y as f64,
        gyro_z as f64,
    )
}

fn livox_point_to_mesurement(&CartesianHighPoint { x, y, z, .. }: &CartesianHighPoint) -> [f64; 3] {
    [x as f64, y as f64, z as f64].map(|x| x * 1e-3)
}
