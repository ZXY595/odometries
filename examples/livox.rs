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
        let point_clouds = lidar_ip.new_default_point_data_port().await?;

        let imu_stream = lidar_ip
            .new_default_imu_port()
            .await?
            .into_stream(|packet| {
                ImuMeasuredStamped::new(
                    packet.header.timestamp as f64 / 1e9,
                    livox_imu_to_mesurement(packet.data),
                )
            });

        let mut imu_stream = pin!(imu_stream);
        let imu_init = (&mut imu_stream)
            .take(200)
            .collect::<Option<ImuInit<_>>>()
            .await
            .unwrap();

        println!("init imu with 200 samples: \n{imu_init:?}");

        let mut lio = imu_init.new_lio(lio::Config::default());

        point_clouds
            .into_stream(|point_cloud| {
                let CoordinateDataRef::CartesianHigh(points) = point_cloud.data else {
                    return;
                };
                let point_start_timestamp = point_cloud.header.timestamp_sec();
                let point_end_timestamp = point_cloud.header.end_timestamp_sec();

                let imu_measurments = stream::block_on(
                    imu_stream
                        .drain()
                        .skip_while(|measurment| measurment.timestamp < point_start_timestamp)
                        .take_while(|measurment| measurment.timestamp < point_end_timestamp),
                );
                lio.extend_point_cloud_with_imu(
                    imu_measurments,
                    (
                        point_end_timestamp,
                        points.iter().map(livox_point_to_mesurement),
                    ),
                );
            })
            .for_each(drop)
            .await;
        Ok(())
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
) -> ImuMeasured<f64> {
    ImuMeasured::new(
        acc_x as f64,
        acc_y as f64,
        acc_z as f64,
        gyro_x as f64,
        gyro_y as f64,
        gyro_z as f64,
    )
}

fn livox_point_to_mesurement(&CartesianHighPoint { x, y, z, .. }: &CartesianHighPoint) -> [f64; 3] {
    [x, y, z].map(|x| x as f64 * 1e-3)
}
