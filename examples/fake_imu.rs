use std::pin::pin;

use odometries::algorithm::lio::{self, ImuInit, ImuMeasured, StampedImu};
use smol::stream::StreamExt;

fn main() {
    let interval = 0.2;
    let fake_imu = smol::stream::unfold((), async |_| {
        smol::Timer::after(std::time::Duration::from_secs_f64(interval)).await;
        let next = rand::random_range(-0.01..0.01);
        Some((next, ()))
    })
    .map(|x| ImuMeasured::new(0., 0., 9.81 + x, 0., 0., 0.))
    .enumerate()
    .map(|(t, measured)| StampedImu::new(t as f64 * interval, measured));

    smol::block_on(async move {
        let mut fake_imu = pin!(fake_imu);
        let imu_init = (&mut fake_imu)
            .take(5)
            .collect::<Option<ImuInit<_>>>()
            .await
            .unwrap();

        let mut lio = imu_init.new_lio(lio::Config::default());
        fake_imu
            .for_each(move |measured| {
                lio.extend([measured]);
                dbg!(lio.get_pose().translation);
            })
            .await;
    });
}
