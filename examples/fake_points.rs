use itertools::Itertools;
use nalgebra::{Rotation3, Vector3, vector};
use odometries::algorithm::lio::{self, LIO, StampedImu, measurement::StampedPoints};

fn main() {
    let mut lio = LIO::new_with_gravity_factor(lio::NoGravityConfig::default(), 0.0, 1.0);

    let mut rng = rand::rng();
    use rand::Rng;

    // generate fake points stream
    let fake_points = (0..)
        .map(|t| {
            Rotation3::from_scaled_axis(
                Vector3::z() * core::f64::consts::PI * (t % 96) as f64 / 48.0,
            ) * vector![
                1.7 + rng.random::<f64>() * 0.005,
                0.0 + rng.random::<f64>() * 0.005,
                0.3 + rng.random::<f64>() * 0.01,
            ]
        })
        .chunks(96);
    let fake_points = fake_points
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| StampedPoints::new((i * 96) as f64 * 0.00025, chunk))
        .take(
            option_env!("STEPS")
                .and_then(|s| {
                    s.parse()
                        .inspect_err(|e| eprintln!("Invalid STEPS: {e}"))
                        .ok()
                })
                .unwrap_or(1000),
        )
        .map(|points| (StampedImu::zeros(points.timestamp), points));
    lio.extend(fake_points);

    let pose = lio.get_pose();
    println!("{:?}", pose.translation);
}
