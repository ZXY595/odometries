use nalgebra::{Rotation3, Vector3};
use odometries::algorithm::lio::{self, LIO};

fn main() {
    let mut lio = LIO::new_with_gravity_factor(lio::Config::default(), 0.0, 1.0);

    // let rec = rerun::RecordingStreamBuilder::new("fake_points").connect_grpc();
    // rerun::RecordingStream::set_global(rerun::StoreKind::Recording, rec.ok());

    // generate fake points stream
    let fake_points = (0..)
        .map(|t| {
            (
                t as f64 * 0.0025,
                Rotation3::from_scaled_axis(
                    Vector3::z() * core::f64::consts::PI * (2 * t % 180) as f64 / 90.0,
                ) * Vector3::new(
                    2.0 + rand::random::<f64>() * 0.01,
                    0.0 + rand::random::<f64>() * 0.01,
                    0.0 + rand::random::<f64>() * 0.01,
                ),
            )
        })
        .take(2000);

    fake_points.for_each(|(t, point)| {
        lio.extend_points(t, [point]);
        let pose = lio.get_pose();
        println!("{:?}", pose.translation);
    })
}
