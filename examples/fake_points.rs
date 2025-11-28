use nalgebra::vector;
use odometries::algorithm::lio::{self, LIO};

fn main() {
    let mut lio = LIO::new_with_gravity_factor(lio::Config::default(), 0.0, 1.0);

    // generate fake points stream
    let mut t = 0.0;
    let fake_points = std::iter::from_fn(|| {
        t += 0.0125;
        Some((t, vector![0.0 + rand::random::<f64>() * 0.03, 0.0 + rand::random::<f64>() * 0.03, 10.0 + rand::random::<f64>() * 0.01]))
    })
    .take(100);

    fake_points.for_each(|(t, point)| {
        lio.extend_points(t, [point]);
        let pose = lio.get_pose();
        println!("{:?}", pose.translation);
    })
}
