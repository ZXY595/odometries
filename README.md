# Odometries

`Odometries` is a pure Rust library for odometry estimation and mapping.

> [!WARNING]
>
> This library is still under deep development, and the API may have breaking change at any time.
> If you also want to contribute to this project, welcome to open an issue or PR :)

# Feautres
- Zero `ROS` dependency, so it is easy to use and deploy on different platforms.
- `Most Generic` interface for different odometry algorithms.
- `Keep it Simple`: This library only provides the core algorithms,
    you might need to manage the data flows like `pointcloud` stream, and use `odometries` to process them to get pose state.

# Current status
- [x] common wrappers and utils like `Framed` and `Uncertained`.
- [x] The most generic `ESKF` framework, including `State` and `Measurement` types.
- [x] Generic `Voxelmap` implementation with `plane` and `uncertain` intergration, based on `slab`.
- [x] `LIO`: tightly‑coupled lidar imu odometry with `Voxelmap` map storage.
- [x] Some examples to test the odometry algorithms.
- [-] `Leg-Kilo`: almost done, which benefit from `LIO` module. But `kinematic` observation works is needed.
- [ ] `Fast-LIO2`: need `KD-Tree` implementation.
- [ ] `Fast-LIVO2`: might reuse `LIO` module for lidar and imu observation, but `vision` implementation is needed.
The crate [`kornia`](https://github.com/kornia/kornia-rs) or the crate [`image`](https://github.com/image-rs/image) with [`imageproc`](https://github.com/image-rs/imageproc.git) could help.
- [ ] Write a `ROS` example package using [`ros2-client`](https://crates.io/crates/ros2-client) to show how to use this library.
