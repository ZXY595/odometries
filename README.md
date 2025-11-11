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
- [ ] `Leg-Kilo`: almost done, but still need more `api` works, tests and documentation.
- [ ] Write a `ROS` example package using [`ros2-client`](https://crates.io/crates/ros2-client) to show how to use this library.
