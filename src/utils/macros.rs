/// A macro to generate a [`nalgebra::Matrix`] type that storage type is `impl` [`nalgebra::Storage`]
///
/// # Example
/// ```rust
/// use nalgebra::{U3, U4};
/// fn test(h: AnyStorageMatrix!(f32, U3, U4)) {
///     let _ = h.transpose();
/// }
/// ```
#[macro_export]
macro_rules! AnyStorageMatrix {
    ( $name:ty, $rows:ty, $cols:ty ) => {
        nalgebra::Matrix<$name, $rows, $cols, impl nalgebra::Storage<$name, $rows, $cols>>
    };
}
