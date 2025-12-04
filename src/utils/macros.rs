/// A macro to generate a [`nalgebra::Matrix`] type that storage type is `impl` [`nalgebra::Storage`]
///
/// # Example
/// ```rust
/// use nalgebra::{U3, U4};
/// use odometries::AnyStorageMatrix;
///
/// fn test(h: AnyStorageMatrix!(f32, U3, U4)) {
///     let _ = h.transpose();
/// }
/// ```
macro_rules! AnyStorageMatrix {
    ( $name:ty, $rows:ty, $cols:ty ) => {
        nalgebra::Matrix<$name, $rows, $cols, impl nalgebra::Storage<$name, $rows, $cols>>
    };
}
pub(crate) use AnyStorageMatrix;

macro_rules! AnyStorageVector {
    ( $name:ty, $rows:ty ) => {
        nalgebra::Vector<$name, $rows, impl nalgebra::Storage<$name, $rows, nalgebra::U1>>
    };
}
pub(crate) use AnyStorageVector;
