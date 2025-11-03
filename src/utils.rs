use nalgebra::{
    Cholesky, ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, ComplexField,
    DefaultAllocator, Dim, DimAdd, DimMin, DimMinimum, DimSum, Matrix, Matrix3, OMatrix, Point3,
    RawStorageMut, Scalar, U1, Vector3, VectorViewMut, ViewStorageMut, allocator::Allocator,
};
use num_traits::{One, Zero, float::FloatCore};
use simba::scalar::SupersetOf;
use std::{hash::Hash, iter::Sum};

pub trait ViewDiagonalMut {
    type Element;
    type Dim: Dim;
    type RStride: Dim;
    fn view_diagonal_mut(
        &mut self,
    ) -> VectorViewMut<'_, Self::Element, Self::Dim, Self::RStride, U1>;
}

impl<T, R, C> ViewDiagonalMut for OMatrix<T, R, C>
where
    T: Scalar,
    R: Dim + DimMin<C> + DimAdd<U1>,
    C: Dim,
    DefaultAllocator: Allocator<R, C>,
{
    type Element = T;
    type Dim = DimMinimum<R, C>;
    type RStride = DimSum<R, U1>;
    #[inline]
    fn view_diagonal_mut(
        &mut self,
    ) -> VectorViewMut<'_, Self::Element, Self::Dim, Self::RStride, U1> {
        let (rows, cols) = self.shape_generic();
        let min_dim = rows.min(cols);
        unsafe {
            let data = ViewStorageMut::from_raw_parts(
                self.data.ptr_mut(),
                (min_dim, U1),
                (rows.add(U1), U1),
            );
            Matrix::from_data_statically_unchecked(data)
        }
    }
}

/// A type that can provides a positive definite substitute value
/// for some inverse operations like [`Cholesky::new_with_substitute`]
///
/// # Safety
/// the value of [`Substitutive::SUBSTITUTE`] must return a positive definite value
pub unsafe trait Substitutive: ComplexField {
    const SUBSTITUTE: Self;

    fn non_zero_or_substitute(self) -> Self {
        if self.is_zero() {
            Self::SUBSTITUTE
        } else {
            self
        }
    }
}

/// # Safety
/// returned value is positive definite
unsafe impl Substitutive for f64 {
    const SUBSTITUTE: Self = 0.0001;
}

/// # Safety
/// returned value is positive definite
unsafe impl Substitutive for f32 {
    const SUBSTITUTE: Self = 0.0001;
}

pub(crate) trait InverseWithSubstitute {
    fn cholesky_inverse_with_substitute(self) -> Self;
}

impl<T: Substitutive, D: Dim> InverseWithSubstitute for OMatrix<T, D, D>
where
    DefaultAllocator: Allocator<D, D>,
{
    fn cholesky_inverse_with_substitute(self) -> Self {
        let cholesky = Cholesky::new_with_substitute(self, T::SUBSTITUTE);
        // Safety:
        // this is safe because the value of `T::SUBSTITUTE` is positive definite
        // and the Cholesky decomposition is always successful
        let cholesky = unsafe { cholesky.unwrap_unchecked() };
        cholesky.inverse()
    }
}

pub trait ToRadians {
    fn to_radians(self) -> Self;
}

impl<T: FloatCore> ToRadians for T {
    fn to_radians(self) -> Self {
        <T as FloatCore>::to_radians(self)
    }
}

#[derive(Default)]
pub struct VectorSquareSum<T: Scalar> {
    count: usize,
    sum: Vector3<T>,
    square_sum: Matrix3<T>,
}

impl<T> VectorSquareSum<T>
where
    T: Scalar
        + Zero
        + One
        + ClosedAddAssign
        + ClosedMulAssign
        + ClosedDivAssign
        + ClosedSubAssign
        + SupersetOf<f64>,
{
    pub fn mean(&self) -> (Vector3<T>, Matrix3<T>) {
        let count: T = nalgebra::convert(self.count as f64);
        let mean = &self.sum / count.clone();
        let covariance = &self.square_sum / count - &mean * mean.transpose();
        (mean, covariance)
    }
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<'a, T> Sum<&'a Vector3<T>> for VectorSquareSum<T>
where
    T: Scalar + Default + Zero + One + ClosedAddAssign + ClosedMulAssign,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Vector3<T>>,
    {
        iter.fold(Self::default(), |mut acc, current| {
            acc.count += 1;
            acc.sum += current;
            acc.square_sum += current * current.transpose();
            acc
        })
    }
}

pub trait ToVoxelIndex {
    type Index: nohash_hasher::IsEnabled + Eq + Hash;
    fn to_voxel_index(self) -> Self::Index;
}

impl ToVoxelIndex for Point3<f64> {
    type Index = i64;
    fn to_voxel_index(self) -> Self::Index {
        let index = self.map(|x| x.floor() as Self::Index);
        ((index.x * 73856093) ^ (index.y * 471943) ^ (index.z * 83492791)) % 10000000
    }
}

impl ToVoxelIndex for Point3<f32> {
    type Index = i32;
    fn to_voxel_index(self) -> Self::Index {
        let index = self.map(|x| x.floor() as Self::Index);
        // TODO: check if this is correct
        ((index.x * 73856093) ^ (index.y * 471943) ^ (index.z * 83492791)) % 10000000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::*;
    fn is_positive_definite(x: impl ComplexField) -> bool {
        ComplexField::try_sqrt(x).is_some()
    }

    #[test]
    fn test_substitute_positive_definite() {
        assert!(is_positive_definite(f64::SUBSTITUTE));
        assert!(is_positive_definite(f32::SUBSTITUTE));
    }

    #[test]
    fn test_diagonal_view() {
        let mut square_m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];
        square_m.view_diagonal_mut().fill(0.0);
        assert_eq!(
            square_m,
            matrix![
                0.0, 2.0, 3.0;
                4.0, 0.0, 6.0;
                7.0, 8.0, 0.0
            ]
        );

        let mut rect_m = matrix![
            1.0,  2.0,  3.0,  4.0,  5.0;
            6.0,  7.0,  8.0,  9.0,  10.0;
            11.0, 12.0, 13.0, 14.0, 15.0;
        ];
        rect_m.view_diagonal_mut().fill(0.0);
        assert_eq!(
            rect_m,
            matrix![
                0.0,  2.0,  3.0,  4.0,  5.0;
                6.0,  0.0,  8.0,  9.0,  10.0;
                11.0, 12.0, 0.0,  14.0, 15.0;
            ]
        );

        let mut rect_m_tr = matrix![
            1.0,  6.0,  11.0;
            2.0,  7.0,  12.0;
            3.0,  8.0,  13.0;
            4.0,  9.0,  14.0;
            5.0,  10.0, 15.0;
        ];
        rect_m_tr.view_diagonal_mut().fill(0.0);
        assert_eq!(
            rect_m_tr,
            matrix![
                0.0,  6.0,  11.0;
                2.0,  0.0,  12.0;
                3.0,  8.0,  0.0;
                4.0,  9.0,  14.0;
                5.0,  10.0, 15.0;
            ]
        );

        // test dyn dim matrix
        let mut rect_m_dyn = rect_m.resize(3, 5, 16.0);
        rect_m_dyn.view_diagonal_mut().fill(1.0);
        assert_eq!(
            rect_m_dyn,
            matrix![
                1.0,  2.0,  3.0,  4.0,  5.0;
                6.0,  1.0,  8.0,  9.0,  10.0;
                11.0, 12.0, 1.0,  14.0, 15.0;
            ]
        );

        let mut rect_m_tr_dyn = rect_m_tr.resize(5, 3, 16.0);
        rect_m_tr_dyn.view_diagonal_mut().fill(1.0);
        assert_eq!(
            rect_m_tr_dyn,
            matrix![
                1.0,  6.0,  11.0;
                2.0,  1.0,  12.0;
                3.0,  8.0,  1.0;
                4.0,  9.0,  14.0;
                5.0,  10.0, 15.0;
            ]
        );
    }
}
