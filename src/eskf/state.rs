use nalgebra::{DimName, DimNameSum, Scalar};

pub mod common;
pub mod sensitivity;
pub mod macro_export {
    pub use super::sensitivity::Unbiased;
    pub use super::{KFState, SubStateEndOffset, SubStateOf, SubStateOffset};
    pub use nalgebra::{DimName, DimNameSum};
}

pub trait KFState {
    type Element: Scalar;
    type Dim: DimName;
}

pub type StateDim<S> = <S as KFState>::Dim;

/// # Overview
/// ```text
/// ├────┬─ Superstate ─┬────┤
///      ├── Substate ──┤
///      │              │
///      Offset         EndOffset
/// ├────╯              │
/// ├───────────────────╯
/// ```
pub trait SubStateOf<Super: KFState>: KFState {
    type Offset: DimName;
}

/// The `Offset` of the `SubState` in the `Super`.
/// See also [`SubStateOf`].
pub type SubStateOffset<S, Super> = <S as SubStateOf<Super>>::Offset;

/// The `Offset` of the end of the `SubState` in the `Super`.
/// See also [`SubStateOf`].
pub type SubStateEndOffset<S, Super> = DimNameSum<<S as SubStateOf<Super>>::Offset, StateDim<S>>;
