use nalgebra::{DimName, Scalar};

pub mod common;
pub mod macro_support {
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
pub trait SubStateOf<Super>: KFState {
    type Offset: DimName;
    type EndOffset: DimName;
}

/// The `Offset` of the `SubState` in the `Super`.
/// See also [`SubStateOf`].
pub type SubStateOffset<S, Super> = <S as SubStateOf<Super>>::Offset;

/// The `Offset` of the end of the `SubState` in the `Super`.
/// See also [`SubStateOf`].
pub type SubStateEndOffset<S, Super> = <S as SubStateOf<Super>>::EndOffset;

// pub trait StateMarker {
//     type State<T: Scalar>: KFState<Element = T>;
// }

// pub type MarkerState<M, T> = <M as StateMarker>::State<T>;
// pub type MarkedSubStateOffset<M, S> =
//     <MarkerState<M, <S as ErrorState>::Element> as SubStateOf<S>>::Offset;
