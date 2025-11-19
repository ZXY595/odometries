pub(crate) mod add_assign_vector;
pub(crate) mod kf_state;
pub(crate) mod sub_state_of;
pub(crate) mod unbiased;
pub(crate) mod utils;
use proc_macro::TokenStream;
use quote::ToTokens;

/// Derive macro for `KFState`, also implements `SubStateOf<this struct>` for every field.
#[proc_macro_derive(KFState, attributes(element, sub_states))]
pub fn derive_kf_state(ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as kf_state::Input)
        .to_token_stream()
        .into()
}

/// # Example
///
/// ```rust
/// struct State<T> {
///     state1: State1,
/// }
/// #[sub_state_of(State)]
/// struct State1<T: Scalar>(State2<T>, State3<T>);
/// ```
#[proc_macro_attribute]
pub fn sub_state_of(mut arg: TokenStream, ts: TokenStream) -> TokenStream {
    arg.extend(ts);
    let ts = arg;
    syn::parse_macro_input!(ts as sub_state_of::Input)
        .to_token_stream()
        .into()
}

/// Derive macro for `AddAssign<Vector<Self::Element, Self::Dim, impl Storage>>`
#[proc_macro_derive(VectorAddAssign, attributes(vector_add_assign))]
pub fn derive_add_assign_vector(ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as add_assign_vector::Input)
        .to_token_stream()
        .into()
}

#[proc_macro_derive(Unbiased)]
pub fn derive_unbiased(ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as unbiased::Input)
        .to_token_stream()
        .into()
}
