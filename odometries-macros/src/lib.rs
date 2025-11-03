pub(crate) mod add_assign_vector;
pub(crate) mod attr_configs;
pub(crate) mod kf_state;
pub(crate) mod utils;
use proc_macro::TokenStream;
use quote::ToTokens;

/// Derive macro for `KFState`, also implements `SubStateOf<this struct>` for every field.
#[proc_macro_derive(KFState, attributes(Element, SubStates))]
pub fn derive_kf_state(ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as kf_state::Input)
        .to_token_stream()
        .into()
}

#[proc_macro_derive(AddAssignVector, attributes(Predicates))]
pub fn derive_add_assign_vector(ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as add_assign_vector::Input)
        .to_token_stream()
        .into()
}

#[proc_macro_attribute]
pub fn configs(_attr: TokenStream, ts: TokenStream) -> TokenStream {
    syn::parse_macro_input!(ts as attr_configs::Input)
        .to_token_stream()
        .into()
}
