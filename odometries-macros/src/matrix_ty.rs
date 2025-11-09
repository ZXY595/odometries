use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    Result, Token, Type,
    parse::{Parse, ParseStream},
};

pub struct AnyStorage {
    element: Type,
    rows: Type,
    cols: Type,
}

impl Parse for AnyStorage {
    fn parse(input: ParseStream) -> Result<Self> {
        let element = input.parse()?;
        let _: Token![,] = input.parse()?;
        let rows = input.parse()?;
        let _: Token![,] = input.parse()?;
        let cols = input.parse()?;
        Ok(Self {
            element,
            rows,
            cols,
        })
    }
}

impl ToTokens for AnyStorage {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let Self {
            element,
            rows,
            cols,
        } = self;
        tokens.extend(quote! {
            nalgebra::Matrix<#element, #rows, #cols, impl nalgebra::Storage<#element, #rows, #cols>>
        })
    }
}
