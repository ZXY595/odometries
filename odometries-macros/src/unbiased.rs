use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    Generics, Ident, Result,
    parse::{Parse, ParseStream},
    parse_quote,
};

use crate::utils::StructNamed;

pub struct Input {
    ident: Ident,
    generics: Generics,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let StructNamed {
            ident,
            mut generics,
            fields,
            ..
        } = input.parse()?;
        let predicates = &mut generics.make_where_clause().predicates;
        fields.named.into_iter().for_each(|field| {
            let ty = &field.ty;
            predicates.push(parse_quote! {
                 #ty: Unbiased
            });
        });
        Ok(Self { ident, generics })
    }
}

impl ToTokens for Input {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let name = &self.ident;
        let (impl_generics, ty_generics, where_clause) = self.generics.split_for_impl();
        tokens.extend(quote! {
            impl #impl_generics Unbiased for #name #ty_generics #where_clause {}
        });
    }
}
