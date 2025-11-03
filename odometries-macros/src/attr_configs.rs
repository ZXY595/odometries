use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Field, Result, Token, TypeParam,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

use crate::utils::StructNamed;

pub struct Input {
    named_struct: StructNamed,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let named_struct = input.parse()?;

        Ok(Input { named_struct })
    }
}

impl ToTokens for Input {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let StructNamed {
            attrs,
            vis,
            ident,
            generics,
            fields,
        } = &self.named_struct;

        let (fields, new_types): (TokenStream2, TokenStream2) = fields
            .named
            .iter()
            .filter_map(|fields| {
                let Field {
                    attrs,
                    vis,
                    ident,
                    colon_token,
                    ty,
                    ..
                } = fields;
                let ident = ident.as_ref()?;
                let ident_span = ident.span();
                let new_type = format_ident!(
                    "{}",
                    heck::AsUpperCamelCase(ident.to_string()).to_string(),
                    span = ident_span
                );
                let mut attrs = attrs.clone();
                let generics_params = attrs
                    .extract_if(.., |attr| attr.path().is_ident("generics"))
                    .next()
                    .and_then(|attr| {
                        attr.parse_args_with(Punctuated::<TypeParam, Token![,]>::parse_terminated)
                            .ok()
                    });
                let ty_generics = generics_params.as_ref().map(|params| {
                    let idents = params.iter().map(|param| &param.ident);
                    quote! { <#(#idents),*> }
                });
                let generics_params = generics_params.map(|params| {
                    quote! { <#params> }
                });
                Some((
                    quote! {
                        #(#attrs)*
                        #vis #ident #colon_token #new_type #ty_generics,
                    },
                    quote! {
                        #vis struct #new_type #generics_params (#ty);
                    },
                ))
            })
            .unzip();
        tokens.extend(quote! {
            #(#attrs)*
            #vis struct #ident #generics {
                #fields
            }
            #new_types
        });
    }
}
