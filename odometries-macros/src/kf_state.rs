use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    FieldsNamed, Generics, Ident, Result, Type,
    parse::{Parse, ParseStream},
};

use crate::{sub_state_of::IntoSubStateIter, utils::StructNamed};

pub struct Input {
    pub ident: Ident,
    pub generics: Generics,
    pub fields: FieldsNamed,
    pub element_ty: Type,
    pub last_ty: Option<Type>,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let StructNamed {
            attrs,
            ident,
            generics,
            fields,
            ..
        } = input.parse()?;

        let attr_element = attrs
            .iter()
            .find(|attr| attr.path().is_ident("element"))
            .ok_or_else(|| input.error("expect #[element(T)]"))?;

        let element_ty = attr_element.parse_args::<Type>()?;
        let last_ty = fields.named.last().map(|field| &field.ty).cloned();

        Ok(Self {
            ident,
            generics,
            fields,
            element_ty,
            last_ty,
        })
    }
}

impl ToTokens for Input {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let Self {
            ident,
            generics,
            fields,
            element_ty,
            last_ty,
        } = self;
        let fields = &fields.named;
        let fields_tys = fields.iter().map(|field| &field.ty);

        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let struct_ty = quote! { #ident #ty_generics };

        let sub_states_impls = fields_tys
            .clone()
            .map(|ty| (None, ty))
            .sub_states_impl(ident, generics);

        let kf_state_impl = last_ty.as_ref().map(|last_ty| {
            let dim_ty = quote! {
                SubStateEndOffset<#last_ty, #struct_ty>
            };

            quote! {
                impl #impl_generics KFState for #struct_ty
                #where_clause
                {
                    type Element = #element_ty;
                    type Dim = #dim_ty;
                }

            }
        });
        let sub_state_impl = quote! {
            impl #impl_generics SubStateOf<Self> for #struct_ty
            #where_clause
            {
                type Offset = nalgebra::U0;
                type EndOffset = Self::Dim;
            }
        };

        tokens.extend(quote! {
            #(#sub_states_impls)*
            #kf_state_impl
            #sub_state_impl
        });
    }
}
