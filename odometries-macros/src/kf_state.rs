use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::{
    Error, FieldsNamed, Generics, Ident, Result, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

use crate::utils::StructNamed;

pub struct Input {
    pub ident: Ident,
    pub generics: Generics,
    pub fields: FieldsNamed,
    pub element_ty: Type,
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
            .find(|attr| attr.path().is_ident("Element"))
            .ok_or_else(|| Error::new(Span::call_site(), "expect #[Element(T)]"))?;

        let element_ty: Type = attr_element.parse_args()?;

        Ok(Self {
            ident,
            generics,
            fields,
            element_ty,
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
        } = self;
        let fields = &fields.named;
        let fields_tys = fields.iter().map(|field| &field.ty);

        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        let struct_ty = quote! { #ident #ty_generics };

        let sub_states_impls = fields_tys
            .clone()
            .cloned()
            .map(|ty| (None, ty))
            .sub_states_impl(ident, generics);

        let field_sub_states_impls = fields.iter().flat_map(|field| {
            field
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("SubStates"))
                .and_then(|attr| {
                    attr.parse_args_with(Punctuated::<Type, Token![,]>::parse_terminated)
                        .ok()
                })
                .into_iter()
                .flatten()
                .map(|ty| (Some(field.ty.clone()), ty))
                .sub_states_impl(ident, generics)
        });

        let last_ty = fields_tys.clone().last();

        let kf_state_impl = last_ty.map(|last_ty| {
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

        tokens.extend(quote! {
            #(#sub_states_impls)*
            #(#field_sub_states_impls)*
            #kf_state_impl
        });
    }
}

trait IntoSubStateIter {
    fn sub_states_impl(
        self,
        ident_super: &Ident,
        generics: &Generics,
    ) -> impl Iterator<Item = TokenStream2>;
}

impl<I> IntoSubStateIter for I
where
    I: Iterator<Item = (Option<Type>, Type)>,
{
    fn sub_states_impl(
        self,
        ident_super: &Ident,
        generics: &Generics,
    ) -> impl Iterator<Item = TokenStream2> {
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
        self.scan(None, move |previous_ty, (super_ty, ty)| {
            let previous_ty = previous_ty.replace(ty.clone());
            let offset = previous_ty
                .map(|previous_ty| {
                    quote! {
                        SubStateEndOffset<#previous_ty, #ident_super #ty_generics>
                    }
                })
                .or(super_ty.map(|super_ty| {
                    quote! {
                        SubStateOffset<#super_ty, #ident_super #ty_generics>
                    }
                }))
                .unwrap_or(quote! { nalgebra::U0 });
            Some(quote! {
                impl #impl_generics SubStateOf<#ident_super #ty_generics> for #ty
                #where_clause
                {
                    type Offset = #offset;
                    type EndOffset = DimNameSum<Self::Offset, Self::Dim>;
                }
            })
        })
    }
}
