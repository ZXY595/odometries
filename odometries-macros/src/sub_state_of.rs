use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    Field, Fields, Generics, Ident, ItemStruct, Result, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

pub struct Input {
    pub super_ty: Ident,
    pub offset_ty: Ident,
    pub fields: Punctuated<Field, Token![,]>,
    pub generics: Generics,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let super_ty = input.parse()?;

        let ItemStruct {
            ident,
            generics,
            fields,
            ..
        } = input.parse()?;

        let Fields::Unnamed(fields) = fields else {
            return Err(input.error("Only unnamed fields are supported"));
        };

        Ok(Self {
            super_ty,
            offset_ty: ident,
            fields: fields.unnamed,
            generics,
        })
    }
}

impl ToTokens for Input {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let Self {
            super_ty,
            offset_ty,
            fields,
            generics,
        } = self;
        let types = fields.iter().map(|f| &f.ty);
        let sub_states_impl = types
            .map(|ty| (Some(offset_ty), ty))
            .sub_states_impl(super_ty, generics);

        tokens.extend(quote! {
            #(#sub_states_impl)*
        });
    }
}

pub(crate) trait IntoSubStateIter {
    fn sub_states_impl(
        self,
        ident_super: &Ident,
        generics: &Generics,
    ) -> impl Iterator<Item = TokenStream2>;
}

impl<'a, I> IntoSubStateIter for I
where
    I: Iterator<Item = (Option<&'a Ident>, &'a Type)>,
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
                        SubStateOffset<#super_ty #ty_generics, #ident_super #ty_generics>
                    }
                }))
                .unwrap_or(quote! { nalgebra::U0 });
            Some(quote! {
                impl #impl_generics SubStateOf<#ident_super #ty_generics> for #ty
                #where_clause
                {
                    type Offset = #offset;
                }
            })
        })
    }
}
