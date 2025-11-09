use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::{
    FieldsNamed, Generics, Ident, MetaList, Result, Token, TypeParamBound, WherePredicate,
    parse::{Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
};

use crate::utils::StructNamed;

pub struct Input {
    pub ident: Ident,
    pub generics: Generics,
    pub fields: FieldsNamed,
    pub predicates_bounds: Option<Punctuated<TypeParamBound, Token![,]>>,
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let StructNamed {
            ident,
            generics,
            fields,
            attrs,
            ..
        } = input.parse()?;

        let predicates_bounds = attrs
            .iter()
            .find(|attr| attr.path().is_ident("vector_add_assign"))
            .and_then(|attr| {
                let meta = attr
                    .parse_args::<MetaList>()
                    .ok()
                    .filter(|meta| meta.path.is_ident("predicates"))?;
                meta.parse_args_with(Punctuated::<TypeParamBound, Token![,]>::parse_terminated)
                    .ok()
            });

        Ok(Self {
            ident,
            generics,
            fields,
            predicates_bounds,
        })
    }
}

impl ToTokens for Input {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let Self {
            ident,
            generics,
            fields,
            predicates_bounds,
        } = self;

        let mut generics = generics.clone();

        let fields = &fields.named;

        let fields_add_assign = fields.iter().filter_map(|field| {
            let ident = field.ident.as_ref()?;
            let bounds = field
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("predicates"))
                .and_then(|attr| {
                    attr.parse_args_with(Punctuated::<TypeParamBound, Token![,]>::parse_terminated)
                        .ok()
                });
            let ty = &field.ty;
            let element_ty = quote! {
                <#ty as KFState>::Element
            };
            Some((
                bounds.map(|bounds| {
                    parse_quote! {
                        #element_ty: #bounds
                    }
                }),
                quote! {
                    self.#ident += rhs.rows_generic(
                        SubStateOffset::<#ty, Self>::DIM,
                        <#ty as KFState>::Dim::name()
                    );
                },
            ))
        });
        let (fields_predicates, sub_add_assigns): (Vec<Option<WherePredicate>>, TokenStream2) =
            fields_add_assign.unzip();

        let ty_generics = generics.split_for_impl().1;

        let ty = quote! {
            #ident #ty_generics
        };

        let ty_dim = quote! {
            <#ty as KFState>::Dim
        };

        let element_ty = quote! {
            <#ty as KFState>::Element
        };

        let rhs_storage = quote! { __AddAssignRhsStorage };

        generics.params.push(parse_quote! {
            #rhs_storage
        });

        let predicates = &mut generics.make_where_clause().predicates;
        predicates.push(parse_quote! { #rhs_storage: nalgebra::Storage<#element_ty, #ty_dim> });
        predicates.extend(predicates_bounds.iter().flatten().map(|bound| {
            let predicate: WherePredicate = parse_quote! { #element_ty: #bound };
            predicate
        }));
        predicates.extend(fields_predicates.into_iter().flatten());

        let (impl_generics, _, where_clause) = generics.split_for_impl();

        tokens.extend(quote! {
            impl #impl_generics std::ops::AddAssign<
                nalgebra::Vector<#element_ty, #ty_dim, #rhs_storage>
            > for #ty
            #where_clause
            {
                fn add_assign(&mut self, rhs: nalgebra::Vector<#element_ty, #ty_dim, #rhs_storage>) {
                    #sub_add_assigns
                }
            }
        });
    }
}
