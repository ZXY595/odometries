use syn::{
    Attribute, Data, DeriveInput, Error, Fields, FieldsNamed, Generics, Ident, Result, Visibility,
    parse::{Parse, ParseStream},
};

pub struct StructNamed {
    pub attrs: Vec<Attribute>,
    pub vis: Visibility,
    pub ident: Ident,
    pub generics: Generics,
    pub fields: FieldsNamed,
}

impl Parse for StructNamed {
    fn parse(input: ParseStream) -> Result<Self> {
        let DeriveInput {
            attrs,
            vis,
            ident,
            generics,
            data,
        } = input.parse()?;

        let Data::Struct(data) = data else {
            return Err(Error::new(input.span(), "Expected a struct"));
        };
        let Fields::Named(fields) = data.fields else {
            return Err(Error::new(input.span(), "Expected named fields"));
        };

        Ok(StructNamed {
            attrs,
            vis,
            ident,
            generics,
            fields,
        })
    }
}
