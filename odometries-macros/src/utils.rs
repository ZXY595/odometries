use syn::{
    Attribute, Data, DeriveInput, Fields, FieldsNamed, Generics, Ident, Result, Visibility,
    parse::{Parse, ParseStream},
};

#[expect(dead_code)]
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
            return Err(input.error("Expected a struct"));
        };
        let Fields::Named(fields) = data.fields else {
            return Err(input.error("Expected named fields"));
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
