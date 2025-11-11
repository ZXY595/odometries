use std::ops::{Deref, DerefMut};

/// A helper struct to hold a value with its context.
pub struct InContext<T, C> {
    /// The inner value of [`Contextual`]
    pub inner: T,
    pub context: C,
}

impl<T, C> InContext<T, C> {
    pub const fn new(inner: T, context: C) -> Self {
        Self { inner, context }
    }
    #[expect(unused)]
    pub fn map_context<C2>(self, f: impl FnOnce(C) -> C2) -> InContext<T, C2> {
        InContext::new(self.inner, f(self.context))
    }
}

impl<T, C> Deref for InContext<T, C> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, C> DerefMut for InContext<T, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[expect(unused)]
pub trait WithContext<C>: Sized {
    fn with_context(self, context: C) -> InContext<Self, C>;
    fn mut_with_context(&mut self, config: C) -> InContext<&mut Self, C>;
}

impl<T, C> WithContext<C> for T {
    fn with_context(self, context: C) -> InContext<Self, C> {
        InContext::new(self, context)
    }
    fn mut_with_context(&mut self, context: C) -> InContext<&mut Self, C> {
        InContext::new(self, context)
    }
}
