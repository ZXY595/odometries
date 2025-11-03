use std::ops::{Deref, DerefMut};

// use reborrow::ReborrowMut;

/// A helper struct to hold a value with its context.
pub struct Contextual<T, C> {
    /// The inner value of [`Contextual`]
    pub inner: T,
    pub context: C,
}

// impl<'short, T: ReborrowMut<'short>, C> ReborrowMut<'short> for Contextual<T, C> {}

impl<T, C> Contextual<T, C> {
    pub const fn new(inner: T, context: C) -> Self {
        Self { inner, context }
    }
    // pub fn as_mut(&mut self) -> Contextual<&mut T, &mut C> {
    //     Contextual {
    //         inner: &mut self.inner,
    //         context: &mut self.context,
    //     }
    // }
    pub fn map_context<C2>(self, f: impl FnOnce(C) -> C2) -> Contextual<T, C2> {
        Contextual::new(self.inner, f(self.context))
    }
}

// impl<T, C> Contextual<T, &C> {
//     pub fn as_inner_mut(&mut self) -> Contextual<&mut T, &C> {
//         Contextual {
//             inner: &mut self.inner,
//             context: self.context,
//         }
//     }
// }

impl<T, C> Deref for Contextual<T, C> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, C> DerefMut for Contextual<T, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub trait WithContext<C>: Sized {
    fn with_context(self, context: C) -> Contextual<Self, C>;
    fn mut_with_context(&mut self, config: C) -> Contextual<&mut Self, C>;
}

impl<T, C> WithContext<C> for T {
    fn with_context(self, context: C) -> Contextual<Self, C> {
        Contextual::new(self, context)
    }
    fn mut_with_context(&mut self, context: C) -> Contextual<&mut Self, C> {
        Contextual::new(self, context)
    }
}
