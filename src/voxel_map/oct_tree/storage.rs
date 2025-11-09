use std::{
    marker::PhantomData,
    num::NonZero,
    ops::{Index, IndexMut},
};

use super::OctTreeNode;

use nalgebra::Scalar;
use slab::Slab;

/// TODO: add another storage type that implements [`Send`] and [`Sync`],
/// for thread-safe storage to integrate with the `rayon` crate.
pub(crate) struct TreeStorage<T: Scalar>(pub(crate) Slab<OctTreeNode<T>>);

#[derive(Debug, PartialEq)]
pub(crate) struct TreeID<T> {
    /// zero is reserved for the root node.
    pub(crate) index: NonZero<usize>,
    _marker: PhantomData<T>,
}

pub(crate) struct RootTreeID<T>(PhantomData<T>);

/// A vacant allocation, this will do nothing until the [`VacantAlloc::alloc`] method is called.
pub(crate) struct VacantAlloc<T>(Option<TreeID<T>>);

impl<T> TreeID<T> {
    const fn new(index: NonZero<usize>) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }
    fn new_maybe_root(index: usize) -> Option<Self> {
        NonZero::new(index).map(|index| Self::new(index))
    }
}

impl<T> RootTreeID<T> {
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar> Index<RootTreeID<T>> for TreeStorage<T> {
    type Output = OctTreeNode<T>;

    #[inline]
    fn index(&self, _: RootTreeID<T>) -> &Self::Output {
        // SAFETY:
        //
        // The root node is always at index 0,
        // so it's safe to assume that the index is valid.
        unsafe { self.0.get_unchecked(0) }
    }
}

impl<T: Scalar> IndexMut<RootTreeID<T>> for TreeStorage<T> {
    #[inline]
    fn index_mut(&mut self, _: RootTreeID<T>) -> &mut Self::Output {
        // SAFETY:
        //
        // See also Self `Index<RootTreeID<T>>` implemenation.
        unsafe { self.0.get_unchecked_mut(0) }
    }
}

impl<T: Scalar> Index<TreeID<T>> for TreeStorage<T> {
    type Output = OctTreeNode<T>;

    #[inline]
    fn index(&self, id: TreeID<T>) -> &Self::Output {
        let index = id.index.get();
        // SAFETY:
        //
        // [`TreeID`] can only be allocated from the [`TreeStorage`] and can't be modifed,
        // so it's safe to assume that the index is valid.
        unsafe { self.0.get_unchecked(index) }
    }
}

impl<T: Scalar> IndexMut<TreeID<T>> for TreeStorage<T> {
    #[inline]
    fn index_mut(&mut self, id: TreeID<T>) -> &mut Self::Output {
        let index = id.index.get();
        // SAFETY:
        //
        // See also Self `Index<TreeID<T>>` implemenation.
        unsafe { self.0.get_unchecked_mut(index) }
    }
}

impl<T: Scalar> Index<Option<TreeID<T>>> for TreeStorage<T> {
    type Output = OctTreeNode<T>;

    #[inline]
    fn index(&self, id: Option<TreeID<T>>) -> &Self::Output {
        id.map_or_else(|| &self[RootTreeID::new()], |id| &self[id])
    }
}

impl<T: Scalar> IndexMut<Option<TreeID<T>>> for TreeStorage<T> {
    #[inline]
    fn index_mut(&mut self, id: Option<TreeID<T>>) -> &mut Self::Output {
        match id {
            Some(id) => &mut self[id],
            None => &mut self[RootTreeID::new()],
        }
    }
}

impl<T: Scalar> TreeStorage<T> {
    pub(crate) fn new(root: OctTreeNode<T>) -> Self {
        let mut storage = Self(Slab::with_capacity(1));
        let index = storage.0.insert(root);
        debug_assert_eq!(index, 0);
        storage
    }
    #[must_use]
    pub fn alloc(&mut self, node: OctTreeNode<T>) -> Option<TreeID<T>> {
        let index = self.0.insert(node);
        TreeID::new_maybe_root(index)
    }
    /// Returns a vacant allocation,
    /// useful for inserting nodes into the tree according a exisiting node.
    pub fn vacant_alloc(&self) -> VacantAlloc<T> {
        let index = self.0.vacant_key();
        VacantAlloc(TreeID::new_maybe_root(index))
    }
}

impl<T: Scalar> VacantAlloc<T> {
    /// Allocates the given node and returns a closure that can be used to insert it into the [`TreeStorage`].
    #[must_use = "The returned closure must be called to actually insert the node into the tree."]
    pub fn alloc(
        self,
        node: OctTreeNode<T>,
        f: impl FnOnce(Option<TreeID<T>>),
    ) -> impl FnOnce(&mut TreeStorage<T>) + 'static {
        #[cfg(debug_assertions)]
        let index = self.0.clone();
        f(self.0);
        move |storage| {
            // Add more test here
            debug_assert_eq!(index, storage.alloc(node));
        }
    }
}

impl<T> Clone for TreeID<T> {
    fn clone(&self) -> Self {
        Self::new(self.index)
    }
}
