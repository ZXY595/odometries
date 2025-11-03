use nalgebra::Scalar;

use crate::voxel_map::uncertain::UncertainPlane;

use super::{OctTree, OctTreeNode};

pub struct PlaneIter<'alloc, T: Scalar> {
    stack: Vec<&'alloc OctTreeNode<'alloc, T>>,
}

impl<'alloc, T: Scalar> PlaneIter<'alloc, T> {
    pub fn new(root: &'alloc OctTreeNode<'alloc, T>) -> Self {
        Self { stack: vec![root] }
    }
}

impl<'alloc, T: Scalar> Iterator for PlaneIter<'alloc, T> {
    type Item = &'alloc UncertainPlane<T>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match &node.tree {
                OctTree::Leaf(leaf) => {
                    if let Some(plane) = &leaf.plane {
                        return Some(plane);
                    }
                }
                OctTree::Branch(branch) => {
                    branch.iter().for_each(|child| self.stack.push(child));
                }
            }
        }
        None
    }
}
