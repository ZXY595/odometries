mod branch;
mod iter;
mod leaf;

use crate::{
    context::{Contextual, WithContext},
    frame::WorldPoint,
    voxel_map::{
        oct_tree::{branch::Branch, leaf::Leaf},
        uncertain::{UncertainPlane, UncertainWorldPoint, plane::PlaneConfig},
    },
};

use nalgebra::{RealField, Scalar};
use typed_arena::Arena;

type TreeAllocator<'alloc, T> = Arena<OctTreeNode<'alloc, T>>;
type UncertainWorldPoints<T> = Vec<UncertainWorldPoint<T>>;

pub struct OctTreeNode<'alloc, T: Scalar> {
    tree: OctTree<'alloc, T>,
    state: NodeState<T>,
}

struct NodeState<T: Scalar> {
    center: WorldPoint<T>,
    /// The quarter length of the side of the node.
    side_quarter_length: T,
    /// Current depth of the node in the tree.
    depth: u8,
}

pub enum OctTree<'alloc, T: Scalar> {
    Branch(Branch<'alloc, T>),
    Leaf(Leaf<T>),
}

impl<'alloc, T: Scalar> OctTreeNode<'alloc, T> {
    pub const fn new_leaf(
        leaf: Leaf<T>,
        center: WorldPoint<T>,
        side_quarter_length: T,
        depth: u8,
    ) -> Self {
        Self {
            tree: OctTree::Leaf(leaf),
            state: NodeState {
                center,
                side_quarter_length,
                depth,
            },
        }
    }
    pub const fn new_branch(
        branch: Branch<'alloc, T>,
        center: WorldPoint<T>,
        side_quarter_length: T,
        depth: u8,
    ) -> Self {
        Self {
            tree: OctTree::Branch(branch),
            state: NodeState {
                center,
                side_quarter_length,
                depth,
            },
        }
    }
}

type ContextualTreeNodeMut<'this, 'alloc, T> = Contextual<
    &'this mut OctTreeNode<'alloc, T>,
    (&'this PlaneConfig<T>, &'alloc TreeAllocator<'alloc, T>),
>;

impl<T> ContextualTreeNodeMut<'_, '_, T>
where
    T: RealField + Default,
{
    pub fn push(&mut self, point: UncertainWorldPoint<T>) {
        let &mut Contextual {
            inner:
                &mut OctTreeNode {
                    ref mut tree,
                    ref state,
                },
            context: (config, allocator),
        } = self;

        match tree {
            OctTree::Branch(branch) => branch.with_context((config, allocator, state)).push(point),
            OctTree::Leaf(leaf) => {
                if let Err(points_not_plane) = leaf.with_context((config, state.depth)).push(point)
                {
                    // pruning the leaf and creating a new branch
                    let mut branch = Branch::default();
                    branch
                        .mut_with_context((config, allocator, state))
                        .extend(points_not_plane);
                    *tree = OctTree::Branch(branch)
                }
            }
        }
    }
}

impl<T> Extend<UncertainWorldPoint<T>> for ContextualTreeNodeMut<'_, '_, T>
where
    T: RealField + Default,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = UncertainWorldPoint<T>>,
    {
        // TODO: could use `rayon`
        iter.into_iter().for_each(|point| self.push(point));
    }
}

impl<'alloc, T: Scalar> OctTreeNode<'alloc, T> {
    pub fn iter_planes(&'alloc self) -> impl Iterator<Item = &'alloc UncertainPlane<T>> {
        iter::PlaneIter::new(self)
    }
}
