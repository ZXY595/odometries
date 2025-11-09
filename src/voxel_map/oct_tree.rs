mod branch;
mod iter;
mod leaf;
mod storage;

use crate::{
    frame::WorldPoint,
    voxel_map::{
        oct_tree::{
            branch::Branch,
            leaf::Leaf,
            storage::{RootTreeID, TreeID, TreeStorage},
        },
        uncertain::{UncertainPlane, UncertainWorldPoint, plane::PlaneConfig},
    },
};

use nalgebra::{RealField, Scalar};

type UncertainWorldPoints<T> = Vec<UncertainWorldPoint<T>>;

pub struct OctTreeRoot<T: Scalar> {
    /// Allocator for the nodes in the tree.
    storage: TreeStorage<T>,
}

pub(crate) struct OctTreeNode<T: Scalar> {
    tree: OctTree<T>,
    state: NodeState<T>,
}

pub(crate) enum OctTree<T: Scalar> {
    Branch(Branch<T>),
    Leaf(Leaf<T>),
}

struct NodeState<T: Scalar> {
    center: WorldPoint<T>,
    /// The quarter length of the side of the node.
    side_quarter_length: T,
    /// Current depth of the node in the tree.
    depth: u8,
}

impl<T> OctTreeRoot<T>
where
    T: RealField + Default,
{
    pub fn new(index: &WorldPoint<T>, voxel_size: T) -> Self {
        let half_quarter_length = voxel_size.clone() / nalgebra::convert(2.0);
        let side_quarter_length = voxel_size / nalgebra::convert(4.0);
        let center = index.map_framed_point(|x| x.floor() + half_quarter_length.clone());
        let root = OctTreeNode::new_leaf(Leaf::new(), center, side_quarter_length, 0);
        Self {
            storage: TreeStorage::new(root),
        }
    }
    #[expect(unused)]
    pub fn get_root_node(&self) -> &OctTreeNode<T> {
        &self.storage[RootTreeID::new()]
    }
    pub fn insert(&mut self, config: &PlaneConfig<T>, point: UncertainWorldPoint<T>) {
        OctTreeNode::insert(&None, config, &mut self.storage, point)
    }
}

impl<T: Scalar> OctTreeRoot<T> {
    pub fn iter_planes(&self) -> impl Iterator<Item = &UncertainPlane<T>> {
        self.storage.0.iter().filter_map(|(_, node)| {
            if let OctTree::Leaf(Leaf { plane, .. }) = &node.tree {
                plane.as_ref()
            } else {
                None
            }
        })
    }
}

impl<T: Scalar> OctTreeNode<T> {
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
    #[expect(unused)]
    pub const fn new_branch(
        branch: Branch<T>,
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

impl<T> OctTreeNode<T>
where
    T: RealField + Default,
{
    pub fn insert(
        tree_id: &Option<TreeID<T>>,
        config: &PlaneConfig<T>,
        storage: &mut TreeStorage<T>,
        point: UncertainWorldPoint<T>,
    ) {
        let (bottom_tree, coord) = iter::DescendIter::new(tree_id.clone(), &point, storage)
            .last()
            .unzip();

        let vacant_alloc = storage.vacant_alloc();
        let node = &mut storage[bottom_tree.clone()];
        match (&mut node.tree, coord) {
            (OctTree::Branch(branch), Some(coord)) => {
                let new_leaf = Branch::new_child(&node.state, &coord, point);
                vacant_alloc.alloc(new_leaf, |new_id| branch[&coord] = new_id)(storage);
            }
            (OctTree::Leaf(leaf), _) => {
                if let Err(points_not_plane) = leaf.insert(config, node.state.depth, point) {
                    // pruning the leaf and creating a new branch
                    node.tree = OctTree::Branch(Branch::new());
                    // TODO: could be optimize using `rayon`
                    points_not_plane.into_iter().for_each(|point| {
                        OctTreeNode::insert(&tree_id.clone(), config, storage, point)
                    });
                }
            }
            _ => {}
        }
    }
}
