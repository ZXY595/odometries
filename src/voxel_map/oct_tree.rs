mod branch;
mod iter;
mod leaf;
mod storage;

use crate::{
    frame::WorldPoint,
    voxel_map::{
        MapIndex,
        oct_tree::{
            branch::Branch,
            leaf::Leaf,
            storage::{RootTreeID, TreeID, TreeStorage},
        },
        uncertain::{UncertainPlane, UncertainWorldPoint, plane::PlaneConfig},
    },
};

use nalgebra::{ComplexField, RealField, Scalar};

type UncertainWorldPoints<T> = Vec<UncertainWorldPoint<T>>;

pub struct OctTreeRoot<T: Scalar> {
    /// Allocator for the nodes in the tree.
    storage: TreeStorage<T>,
}

pub struct OctTreeNode<T: Scalar> {
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
    quarter_side_length: T,
    /// Current depth of the node in the tree.
    depth: u8,
}

impl<T: Scalar> OctTreeRoot<T> {
    pub fn iter_planes(&self) -> impl Iterator<Item = &UncertainPlane<T>> {
        self.storage
            .iter_nodes()
            .flat_map(|node| node.tree.leaf_ref()?.plane.as_ref())
    }
}

impl<T: ComplexField> OctTreeRoot<T> {
    pub fn new(index: &WorldPoint<T>, voxel_size: T) -> Self {
        let half_side_length = voxel_size.clone() / nalgebra::convert(2.0);
        let quarter_side_length = voxel_size / nalgebra::convert(4.0);
        let center = index.map_framed_point(|x| x.floor() + half_side_length.clone());
        let root = OctTreeNode::new_leaf(Leaf::new(), center, quarter_side_length, 0);
        Self {
            storage: TreeStorage::new(root),
        }
    }

    pub fn get_root_node(&self) -> &OctTreeNode<T> {
        &self.storage[RootTreeID::new()]
    }
}

impl<T: RealField> OctTreeRoot<T> {
    #[inline]
    pub fn insert(&mut self, config: &PlaneConfig<T>, point: UncertainWorldPoint<T>) {
        OctTreeNode::insert(&None, config, &mut self.storage, point)
    }

    pub fn nearest_voxel(&self, point: &WorldPoint<T>, mut coord: MapIndex<T>) -> MapIndex<T> {
        let NodeState {
            center,
            quarter_side_length,
            ..
        } = &self.get_root_node().state;

        itertools::multizip((point.iter(), center.iter(), coord.iter_mut())).for_each(
            |(point, center, coord)| {
                if *point > center.clone() + quarter_side_length.clone() {
                    *coord += 1;
                } else if *point < center.clone() - quarter_side_length.clone() {
                    *coord -= 1;
                }
            },
        );
        coord
    }
}

impl<T: Scalar> OctTreeNode<T> {
    pub const fn new_leaf(
        leaf: Leaf<T>,
        center: WorldPoint<T>,
        quarter_side_length: T,
        depth: u8,
    ) -> Self {
        Self {
            tree: OctTree::Leaf(leaf),
            state: NodeState {
                center,
                quarter_side_length,
                depth,
            },
        }
    }
    #[expect(unused)]
    const fn new_branch(
        branch: Branch<T>,
        center: WorldPoint<T>,
        side_quarter_length: T,
        depth: u8,
    ) -> Self {
        Self {
            tree: OctTree::Branch(branch),
            state: NodeState {
                center,
                quarter_side_length: side_quarter_length,
                depth,
            },
        }
    }
}

impl<T> OctTreeNode<T>
where
    T: RealField,
{
    fn insert(
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
                let Err(points_not_plane) = leaf.insert(config, node.state.depth, point) else {
                    return;
                };
                // pruning the leaf and creating a new branch
                node.tree = OctTree::Branch(Branch::new());
                // TODO: could be optimize using `rayon`
                points_not_plane
                    .into_iter()
                    .for_each(|point| OctTreeNode::insert(tree_id, config, storage, point));
            }
            _ => {}
        }
    }
}

impl<T: Scalar> OctTree<T> {
    pub(crate) fn leaf_ref(&self) -> Option<&Leaf<T>> {
        if let OctTree::Leaf(leaf) = self {
            Some(leaf)
        } else {
            None
        }
    }
}
