use std::ops::Deref;

use nalgebra::{RealField, Scalar, Vector3};

use crate::voxel_map::uncertain::UncertainWorldPoint;

use super::{
    OctTree, branch,
    storage::{TreeID, TreeStorage},
};

pub(crate) struct DescendIter<'p, 'store, T: Scalar> {
    /// None when `current` is root node
    current: Option<TreeID<T>>,
    point: &'p UncertainWorldPoint<T>,
    storage: &'store TreeStorage<T>,
}

impl<'p, 'store, T: Scalar> DescendIter<'p, 'store, T> {
    #[expect(unused)]
    pub const fn from_root(
        point: &'p UncertainWorldPoint<T>,
        storage: &'store TreeStorage<T>,
    ) -> Self {
        Self::new(None, point, storage)
    }
    pub const fn new(
        current: Option<TreeID<T>>,
        point: &'p UncertainWorldPoint<T>,
        storage: &'store TreeStorage<T>,
    ) -> Self {
        Self {
            current,
            point,
            storage,
        }
    }
}

impl<'p, 'store, T: RealField> Iterator for DescendIter<'p, 'store, T> {
    type Item = (TreeID<T>, Vector3<bool>);

    fn next(&mut self) -> Option<Self::Item> {
        let node = &self.storage[self.current.clone()];

        match &node.tree {
            OctTree::Branch(branch) => {
                let center = &node.state.center;
                let coord = branch::point_to_coord(self.point.deref(), center);
                let descend_id = branch[&coord].clone()?;
                self.current = Some(descend_id.clone());
                Some((descend_id, coord))
            }
            OctTree::Leaf(_) => None,
        }
    }
}
