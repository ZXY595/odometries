use std::ops::{Index, IndexMut};

use num_traits::Zero;

use crate::{
    frame::WorldPoint,
    voxel_map::{
        oct_tree::{NodeState, OctTreeNode},
        uncertain::UncertainWorldPoint,
    },
};

use super::{Leaf, storage::TreeID};

use nalgebra::{ClosedSubAssign, RealField, Scalar, Vector3};

type Childrens<T> = [[[Option<TreeID<T>>; 2]; 2]; 2];

pub(crate) struct Branch<T: Scalar> {
    pub(crate) childrens: Childrens<T>,
}

impl<T: Scalar> Branch<T> {
    pub const fn new() -> Self {
        Self {
            childrens: [[[None, None], [None, None]], [[None, None], [None, None]]],
        }
    }
}

impl<T: Scalar> Default for Branch<T> {
    fn default() -> Self {
        Self {
            childrens: Default::default(),
        }
    }
}

impl<T> Index<&Vector3<bool>> for Branch<T>
where
    T: Scalar + ClosedSubAssign + PartialOrd + Zero,
{
    type Output = Option<TreeID<T>>;

    #[inline]
    fn index(&self, coord: &Vector3<bool>) -> &Self::Output {
        let index = coord.map(|x| x as usize);
        &self.childrens[index.z][index.y][index.x]
    }
}

impl<T> IndexMut<&Vector3<bool>> for Branch<T>
where
    T: Scalar + ClosedSubAssign + PartialOrd + Zero,
{
    #[inline]
    fn index_mut(&mut self, coord: &Vector3<bool>) -> &mut Self::Output {
        let index = coord.map(|x| x as usize);
        &mut self.childrens[index.z][index.y][index.x]
    }
}

impl<T> Branch<T>
where
    T: RealField,
{
    pub(super) fn new_child(
        state: &NodeState<T>,
        coord: &Vector3<bool>,
        point: UncertainWorldPoint<T>,
    ) -> OctTreeNode<T> {
        let side_quarter_length = &state.side_quarter_length;
        let depth = state.depth;

        let child_center = &state.center
            + coord.map(|x| {
                if x {
                    side_quarter_length.clone()
                } else {
                    -side_quarter_length.clone()
                }
            });
        let half_size_length = side_quarter_length.clone() / nalgebra::convert(2.0);
        OctTreeNode::new_leaf(
            Leaf::new_with_point(point),
            child_center,
            half_size_length,
            depth + 1,
        )
    }
}

pub(crate) fn point_to_coord<T>(point: &WorldPoint<T>, center: &WorldPoint<T>) -> Vector3<bool>
where
    T: Scalar + ClosedSubAssign + PartialOrd + Zero,
{
    (point - center).map(|x| x > T::zero())
}
