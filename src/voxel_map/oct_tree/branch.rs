use std::ops::{Deref, DerefMut, Index, IndexMut};

use num_traits::Zero;

use crate::{
    context::{Contextual, WithContext},
    frame::WorldPoint,
    voxel_map::{
        oct_tree::{NodeState, OctTreeNode, TreeAllocator},
        uncertain::{UncertainWorldPoint, plane::PlaneConfig},
    },
};

use super::Leaf;

use nalgebra::{ClosedSubAssign, RealField, Scalar, Vector3};

type Childrens<'alloc, T> = [[[Option<&'alloc mut OctTreeNode<'alloc, T>>; 2]; 2]; 2];

pub struct Branch<'alloc, T: Scalar> {
    pub(crate) childrens: Childrens<'alloc, T>,
}

impl<'alloc, T: Scalar> Branch<'alloc, T> {
    pub fn iter(&self) -> impl Iterator<Item = &&'alloc mut OctTreeNode<'alloc, T>> {
        self.childrens.iter().flatten().flatten().flatten()
    }
}

impl<T: Scalar> Default for Branch<'_, T> {
    fn default() -> Self {
        Self {
            childrens: Default::default(),
        }
    }
}

impl<'alloc, T> Index<Vector3<bool>> for Branch<'alloc, T>
where
    T: Scalar + ClosedSubAssign + PartialOrd + Zero,
{
    type Output = Option<&'alloc mut OctTreeNode<'alloc, T>>;

    fn index(&self, coord: Vector3<bool>) -> &Self::Output {
        &self.childrens[coord.z as usize][coord.y as usize][coord.x as usize]
    }
}

impl<T> IndexMut<Vector3<bool>> for Branch<'_, T>
where
    T: Scalar + ClosedSubAssign + PartialOrd + Zero,
{
    fn index_mut(&mut self, coord: Vector3<bool>) -> &mut Self::Output {
        &mut self.childrens[coord.z as usize][coord.y as usize][coord.x as usize]
    }
}

type ContextualBranch<'this, 'alloc, T> = Contextual<
    &'this mut Branch<'alloc, T>,
    (
        &'this PlaneConfig<T>,
        &'alloc TreeAllocator<'alloc, T>,
        &'this NodeState<T>,
    ),
>;

impl<T> ContextualBranch<'_, '_, T>
where
    T: RealField + Default,
{
    pub fn push(&mut self, point: UncertainWorldPoint<T>) {
        let &mut Contextual {
            inner: &mut ref mut branch,
            context:
                (
                    config,
                    allocator,
                    NodeState {
                        center,
                        side_quarter_length,
                        depth,
                    },
                ),
        } = self;

        let child_coord = point_to_coord(point.deref(), center);
        let sub_tree = &mut branch[child_coord];

        match sub_tree {
            Some(sub_tree) => {
                sub_tree
                    .deref_mut()
                    .with_context((config, allocator))
                    .push(point);
            }
            None => {
                let child_center = center
                    + child_coord.map(|x| {
                        if x {
                            side_quarter_length.clone()
                        } else {
                            -side_quarter_length.clone()
                        }
                    });
                let half_size_length = side_quarter_length.clone() / nalgebra::convert(2.0);
                let new_tree = allocator.alloc(OctTreeNode::new_leaf(
                    Leaf::new_with_point(point),
                    child_center,
                    half_size_length,
                    depth + 1,
                ));
                *sub_tree = Some(new_tree)
            }
        };
    }
}

impl<T> Extend<UncertainWorldPoint<T>> for ContextualBranch<'_, '_, T>
where
    T: RealField + Default,
{
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = UncertainWorldPoint<T>>,
    {
        // TODO: could use `rayon`
        iter.into_iter().for_each(|point| {
            self.push(point);
        })
    }
}
fn point_to_coord<T>(point: &WorldPoint<T>, center: &WorldPoint<T>) -> Vector3<bool>
where
    T: Scalar + ClosedSubAssign + PartialOrd + Zero,
{
    (point - center).map(|x| x > T::zero())
}
