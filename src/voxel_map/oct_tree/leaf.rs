use crate::voxel_map::uncertain::{
    UncertainPlane, UncertainWorldPoint,
    plane::{PlaneConfig, PlaneInitError},
};

use nalgebra::{RealField, Scalar};

use crate::context::Contextual;

use super::UncertainWorldPoints;

pub struct Leaf<T: Scalar> {
    pub plane: Option<UncertainPlane<T>>,
    /// Cached points for initializing a new plane or updating the existing plane.
    /// If the cached points is `None`, the plane is not need to be updated.
    pub(crate) cached_points: Option<UncertainWorldPoints<T>>,
}

impl<T: Scalar> Leaf<T> {
    pub const fn new() -> Self {
        Self {
            plane: None,
            cached_points: Some(Vec::new()),
        }
    }
    pub fn new_with_point(point: UncertainWorldPoint<T>) -> Self {
        Self {
            plane: None,
            cached_points: Some(vec![point]),
        }
    }
}

impl<T> Contextual<&mut Leaf<T>, (&PlaneConfig<T>, u8)>
where
    T: RealField + Default,
{
    /// Returns
    ///
    /// - `Err(_)` if the try to create a plane but not vilid, this might need a tree pruning.
    pub fn push(&mut self, point: UncertainWorldPoint<T>) -> Result<(), UncertainWorldPoints<T>> {
        let &mut Contextual {
            inner: Leaf {
                plane,
                cached_points,
            },
            context: (config, depth),
        } = self;

        let Some(points) = cached_points else {
            // cached points is disabled, the plane is not need to be updated.
            return Ok(());
        };

        points.push(point);

        let len = points.len();

        let is_plane_needs_update = || len % config.update_threshold == 0;

        // drop the old plane if it needs update
        let _ = plane.take_if(|_| is_plane_needs_update());

        *plane = match UncertainPlane::from_uncertain_world_points(points, config) {
            Ok(plane) => Some(plane),
            Err(PlaneInitError::TooFewPoints) => None,
            Err(PlaneInitError::EigenValueTooBig) if depth < config.max_layer => {
                // return and pruning the tree.
                return Err(std::mem::take(points));
            }
            _ => None,
        };

        let plane_is_full = len >= config.max_points;

        if plane_is_full {
            // disable plane initialize and update
            *cached_points = None;
        }
        Ok(())
    }
}
