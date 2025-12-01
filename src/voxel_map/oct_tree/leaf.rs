use crate::voxel_map::uncertain::{
    UncertainPlane, UncertainWorldPoint,
    plane::{PlaneConfig, PlaneInitError},
};

use nalgebra::{RealField, Scalar};

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

impl<T> Leaf<T>
where
    T: RealField,
{
    /// Returns
    ///
    /// - `Err(_)` if the try to create a plane but not vilid, this might need a tree pruning.
    pub fn insert(
        &mut self,
        config: &PlaneConfig<T>,
        depth: u8,
        point: UncertainWorldPoint<T>,
    ) -> Result<(), UncertainWorldPoints<T>> {
        let Leaf {
            plane,
            cached_points,
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

        if plane.is_none() {
            *plane = UncertainPlane::from_uncertain_world_points(points, config).map_or_else(
                |err| {
                    if let PlaneInitError::EigenValueTooBig = err
                        && depth < config.max_layer
                    {
                        Err(std::mem::take(points))
                    } else {
                        Ok(None)
                    }
                },
                |plane| Ok(Some(plane)),
            )?;
        }

        let plane_is_full = len >= config.max_points;

        if plane_is_full {
            // disable plane initialize and update
            *cached_points = None;
        }
        Ok(())
    }
}
