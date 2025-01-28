"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.core.cameradata import CameraData2D
from pyvale.core.sensordata import SensorData

# NOTE: This module is a feature under developement.

def build_pixel_vec_px(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    px_vec_x = np.arange(0,cam_data.num_pixels[0],1)
    px_vec_y = np.arange(0,cam_data.num_pixels[1],1)
    return (px_vec_x,px_vec_y)

def build_pixel_grid_px(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_x,px_vec_y) = build_pixel_vec_px(cam_data)
    return np.meshgrid(px_vec_x,px_vec_y)

def vectorise_pixel_grid_px(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_grid_x,px_grid_y) = build_pixel_grid_px(cam_data)
    return (px_grid_x.flatten(),px_grid_y.flatten())


def build_pixel_vec_leng(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    px_vec_x = np.arange(cam_data.leng_per_px/2,
                         cam_data.field_of_view_local[0],
                         cam_data.leng_per_px)
    px_vec_y = np.arange(cam_data.leng_per_px/2,
                         cam_data.field_of_view_local[1],
                         cam_data.leng_per_px)
    return (px_vec_x,px_vec_y)

def build_pixel_grid_leng(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_x,px_vec_y) = build_pixel_vec_leng(cam_data)
    return np.meshgrid(px_vec_x,px_vec_y)

def vectorise_pixel_grid_leng(cam_data: CameraData2D) -> tuple[np.ndarray,np.ndarray]:
    (px_grid_x,px_grid_y) = build_pixel_grid_leng(cam_data)
    return (px_grid_x.flatten(),px_grid_y.flatten())

def calc_resolution_from_sim(num_px: np.ndarray,
                             coords: np.ndarray,
                             border_px: int,
                             view_plane: tuple[int,int] = (0,1),
                             ) -> float:

    coords_min = np.min(coords, axis=0)
    coords_max = np.max(coords, axis=0)
    field_of_view = np.abs(coords_max - coords_min)
    roi_px = np.array(num_px - 2*border_px,dtype=np.float64)

    resolution = np.zeros_like(view_plane,dtype=np.float64)
    for ii in view_plane:
        resolution[ii] = field_of_view[view_plane[ii]] / roi_px[ii]

    return np.max(resolution)


def calc_centre_from_sim(coords: np.ndarray,
                         view_axes: tuple[int,int] = (0,1)) -> np.ndarray:
    centre = np.mean(coords,axis=0)

    for ii,_ in enumerate(centre):
        if ii not in view_axes:
            centre[ii] = 0.0

    return centre


def build_sensor_data_from_camera(cam_data: CameraData2D) -> SensorData:
    pixels_vectorised = vectorise_pixel_grid_leng(cam_data)

    positions = np.zeros((pixels_vectorised[0].shape[0],3))
    for ii,vv in enumerate(cam_data.view_axes):
        positions[:,vv] = pixels_vectorised[ii] + cam_data.roi_shift_world[ii]

    if cam_data.angle is None:
        angle = None
    else:
        angle = (cam_data.angle,)

    sens_data = SensorData(positions=positions,
                           sample_times=cam_data.sample_times,
                           angles=angle)

    return sens_data


#-------------------------------------------------------------------------------
# NOTE: keep these functions!
# These functions work for 3D cameras calculating imaging dist and fov taking
# account of camera rotation by rotating the bounding box of the sim into cam
# coords

def fov_from_cam_rot(cam_rot: Rotation,
                     coords_world: np.ndarray) -> np.ndarray:
    (xx,yy,zz) = (0,1,2)

    cam_to_world_mat = cam_rot.as_matrix()
    world_to_cam_mat = np.linalg.inv(cam_to_world_mat)

    bb_min = np.min(coords_world,axis=0)
    bb_max = np.max(coords_world,axis=0)

    bound_box_world_vecs = np.array([[bb_min[xx],bb_min[yy],bb_max[zz]],
                                     [bb_max[xx],bb_min[yy],bb_max[zz]],
                                     [bb_max[xx],bb_max[yy],bb_max[zz]],
                                     [bb_min[xx],bb_min[yy],bb_max[zz]],
                                     [bb_min[xx],bb_min[yy],bb_min[zz]],
                                     [bb_max[xx],bb_min[yy],bb_min[zz]],
                                     [bb_max[xx],bb_max[yy],bb_min[zz]],
                                     [bb_min[xx],bb_min[yy],bb_min[zz]],])

    bound_box_cam_vecs = np.matmul(world_to_cam_mat,bound_box_world_vecs.T)
    boundbox_cam_leng = (np.max(bound_box_cam_vecs,axis=1)
                         - np.min(bound_box_cam_vecs,axis=1))

    return np.array((boundbox_cam_leng[xx],boundbox_cam_leng[yy]))


def image_dist_from_fov(num_pixels: np.ndarray,
                        pixel_size: np.ndarray,
                        focal_leng: float,
                        fov_leng: np.ndarray) -> np.ndarray:

    sensor_dims = num_pixels * pixel_size
    fov_angle = 2*np.arctan(sensor_dims/(2*focal_leng))
    image_dist = fov_leng/(2*np.tan(fov_angle/2))
    return image_dist

#-------------------------------------------------------------------------------