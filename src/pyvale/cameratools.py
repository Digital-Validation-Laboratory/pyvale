
'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
from pyvale.cameradata import CameraData2D
from pyvale.sensordata import SensorData

#-------------------------------------------------------------------------------
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

#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------

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


#-------------------------------------------------------------------------------
def build_sensor_data_from_camera(cam_data: CameraData2D) -> SensorData:
    pixels_vectorised = vectorise_pixel_grid_leng(cam_data)

    positions = np.zeros((pixels_vectorised[0].shape[0],3))
    for ii,vv in enumerate(cam_data.view_axes):
        positions[:,vv] = pixels_vectorised[ii] + cam_data.roi_shift_world[ii]

    sens_data = SensorData(positions=positions,
                           sample_times=cam_data.sample_times,
                           angles=cam_data.angle)

    return sens_data