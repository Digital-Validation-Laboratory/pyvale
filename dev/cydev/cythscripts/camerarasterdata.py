"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation
import numpy as np

@dataclass(slots=True)
class CameraRasterData:
    num_pixels: np.ndarray
    pixel_size: np.ndarray

    pos_world: np.ndarray
    rot_world: Rotation

    roi_center_world: np.ndarray

    focal_length: float = 50.0
    sub_samp: int = 2

    back_face_removal: bool = True

    sensor_size: np.ndarray = field(init=False)
    image_dims: np.ndarray = field(init=False)
    image_dist: float = field(init=False)
    cam_to_world_mat: np.ndarray = field(init=False)
    world_to_cam_mat: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.image_dist = np.linalg.norm(self.pos_world - self.roi_center_world)
        self.sensor_size = self.num_pixels*self.pixel_size
        self.image_dims = (self.image_dist
                           *self.sensor_size/self.focal_length)

        self.cam_to_world_mat = np.zeros((4,4))
        self.cam_to_world_mat[0:3,0:3] = self.rot_world.as_matrix()
        self.cam_to_world_mat[-1,-1] = 1.0
        self.cam_to_world_mat[0:3,-1] = self.pos_world
        self.world_to_cam_mat = np.linalg.inv(self.cam_to_world_mat)
