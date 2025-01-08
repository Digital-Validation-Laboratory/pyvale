"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation


# NOTE: This module is a feature under developement.
#
# - Camera Local Coords: Pixel positions in pixels/meters
# - Global Sim Coords: Transform from local pixel positions to sim coords in meters
# - For this transformation we need user to specify center of ROI in sim coords
# - There are going to be different ways to specify the camera properties

# For thin lens theory will need to know some combination of:
#   - The focal length of the lense
#   - The working distance

@dataclass(slots=True)
class CameraData2D:
    #shape=(n_px_X,n_px_Y)
    num_pixels: np.ndarray

    # Center location of the region of interest in world coords
    #shape=(3,) as (x,y,z)
    roi_center_world: np.ndarray

    # Converts pixels to length units to align with global coords
    leng_per_px: float

    #shape=(n_time_steps,)
    sample_times: np.ndarray | None = None

    #TODO: this only works for flat surfaces aligned with the axis
    view_axes: tuple[int,int] = (0,1)

    bits_sensor: int = 16
    bits_file: int = 16

    angle: Rotation | None = None

    field_of_view_center_local: np.ndarray = field(init=False)
    field_of_view_local: np.ndarray = field(init=False)
    roi_shift_world: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.field_of_view_local = self.num_pixels*self.leng_per_px
        self.field_of_view_center_local = self.field_of_view_local/2

        self.roi_shift_world = np.zeros_like(self.roi_center_world)
        for ii,vv in enumerate(self.view_axes):
            self.roi_shift_world[vv] = self.roi_center_world[vv] - \
                self.field_of_view_center_local[ii]




