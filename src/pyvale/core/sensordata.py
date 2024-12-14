'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.core.integratortype import EIntSpatialType


@dataclass(slots=True)
class SensorData:
    #shape=(n_sensors,3) where second dim=[x,y,z]
    positions: np.ndarray | None = None
    #shape=(n_time_steps,)
    sample_times: np.ndarray | None = None
    #shape=(n_sensors,)
    angles: tuple[Rotation,...] | None = None
    spatial_averager: EIntSpatialType | None = None
    #shape=(3,) where  dim=[x,y,z]
    spatial_dims: np.ndarray | None = None





