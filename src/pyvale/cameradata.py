'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass(slots=True)
class CameraData:
    num_pixels: tuple[int, int]
    m_per_px: float

    #shape=(n_time_steps,)
    sample_times: np.ndarray | None = None

    position: tuple[float,float,float]
    angle: Rotation
    field_of_view: tuple[float,float]

