'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

# Need to know:
# The centre locations
# The dimensions of the quad
#   - Calculate the area
# The orientation of the quad - assume aligned for now
#   - Assume unit normal perpendicular to the plane
#
# The number of points to sample: can default
class Quad2D:
    def __init__(self,
                 sens_pos: np.ndarray,
                 dims: tuple[float,float]) -> None:
        self._sens_pos = sens_pos
        self._dims = dims




class Disc2D:
    def __init__(self) -> None:
        pass
