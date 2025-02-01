"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage