
"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import time
from pathlib import Path
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import convolve2d

import mooseherder as mh
import pyvale

### cython ###
from icecream import ic


def edge_function(vert_a: np.ndarray,
                  vert_b: np.ndarray,
                  vert_c: np.ndarray) -> np.ndarray:
    xx: int = 0
    yy: int = 1
    edge_fun = ((vert_c[xx] - vert_a[xx]) * (vert_b[yy] - vert_a[yy])
              - (vert_c[yy] - vert_a[yy]) * (vert_b[xx] - vert_a[xx]))
    return edge_fun


def average_subpixel_image(subpx_image: np.ndarray,
                           subsample: int) -> np.ndarray:
    conv_mask = np.ones((subsample,subsample))/(subsample**2)
    if subsample > 1:
        subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
        avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                     round(subsample/2)-1::subsample]
    else:
        subpx_image_conv = subpx_image
        avg_image = subpx_image

    return avg_image