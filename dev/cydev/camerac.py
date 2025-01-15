"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np

def meshgrid2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x_grid = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    y_grid = np.empty(shape=(y.size, x.size), dtype=y.dtype)

    for ii in range(y.size):
        for jj in range(x.size):
            x_grid[ii,jj] = x[jj]
            y_grid[ii,jj] = y[ii]

    return (x_grid,y_grid)