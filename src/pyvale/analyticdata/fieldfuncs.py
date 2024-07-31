'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

def scalar_linear2d(coord_y: np.ndarray | float,
                coord_x: np.ndarray | float,
                time: np.ndarray | float = 1.0,
                x_grad: float = 10.0,
                y_grad: float = 10.0,
                offset: float = 20.0) -> np.ndarray | float:

    f_eval = ((x_grad*coord_x +
               y_grad*coord_y))*time + offset
    return f_eval


def scalar_linear2d_int(coord_y: np.ndarray | float,
                    coord_x: np.ndarray | float,
                    time: np.ndarray | float = 1.0,
                    x_grad: float = 10.0,
                    y_grad: float = 10.0,
                    offset: float = 20.0) -> np.ndarray | float:

    f_eval = ((x_grad/2 * coord_x**2 * coord_y +
              y_grad/2 * coord_y**2 * coord_x)*time +
              offset*coord_x*coord_y)
    return f_eval


def scalar_quad2d(coord_y: np.ndarray | float,
                coord_x: np.ndarray | float,
                time: np.ndarray | float = 1.0,
                roots_x: tuple[float,float] = (0.0,10.0),
                roots_y: tuple[float,float] = (0.0,10.0),
                offset: float = 20.0) -> np.ndarray | float:

    f_eval = ((coord_x - roots_x[0]) * (coord_x - roots_x[1]) *
              (coord_y - roots_y[0]) * (coord_y - roots_y[1]))*time + offset
    return f_eval