'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import numpy as np


def rectangle_mesh_2d(leng_x: float,
                      leng_y: float,
                      n_elem_x: int,
                      n_elem_y) -> tuple[np.ndarray,np.ndarray]:

    n_elems = n_elem_x*n_elem_y
    n_node_x = n_elem_x+1
    n_node_y = n_elem_y+1
    nodes_per_elem = 4

    coord_x = np.linspace(0,leng_x,n_node_x)
    coord_y = np.linspace(0,leng_y,n_node_y)
    (coord_grid_x,coord_grid_y) = np.meshgrid(coord_x,coord_y)

    coord_x = np.atleast_2d(coord_grid_x.flatten()).T
    coord_y = np.atleast_2d(coord_grid_y.flatten()).T
    coord_z = np.zeros_like(coord_x)
    coords = np.hstack((coord_x,coord_y,coord_z))

    connect = np.zeros((n_elems,nodes_per_elem)).astype(np.int64)
    row = 1
    nn = 0
    for ee in range(n_elems):
        nn += 1
        if nn >= row*n_node_x:
            row += 1
            nn += 1

        connect[ee,:] = np.array([nn,nn+1,nn+n_node_x+1,nn+n_node_x])
    connect = connect.T

    return (coords,connect)