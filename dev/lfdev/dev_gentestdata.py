'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

import mooseherder as mh

def scalar_linear(coords: np.ndarray) -> np.ndarray:
    (x_min,x_max) = (np.min(coords[:,0]),np.max(coords[:,0]))
    (y_min,y_max) = (np.min(coords[:,1]),np.max(coords[:,1]))
    return np.array([])


def main() -> None:
    leng_x = 100
    leng_y = 75
    n_elem_x = 4
    n_elem_y = 3

    n_elems = n_elem_x*n_elem_y
    l_elem_x = leng_x/n_elem_x
    l_elem_y = leng_y/n_elem_y

    n_node_x = n_elem_x+1
    n_node_y = n_elem_y+1
    nodes_per_elem = 4
    n_nodes = n_node_x*n_node_y

    coord_x = np.linspace(0,leng_x,n_node_x)
    coord_y = np.linspace(0,leng_y,n_node_y)

    (coord_grid_x,coord_grid_y) = np.meshgrid(coord_x,coord_y)

    coord_x = np.atleast_2d(coord_grid_x.flatten()).T
    coord_y = np.atleast_2d(coord_grid_y.flatten()).T
    coord_z = np.zeros_like(coord_x)
    coords = np.hstack((coord_x,coord_y,coord_z))

    node_nums = np.arange(0,n_nodes)+1

    connect = np.zeros((n_elems,nodes_per_elem))
    row = 1
    nn = 0
    for ee in range(n_elems):
        nn += 1
        if nn >= row*n_node_x:
            row += 1
            nn += 1

        connect[ee,:] = np.array([nn,nn+1,nn+n_node_x+1,nn+n_node_x])

        print(f'connect e{ee+1} = {connect[ee,:]}')

    for nn in range(n_nodes):
        print(f'n{nn+1}={coords[nn,:]}')

    sim_data = mh.SimData()
    sim_data.coords = coords


if __name__ == '__main__':
    main()
