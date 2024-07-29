'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale
import pyvale.visualisation
import pyvale.visualisation.plotters

def scalar_linear(coords: np.ndarray,
                  time_steps: np.ndarray) -> np.ndarray:
    xi = 0
    yi = 1
    leng_x = np.max(coords[:,xi]) - np.min(coords[:,xi])
    leng_y = np.max(coords[:,yi]) - np.min(coords[:,yi])

    # shape=(n_nodes,n_timesteps)
    coord_x = np.repeat(np.atleast_2d(coords[:,xi]),time_steps.shape[0],axis=0).T
    coord_y = np.repeat(np.atleast_2d(coords[:,yi]),time_steps.shape[0],axis=0).T
    time_steps = np.repeat(np.atleast_2d(time_steps),coords.shape[0],axis=0)


    f = ((10/(leng_x)*coord_x + 10/(leng_y)*coord_y)*time_steps)+20
    print(f)
    return f


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

    time_steps = np.linspace(0.0,10.0,11)

    connect = np.zeros((n_elems,nodes_per_elem)).astype(np.int64)
    row = 1
    nn = 0
    for ee in range(n_elems):
        nn += 1
        if nn >= row*n_node_x:
            row += 1
            nn += 1

        connect[ee,:] = np.array([nn,nn+1,nn+n_node_x+1,nn+n_node_x])

    sim_data = mh.SimData()
    sim_data.num_spat_dims = 2
    sim_data.coords = coords
    sim_data.connect = dict()
    sim_data.connect['connect1'] = connect.T

    #pv_plot = pyvale.visualisation.plotters.plot_sim_mesh(sim_data)
    #pv_plot.show()

    temp = scalar_linear(coords,time_steps)
    print(temp[:,0])
    temp_grid = np.reshape(temp[:,-1],coord_grid_x.shape)

    fig, ax = plt.subplots()
    cs = ax.contourf(coord_grid_x,
                     coord_grid_y,
                     temp_grid)
    cbar = fig.colorbar(cs)
    plt.show()




if __name__ == '__main__':
    main()
