'''
================================================================================
pycave: the python computer aided validation engine.
License: LGPL-2.1
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import pyvista as pv
from pyvista import CellType

import mooseherder as mh


class Field:
    def __init__(self, sim_data: mh.SimData, name: str, dim: int = 3) -> None:
        self._name = name
        self._data_grid = convert_simdata_to_pyvista(sim_data,dim)
        self._data_grid = attach_field_to_pyvista(self._data_grid,
                                                  sim_data.node_vars[name], # type: ignore
                                                  name)
        self._time_steps = sim_data.time

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps # type: ignore

    def sample(self, sample_points: np.ndarray,
               sample_times: np.ndarray | None = None
               ) -> np.ndarray:

        pv_points = pv.PolyData(sample_points)
        sample_data = pv_points.sample(self._data_grid)
        sample_data = np.array(sample_data[self._name]) # type: ignore

        if sample_times is None:
            return sample_data

        sample_func = lambda x: np.interp(sample_times,self._time_steps,x) # type: ignore
        sample_vals = np.apply_along_axis(sample_func,1,sample_data)

        return sample_vals

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._data_grid


def convert_simdata_to_pyvista(sim_data: mh.SimData, dim: int = 3
                               ) -> pv.UnstructuredGrid:

    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    if sim_data.connect is None:
        raise RuntimeError("SimData does not have a connectivity table, unable to convert to pyvista")

    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape

        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)

        this_cell_type = get_cell_type(nodes_per_elem,dim=dim)
        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)


    cells = flat_connect
    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)

    return pv_grid


def attach_field_to_pyvista(pv_grid: pv.UnstructuredGrid,
                                 node_field: np.ndarray,
                                 name: str) -> pv.UnstructuredGrid:
    pv_grid[name] = node_field
    return pv_grid


def get_cell_type(nodes_per_elem: int, dim: int = 3) -> int:
    cell_type = 0

    if dim == 2:
        if nodes_per_elem == 4:
            cell_type = CellType.QUAD
        elif nodes_per_elem == 3:
            cell_type = CellType.TRIANGLE
        else:
            cell_type = CellType.QUAD
    else:
        if nodes_per_elem == 8:
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
        else:
            cell_type = CellType.HEXAHEDRON

    return cell_type

