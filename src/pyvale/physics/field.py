'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np
import pyvista as pv
from pyvista import CellType

import mooseherder as mh


class FieldError(Exception):
    pass

class IField(ABC):
    @abstractmethod
    def set_sim_data(self,sim_data: mh.SimData) -> None:
        pass

    @abstractmethod
    def get_time_steps(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_visualiser(self) -> pv.UnstructuredGrid:
        pass

    @abstractmethod
    def get_all_components(self) -> tuple[str,...]:
        pass

    @abstractmethod
    def get_component_index(self,comp: str) -> int:
        pass

    @abstractmethod
    def sample_field(self,
                    sample_points: np.ndarray,
                    sample_times: np.ndarray | None = None
                    ) -> np.ndarray:
        pass


#-------------------------------------------------------------------------------
def conv_simdata_to_pyvista(sim_data: mh.SimData,
                            components: tuple[str,...] | None,
                            spat_dim: int) -> pv.UnstructuredGrid:

    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    if sim_data.connect is None:
        raise FieldError("SimData does not have a connectivity table, unable to convert to pyvista")

    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape

        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)

        this_cell_type = get_cell_type(nodes_per_elem,spat_dim)
        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)

    cells = flat_connect
    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)

    if components is not None and sim_data.node_vars is not None:
        for cc in components:
            pv_grid[cc] = sim_data.node_vars[cc]

    return pv_grid


def get_cell_type(nodes_per_elem: int, spat_dim: int) -> int:
    cell_type = 0

    if spat_dim == 2:
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


# NOTE: sampling outside the bounds of the sample returns a value of 0
def sample_pyvista(components: tuple,
                pyvista_grid: pv.UnstructuredGrid,
                time_steps: np.ndarray,
                sample_points: np.ndarray,
                sample_times: np.ndarray | None = None
                ) -> np.ndarray:

    pv_points = pv.PolyData(sample_points)
    sample_data = pv_points.sample(pyvista_grid)

    if sample_data is None:
        raise(FieldError("Sampling simulation data at sensors locations with pyvista failed."))

    n_comps = len(components)
    (n_sensors,n_time_steps) = np.array(sample_data[components[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_sim_time[:,ii,:] = np.array(sample_data[cc])

    if sample_times is None:
        return sample_at_sim_time

    sample_time_interp = lambda x: np.interp(sample_times,time_steps,x) # type: ignore

    n_time_steps = sample_times.shape[0]
    sample_at_spec_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_spec_time[:,ii,:] = np.apply_along_axis(sample_time_interp,-1,
                                                    sample_at_sim_time[:,ii,:])

    return sample_at_spec_time
