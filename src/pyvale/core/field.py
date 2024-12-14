'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
from pyvista import CellType

import mooseherder as mh


class IField(ABC):
    @abstractmethod
    def set_sim_data(self,sim_data: mh.SimData) -> None:
        pass

    @abstractmethod
    def get_sim_data(self) -> mh.SimData:
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
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:
        pass


#-------------------------------------------------------------------------------
def conv_simdata_to_pyvista(sim_data: mh.SimData,
                            components: tuple[str,...] | None,
                            spat_dim: int
                            ) -> tuple[pv.UnstructuredGrid,pv.UnstructuredGrid]:

    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

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
    pv_grid_vis = pv.UnstructuredGrid(cells, cell_types, points)

    if components is not None and sim_data.node_vars is not None:
        for cc in components:
            pv_grid[cc] = sim_data.node_vars[cc]

    return (pv_grid,pv_grid_vis)


def get_cell_type(nodes_per_elem: int, spat_dim: int) -> int:
    cell_type = 0

    if spat_dim == 2:
        if nodes_per_elem == 4:
            cell_type = CellType.QUAD
        elif nodes_per_elem == 3:
            cell_type = CellType.TRIANGLE
        elif nodes_per_elem == 6:
            cell_type = CellType.QUADRATIC_TRIANGLE
        elif nodes_per_elem == 7:
            cell_type = CellType.BIQUADRATIC_TRIANGLE
        elif nodes_per_elem == 8:
            cell_type = CellType.QUADRATIC_QUAD
        elif nodes_per_elem == 9:
            cell_type = CellType.BIQUADRATIC_QUAD
        else:
            warnings.warn(f"Cell type 2D with {nodes_per_elem} "
                          + "nodes not recognised. Defaulting to 4 node QUAD")
            cell_type = CellType.QUAD
    else:
        if nodes_per_elem == 8:
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
        elif nodes_per_elem == 10:
            cell_type = CellType.QUADRATIC_TETRA
        elif nodes_per_elem == 20:
            cell_type = CellType.QUADRATIC_HEXAHEDRON
        elif nodes_per_elem == 27:
            cell_type = CellType.TRIQUADRATIC_HEXAHEDRON
        else:
            warnings.warn(f"Cell type 3D with {nodes_per_elem} "
                + "nodes not recognised. Defaulting to 8 node HEX")
            cell_type = CellType.HEXAHEDRON

    return cell_type


# NOTE: sampling outside the bounds of the sample returns a value of 0
def sample_pyvista(components: tuple,
                pyvista_grid: pv.UnstructuredGrid,
                time_steps: np.ndarray,
                points: np.ndarray,
                times: np.ndarray | None = None
                ) -> np.ndarray:

    # Use pyvista and shape functions for spatial interpolation at sim times
    pv_points = pv.PolyData(points)
    sample_data = pv_points.sample(pyvista_grid)

    # Push into the measurement array, shape=(n_sensors,n_comps,n_time_steps)
    n_comps = len(components)
    (n_sensors,n_time_steps) = np.array(sample_data[components[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_sim_time[:,ii,:] = np.array(sample_data[cc])

    # If sensor times are sim times then we return
    if times is None:
        return sample_at_sim_time

    # Use linear interpolation to extract sensor times
    def sample_time_interp(x):
        return np.interp(times, time_steps, x)

    n_time_steps = times.shape[0]
    sample_at_spec_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_spec_time[:,ii,:] = np.apply_along_axis(sample_time_interp,-1,
                                                    sample_at_sim_time[:,ii,:])

    return sample_at_spec_time

