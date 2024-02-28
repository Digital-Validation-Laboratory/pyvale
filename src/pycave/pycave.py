'''
================================================================================
pycave: mono repo

authors: thescepticalrabbit
================================================================================
'''
from abc import ABC, abstractmethod
from typing import Callable
from functools import partial

import numpy as np
import pyvista as pv
from pyvista import CellType

import mooseherder as mh


'''
================================================================================
'''
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

'''
================================================================================
'''
def attach_field_to_pyvista(pv_grid: pv.UnstructuredGrid,
                                 node_field: np.ndarray,
                                 name: str) -> pv.UnstructuredGrid:
    pv_grid[name] = node_field
    return pv_grid


'''
================================================================================
'''
def get_cell_type(nodes_per_elem: int, dim: int = 3) -> int:
    cell_type = 0
    if dim == 3:
        if nodes_per_elem == 8:
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.QUAD
        elif nodes_per_elem == 3:
            cell_type = CellType.TRIANGLE
    return cell_type


'''
================================================================================
'''
class Field:
    def __init__(self, sim_data: mh.SimData, name: str, dim: int = 3) -> None:
        self._name = name
        self._data_grid = convert_simdata_to_pyvista(sim_data,dim)
        self._data_grid = attach_field_to_pyvista(self._data_grid,
                                                  sim_data.node_vars[name], # type: ignore
                                                  name)
        self._time_steps = sim_data.time

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def sample(self, sample_points: np.ndarray) -> np.ndarray:
        pv_points = pv.PolyData(sample_points)
        sample_data = pv_points.sample(self._data_grid)
        return np.array(sample_data[self._name]) # type: ignore

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._data_grid


'''
================================================================================
'''
class SensorArray(ABC):
    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_measurements(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_truth_values(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_systematic_errs(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_random_errs(self) -> np.ndarray:
        pass


'''
================================================================================
'''
class ThermocoupleArray(SensorArray):
    def __init__(self,
                 positions: np.ndarray,
                 field: Field) -> None:

        self._positions = positions
        self._field = field

        self._rand_err_func = None
        self._sys_err_func = None
        '''
        self._rand_err_func = partial(np.random.default_rng().normal,
                                      loc=0.0,
                                      scale=1.0)
        self._sys_err_func = partial(np.random.default_rng().uniform,
                                     low=-1.0,
                                     high=1.0)
        '''

    def get_positions(self) -> np.ndarray:
        return self._positions


    def get_num_sensors(self) -> int:
        return self._positions.shape[0]


    def get_measurement_shape(self) -> tuple[int,int]:
        return (self.get_num_sensors(),
                self._field.get_time_steps().shape[0])


    def get_measurements(self) -> np.ndarray:
        return self.get_truth_values() + \
            self.get_systematic_errs() + \
            self.get_random_errs()


    def get_truth_values(self) -> np.ndarray:
        return self._field.sample(self._positions)


    def get_systematic_errs(self) -> np.ndarray:
        if self._sys_err_func is None:
            return np.zeros(self.get_measurement_shape())

        return self._sys_err_func(size=self.get_measurement_shape())


    def set_random_err_func(self, rand_fun: Callable | None) -> None:
        pass


    def get_random_errs(self) -> np.ndarray:
        if self._rand_err_func is None:
            return np.zeros(self.get_measurement_shape())

        return self._rand_err_func(size=self.get_measurement_shape())


    def get_visualiser(self) -> pv.PolyData:
        return pv.PolyData(self._positions)


'''
================================================================================
'''
def plot_sensors(pv_simdata: pv.UnstructuredGrid,
                 pv_sensdata: pv.PolyData) -> None:
    #pv.set_plot_theme('dark') # type: ignore
    pv_plot = pv.Plotter(window_size=[1000, 1000]) # type: ignore
    pv_plot.add_mesh(pv_sensdata,
                     label='sensors',
                     color='red',
                     render_points_as_spheres=True,
                     point_size=20
                     )

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata['temperature'][:,-1],
                     label='sim data',
                     show_edges=True)
    #pv_plot.camera_position = 'zy'
    pv_plot.add_axes_at_origin(labels_off=True)
    pv_plot.set_scale(xscale = 100, yscale = 100, zscale = 100)
    pv_plot.show()
