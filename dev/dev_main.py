'''
-------------------------------------------------------------------------------
pycave: dev_main

authors: thescepticalrabbit
-------------------------------------------------------------------------------
'''
from pprint import pprint
from pathlib import Path
import mooseherder as mh
import numpy as np
import pyvista as pv
from pyvista import CellType

#-------------------------------------------------------------------------------
def convert_simdata_to_pyvista(sim_data: mh.SimData,) -> pv.UnstructuredGrid:
    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape

        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)

        this_cell_type = get_cell_type(nodes_per_elem,dim=2)
        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)


    cells = flat_connect
    points = sim_data.coords
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    for nn in sim_data.node_vars:
        grid[nn] = sim_data.node_vars[nn]

    return grid

#-------------------------------------------------------------------------------
def get_cell_type(nodes_per_elem: int, dim: int = 3) -> int:
    cell_type = 0
    if dim == 3:
        if nodes_per_elem == 8:
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
    elif dim == 2:
        if nodes_per_elem == 4:
            cell_type = CellType.QUAD
        elif nodes_per_elem == 3:
            cell_type = CellType.TRIANGLE
    return cell_type

#-------------------------------------------------------------------------------
# Use pyvista.sample to use the FE mesh to do interpolation of a field

class Field:
    def __init__(self) -> None:
        pass



#-------------------------------------------------------------------------------
class Sensor:
    def __init__(self, position: np.ndarray) -> None:
        self._position = position

    def get_position(self) -> np.ndarray:
        return self._position

    def get_measurement(self,field) -> np.ndarray:
        return self.get_truth() + \
            self.get_systematic_err() + \
            self.get_random_err()

    def get_truth(self) -> np.ndarray:
        return np.array([])

    def get_systematic_err(self) -> np.ndarray:
        return np.array([])

    def get_random_err(self) -> np.ndarray:
        return np.array([])

#-------------------------------------------------------------------------------
def build_sensor(position: np.ndarray) -> np.ndarray:
    return np.array((Sensor(position)))

#-------------------------------------------------------------------------------
def main() -> None:
    data_path = Path('data/monoblock_thermal_out.e')
    data_path = Path('data/moose_2d_thermal_basic_out.e')

    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    pv_simdata = convert_simdata_to_pyvista(sim_data)

    # Create sensors
    x_sens = 3
    x_min = 0
    x_max = 2
    y_sens = 2
    y_min = 0
    y_max = 1

    sens_pos_x = np.linspace(x_min,x_max,x_sens+2)[1:-1]
    sens_pos_y = np.linspace(y_min,y_max,y_sens+2)[1:-1]
    (sens_grid_x,sens_grid_y) = np.meshgrid(sens_pos_x,sens_pos_y)

    sens_pos_x = sens_grid_x.flatten()
    sens_pos_y = sens_grid_y.flatten()
    sens_pos_z = np.zeros(sens_pos_x.shape)
    sens_pos = np.vstack((sens_pos_x,sens_pos_y,sens_pos_z)).T

    sensor_array = np.apply_along_axis(build_sensor,1,sens_pos)

    pv_sensdata = pv.PolyData(sens_pos)

    pv.set_plot_theme('dark')
    pv_plot = pv.Plotter(window_size=[1000, 1000])
    pv_plot.add_mesh(pv_sensdata,
                     label='sensors',
                     color='red',
                     render_points_as_spheres=True,
                     point_size=20
                     )

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata['T'][:,-1],
                     label='sim data',
                     show_edges=True)
    pv_plot.camera_position = 'xy'
    pv_plot.show()

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

