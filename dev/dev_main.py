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
def convert_simdata_to_pyvista(sim_data: mh.SimData,dim: int = 3
                               ) -> pv.UnstructuredGrid:
    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

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
    grid = pv.UnstructuredGrid(cells, cell_types, points)

    for nn in sim_data.node_vars:
        grid[nn] = sim_data.node_vars[nn]#[:,-1]

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

    def get_measurement(self) -> np.ndarray:
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
def plot_sensors(pv_simdata: pv.UnstructuredGrid,
                 pv_sensdata: pv.PolyData) -> None:
    pv.set_plot_theme('dark')
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

#-------------------------------------------------------------------------------
def main() -> None:
    data_path = Path('data/monoblock_thermal_out.e')
    #data_path = Path('data/moose_2d_thermal_basic_out.e')

    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    pv_simdata = convert_simdata_to_pyvista(sim_data,dim=3)

    x_sens = 1
    x_min = 11.5e-3
    x_max = 11.5e-3

    y_sens = 4
    y_min = -11.5e-3
    y_max = 19.5e-3

    z_sens = 3
    z_min = 0
    z_max = 12e-3

    sens_pos_x = np.linspace(x_min,x_max,x_sens+2)[1:-1]
    sens_pos_y = np.linspace(y_min,y_max,y_sens+2)[1:-1]
    sens_pos_z = np.linspace(z_min,z_max,z_sens+2)[1:-1]
    (sens_grid_x,sens_grid_y,sens_grid_z) = np.meshgrid(
        sens_pos_x,sens_pos_y,sens_pos_z)

    sens_pos_x = sens_grid_x.flatten()
    sens_pos_y = sens_grid_y.flatten()
    sens_pos_z = sens_grid_z.flatten()
    sens_pos = np.vstack((sens_pos_x,sens_pos_y,sens_pos_z)).T

    sensor_array = np.apply_along_axis(build_sensor,1,sens_pos)

    pv_sensdata = pv.PolyData(sens_pos)

    sens_vals = pv_sensdata.sample(pv_simdata)

    pprint(sens_vals)
    for nn in sim_data.node_vars:
        pprint(nn)
        pprint(sens_vals[nn])
        pprint(sens_vals[nn].shape)



    #plot_sensors(pv_simdata,pv_sensdata)




#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

