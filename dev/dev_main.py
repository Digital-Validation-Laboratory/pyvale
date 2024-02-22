'''
-------------------------------------------------------------------------------
pycave: dev_main

authors: thescepticalrabbit
-------------------------------------------------------------------------------
'''
from pprint import pprint
from pathlib import Path
from mooseherder.exodusreader import ExodusReader
import numpy as np
import pyvista as pv
from pyvista import CellType



def main() -> None:
    # NOTE: need to -1 from element numbers in connectivity

    data_path = Path('data/monoblock_thermal_out.e')

    data_reader = ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    #sim_data.connect.pop('connect2')
    #sim_data.connect.pop('connect3')

    flat_connect = np.array([],dtype=np.int64)
    elem_count = 0
    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape
        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)
        elem_count += n_elems

    cells = flat_connect
    print(flat_connect)
    print(temp_connect)
    cell_types = np.full(elem_count,CellType.HEXAHEDRON)
    points = sim_data.coords

    print(cells.shape)
    print(cell_types.shape)
    print(points.shape)

    grid = pv.UnstructuredGrid(cells, cell_types, points)

    for ii,nn in enumerate(sim_data.node_vars):
        grid[nn] = sim_data.node_vars[nn][:,-1]

    print(grid)
    grid.plot(show_edges=True)

    #pv_data.point_data['T'] = sim_data.node_vars['T'][:,-1]

    # Plot the temperature field
    #print(pv_data)
    #pv_data.plot()




if __name__ == '__main__':
    main()

