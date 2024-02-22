'''
-------------------------------------------------------------------------------
pycave: dev_main

authors: thescepticalrabbit
-------------------------------------------------------------------------------
'''
from pathlib import Path
from mooseherder import ExodusReader
from mooseherder import SimData
import numpy as np
import pyvista as pv
from pyvista import CellType


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


def main() -> None:
    data_path = Path('data/monoblock_thermal_out.e')

    data_reader = ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)
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
    #cell_types = np.full(elem_count,CellType.HEXAHEDRON)
    points = sim_data.coords

    grid = pv.UnstructuredGrid(cells, cell_types, points)

    for nn in sim_data.node_vars:
        grid[nn] = sim_data.node_vars[nn][:,-1]

    print(grid)
    grid.plot(show_edges=True)



if __name__ == '__main__':
    main()

