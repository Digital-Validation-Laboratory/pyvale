import numpy as np
import pyvista as pv
from pyvista import CellType

pv.set_plot_theme("document")

# Define points for the hexahedron and tetrahedron
points = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
        [0.75, 0.5, 1.25],  # 8
    ],
    dtype=float,
)

# Define hexahedral cell connectivity
hex_cell = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7], np.int64)

# Define tetrahedral cell connectivity
tet_cell = np.array([4, 4, 5, 6, 8], np.int64)

# Combine cell arrays
cells = np.hstack((hex_cell, tet_cell))

# Define cell types
cell_types = np.array([CellType.HEXAHEDRON, CellType.TETRA])
print(cell_types)

# Create unstructured grid
grid = pv.UnstructuredGrid(cells, cell_types, points)

# Visualize the unstructured grid
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True)
pl.show_axes()
pl.show()