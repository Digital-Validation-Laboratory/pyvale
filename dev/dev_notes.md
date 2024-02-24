# Notes: pycave developement

## Notes: architecture

## Notes: numpy
Avoiding loops by applying a function along an array dimension:

## Notes: pyvista

### Mesh structure in pyvista
NOTE: pyvista element numbers are 0 indexed so need to -1 from whole element connectivity table to correct it.

The most general mesh structure in `pyvista` is the `pyvista.UnstructuredGrid`. This requires 3 types of information to construct. Here we assume we have a mesh with `N` nodes and `E` elements with `n_per_e` nodes per element which:
 1. A flattened connectivity array (`cells`) of the form `[n_per_e,n_1,n_2,n_3,...]` repeated for each element.
 2. A flattened array that specifies the element types (`cell_types`) as an array of length `E` with the `py.CellType` as the value for each element.
 3. An array specifying the nodal coordinates as an `Nx3` array of nodal coordinates.

Example multi element mesh:
```python
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
```

### Plotting in pyvista
Adding multiple meshes to one plot:
```python
import pyvista as pv

pv.set_plot_theme('dark')
pv_plot = pv.Plotter(window_size=[1000, 1000]) # type: ignore
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
```

### Interpolating using an FE mesh
Syntax is:
```python
sampled_values = points_to_sample.sample(fe_field_data_to_sample)
```