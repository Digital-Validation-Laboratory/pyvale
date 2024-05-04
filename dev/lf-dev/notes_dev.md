# Notes: pyvale developement

# TODO
- Need an simulated experiment generator with mooseherder
    - Build monoblock models of increasing fidelity (thermal -> thermo-mech, single value mat props -> temp dependence)
    - Monte Carlo or Latin Hypercube sampling
    - Experiments that just perturb one parameter
        - Material Properties
        - Load and BCs
        - Geometry
    - Look at sensitivity maps? - how does this work with a geometric perturbation
    - Start with the purely thermal case with thermocouples

# MOOSE sim naming convention
- Need 2d vs 3d
- monoblock vs plate (monob, plate)
- Steady vs Transient (stead, trans)
- therm vs mech vs therm-mech
- mat props temp dependent vs not

# Deliverables FY 24-25
- Python coding pinciples.
    - Use git, vscode and pylint (PEP8)
    - Use descriptive variable names, no single letter variables
    - Avoid comments unless needed to explain something weird (like 1 vs 0 indexing) – the code and variable names should speak for themselves
    - Work in your own 'feature' branch, pull to 'dev' - don't push to main!
    - Type hint everything: e.g. 'def add_ints(a: int, b: int) -> int:'
    - Default mutable data types to None
    - Numpy is your friend - no for/while loops!
    - No inheritance unless it is an interface / ABC - use composition
    - Use a mixture of functions and classes with methods where they make sense
    - Write good docstrings when the code is ready for sharing – use auto docstring to help.
    - Use code reviews to help each other but be nice!

- Module: `ExperimentWorkflow`
    - Manages and builds the overall workflow

- Module: `Sampler`
    - Samples from various distributions using monte carlo or latin hypercube
    - Separates espitemic and aleatory errors? - might not be needed

- Module: `RandErrGenerator`= Enhanced uncertainty function generation for random errors focusing on point sensors including:
    - Specification of noise as a function/percentage of sensor measurement value

- Module: `SysErrGenerator` = Enhanced uncertainty function generation for systematic errors focusing on point sensors including:
    - Calibration errors
    - Digitisation errors
    - Positioning errors
    - Spatial averaging errors
    - Temporal averaging errors
    - Ability to collapse all of the above into a single empirical/probability density function

- Module: `SensorLibrary` = Developement of library sensor models.
    - ABC: `SensorArray`
    - Module: `ThermocoupleArray`
    - Module: `CameraSensor`= Developement of simplified camera sensor models for:
        - Infrared cameras measuring temperature fields
        - Digital image correlation measuring displacement field on a surface

- ABC: `Field` - might not be able to be an ABC because scalar vs vector is quite different
    - Module: `ScalarField`
    - Module: `VectorField`
    - **Ext**, Module: `TensorField`
    - **Ext**, How do these reconstruct fields from sparse values? e.g. using GPs

- Module: `Validator` = A toolbox for calculating validation metrics from sensor data (simulated or real)
    - Applicable to point sensors for thermal fields
    - **Ext** Applicable to camera sensors for thermal fields

- Testing: A software test suite for point sensor functionality after completion of the additional features.
- Documentation: and worked examples using the following test cases:
    - Thermo-mechanical analysis of a simple 2D plate
    - Thermo-mechanical analysis of a 3D divertor monoblock model
- Modules: `Calibrator` and `Optimiser`
    - Based on Adel's thermocouple optimiser
    - Optimiser wraps `pymoo`


# TODO

**General**
- Allow user to specify sideset to locate sensors
- Need to allow user to specify noise as a percentage of the sensor value like a COV
- Allow chaining/list of functions for sys/rand errors
- Split systematic and random error handlers into own objects?

**MOOSE Models**
- Rejig monoblock model:
    - Rebuild mesh in Gmsh
    - Simplify to single material properties instead of temp dependent
    - Include a tensor mechanics version
- Build KC4 STC as simple thermo-mechanical model
    - Build parametric mesh in Gmsh
    - Build simplified single material parameter models
        - Pure thermal
        - Thermo-mechanical

**Systematic error handling**
- Probably need to split into its own class
- Add a sampling geometry to the sensors to allow systematic error calculation
- Add positioning errors to the sensors
- Add digitisation / voltage / calibration errors

**Random error handling**
- Probably need to split into its own class

# Notes: sensors

A sensor should have:
- A spatial measurement geometry: point, line, area, volume
- A measurement position, centroid of the measurement geometry
- A measurement point/area/line/volume
- A measurement frequency in time
- A calibration curve
- A digitisation

## Notes: thermocouples

https://www.mstarlabs.com/sensors/thermocouple-calibration.html
T  =  -0.01897 + 25.41881 V - 0.42456 V^2 + 0.04365 V^3
where V is voltage in units of millivolts
and   T is temperature in degrees C

## Systematic errors

Comes from many sources:
- Positioning error
- Averaging over time (integration time)
- Averaging over space (integration space)
- Digistisation error
- Calibration error
- Temporal lag of the signal
- Drift


## Random errors

Comes from sensors noise, need a way to specify a probability distribution to sample from.

# Notes: numpy

Avoiding loops by applying a function along an array dimension: `apply_along_dimension`

Random number generation in numpy:
```python
import numpy as np

mu, sigma = 0, 0.1
s = np.random.default_rng().normal(mu, sigma, 1000)

s = np.random.default_rng().uniform(-1,0,1000)
```

# Notes: pyvista

## Mesh structure in pyvista

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

## Plotting in pyvista

https://docs.pyvista.org/version/stable/examples/02-plot/orbit#sphx-glr-examples-02-plot-orbit-py

https://docs.pyvista.org/version/stable/api/core/camera#cameras-api

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

## Misc pyvista plotting commands

```python
import pyvista as pv

pv.set_plot_theme('dark')

pv_plot = pv.Plotter(window_size=[1280, 800],off_screen=True)
pv_plot.add_axes_at_origin(labels_off=False)
pv_plot.set_scale(xscale = 100, yscale = 100, zscale = 100)
pv_plot.camera_position = 'xy'
pv_plot.camera.zoom(5)
pv_plot.show()
```

## Labels on pyvista plots

https://tutorial.pyvista.org/tutorial/03_figures/bonus/e_labels.html

```python
poly = pv.PolyData(np.random.rand(10, 3))
poly["My Labels"] = [f"Label {i}" for i in range(poly.n_points)]

plotter = pv.Plotter()
plotter.add_point_labels(poly, "My Labels", point_size=20, font_size=36)
plotter.show()
```

## Scalar bars on plots

https://docs.pyvista.org/version/stable/examples/02-plot/scalar-bars#sphx-glr-examples-02-plot-scalar-bars-py

```python
pv_plot.add_scalar_bar('Temperature, T [degC]')
```

## Picking from a plot

```python
sphere = pv.Sphere()

pl = pv.Plotter()
pl.add_mesh(sphere, show_edges=True, pickable=True)
pl.enable_element_picking(mode=ElementType.EDGE)

pl.camera_position = [
    (0.7896646029990011, 0.7520805261169909, 0.5148524767495051),
    (-0.014748048334009667, -0.0257133671899262, 0.07194025085895145),
    (-0.26016740957025775, -0.2603941863919363, 0.9297891087180916),
]

pl.show(auto_close=False)
```

```python
mesh = pv.Wavelet()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, pickable=True)
pl.enable_element_picking(mode=ElementType.FACE)

pl.camera_position = [
    (13.523728057554308, 9.910583926360937, 11.827103195167833),
    (2.229008884793069, -2.782397236304676, 6.84282248642347),
    (-0.17641568583704878, -0.21978122178947299, 0.9594653304520027),
]

pl.show(auto_close=False)

# Programmatically pick a face to make example look nice
try:
    width, height = pl.window_size
    pl.iren._mouse_right_button_press(419, 263)
    pl.iren._mouse_right_button_release()
except AttributeError:
    # ignore this section when manually closing the window
    pass

```

## Interpolating using an FE mesh

Syntax is:

```python
sampled_values = points_to_sample.sample(fe_field_data_to_sample)
```

## Camera positions for pyvista plots

https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.plotter.camera_position#pyvista.Plotter.camera_position

- Example 1: Plate
```
[(-0.295, 1.235, 3.369),
 (1.0274, 0.314, 0.0211),
 (0.081, 0.969, -0.234)]
```

## Saving images in pyvista

- Option 1: screenshot to png. Need to create a plotter with off_screen=True

```python
import pyvista as pv
pv_plot = pv.Plotter(off_screen=True)
# Add whatever mesh you want
pv_plot.show(screen_shot='save_image.png')
```

- Option 2: export to vector graphics as .svg .eps .ps .pdf .tex

```python
import pyvista as pv
pv_plot = pv.Plotter()
# Add whatever mesh you want
pv_plot.save_graphic('save_image.svg')
```

# Notes: pymoo

## GA: Genetic Alogorithm

```python
class GA(GeneticAlgorithm):
    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_and_fitness),
                 crossover=SBX(),
                 mutation=PM(),
                 survival=FitnessSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=SingleObjectiveOutput(),
                 **kwargs):

```

## PSO: Particle Swarm Optimisation

```python
class PSO(Algorithm):
    def __init__(self,
                 pop_size=25,
                 sampling=LHS(),
                 w=0.9,
                 c1=2.0,
                 c2=2.0,
                 adaptive=True,
                 initial_velocity="random",
                 max_velocity_rate=0.20,
                 pertube_best=True,
                 repair=NoRepair(),
                 output=PSOFuzzyOutput(),
                 **kwargs):
```

## Problem with constraints
```python
class SphereWithConstraint(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2, axis=1)
        out["G"] = 0.1 - out["F"]
```

## Parallelisation:
https://pymoo.org/problems/parallelization.html
Example:
```python
from pymoo.core.problem import Problem

pool = ThreadPool(8)

class MyProblem(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):

        # define the function
        def my_eval(x):
            return (x ** 2).sum()

        # prepare the parameters for the pool
        params = [[X[k]] for k in range(len(X))]

        # calculate the function values in a parallelized manner and wait until done
        F = pool.starmap(my_eval, params)

        # store the function values and return them.
        out["F"] = np.array(F)

problem = MyProblem()
res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)
pool.close()
```
