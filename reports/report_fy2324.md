# Digital-Physical Interface Tools: Sensor Simulation Engine

Lloyd Fletcher

## Introduction

### Motivation & Impact
Qualification of fusion technology is reliant on simulations to predict the performance of components in extreme (e.g., thermal and electromagnetic) and untestable (e.g., fusion neutron fluxes) environments. Enabling the use of simulations for design qualification requires that they are validated with experimental data over testable domains to reduce uncertainty when extrapolating into irradiated conditions. A key set of tools for simulation validation are the statistical metrics used to assess the agreement between the model and experimental data. High agreement between a model and experimental data increases the credibility of the model and confidence in decisions made based on model predictions.

Validation metrics must account for uncertainties (systematic and random) in the simulation as well as the experimental data. The Advanced Engineering Simulation (AES) Group in the Computing Division at UKAEA are developing the tools necessary to produce probabilistic simulation predictions accounting for uncertainties in model inputs such as geometry, material properties and boundary conditions/loads. The purpose of this project is to develop a software engine to simulate experimental data from a given model and use this to assess the impact of uncertainty in the experimental domain.

A software engine for simulating experimental data from simulations would has a wide variety of applications: 1) experiment design and sensor placement optimisation; 2) provide ground-truth data for benchmarking and developing validation metrics; and 3) testing the predictive capability of digital shadows/twins.  A digital shadow is connected to a real-world system and receives sensor data from the system to update the model state. A digital twin takes this one step further by acting on the model state and feeding back control signals into the real-world system. Both of these can be tested by connecting them to another simulation as a surrogate for the real-world system with a layer of software between that acts to simulate the sensor signals. It is then possible to perturb the surrogate real world system to model failure and then assess the predictive capability of the digital twin/shadow.

### Aims & Objectives

The aim of this project is to develop a software engine that can use an input multi-physics simulation to produce a set of simulated experimental data with realistic uncertainties. This software engine will be developed as a python package, as use of python provides access to a range of scientific computing, optimisation and machine learning libraries. The package will be called the python computer aided validation engine (`pycave`). The underlying engineering simulation tool is assumed to be the Multi-Physics Object Oriented Simulation Environment (MOOSE) being developed for fusion digital twin applications by the AES group. The objectives of this project are to create:

1. An overarching conceptual model of a sensor to be implemented as an abstract base class.
2. A module that contains a library of different sensors.
    Including different spatial arrangements (e.g. point, line, camera/area, volume).
    * Including increasing complexity of quantities of interest: scalar, vector & tensor.
    * A sensor array module that can manage a variety of sensors selected from the library.
3. An experiment simulation module that can take an input multi-physics simulation (from MOOSE) and a sensor array using this to generate a simulated experimental dataset.
4. A validation dataset factory module that can perturb an input multi-physics simulation (e.g. geometry, material model, boundary conditions) and apply a sensor array to create a series of ‘invalid’ datasets to assess validation metrics.

### Deliverables for Financial Year 2023-2024
The scope of the following deliverables were set based on the project starting half way through the year with an equivalent allocation of 0.75 FTE. All deliverables have been achieved and the results are detailed in this report.

1. A report detailing a development work plan for the package as well as full system specifications.
2. A flow chart for the package showing the key classes/functions and their relationships as well as external dependencies.
    * See the [flow chart](#flow-chart) below.
3. A first version of `mooseherder`, a package being developed to be able to run MOOSE simulations in parallel, which is required for objective 4 and 5 as well as other projects within AMT on test design and topology optimisation.
    * Source code for `mooseherder` v0.1: https://github.com/Applied-Materials-Technology/mooseherder
4. A first version of pycave demonstrated on the simplest test case of a scalar field with point sensors - specifically, simulated thermocouple data for a divertor monoblock simulation in MOOSE.
    * Source code for `pycave` prototype: https://github.com/Applied-Materials-Technology/pycave
    * A prototype demonstration is detailed in this report.

## Package Workflow & Flow Chart: `pycave`

The overall structure of the `pycave` package is shown in the figure below. The user inputs to the package include an output exodus file from the MOOSE simulation to be analysed; a list of sensor positions as a `numpy` array where each row is a sensor with position [x,y,z] in the simulation coordinates; and the optional parameter of the sample times at which sensor measurements should be simulated as a `numpy` array. If the user specifies sample times then the sensor values are linearly interpolated between the simulation time steps. Note that if the sample times are not specified they are assumed to coincide with the simulation time steps and no interpolation is performed.

The package has two main classes the first is a `Field` which interpolates the underlying simulation results to extract the ground truth values for the sensors at the specified times and locations. The second is a `SensorArray` which is an abstract base class (ABC) containing four key methods for simulating sensor output. The `ThermocoupleArray` is a concrete implementation of the `SensorArray` ABC that allows the user to extract measurements with simulated experimental errors.

It should be noted that the `mooseherder` package has a range of additional functionality which is not shown in the figure below as only the current dependencies for the `pycave` package are shown. This includes the ability to: 1) Dynamically update parameters in a MOOSE or Gmsh input file; 2) Run MOOSE simulations from python with Gmsh mesh generation; 3) Run a parameter sweep of a chain of MOOSE/Gmsh simulations in parallel; and 4) Read the output of the parameter sweep in parallel. This additional functionality of `mooseherder` is demonstrated in the worked examples in the `mooseherder` github repository.

|![fig_pycave_flowchart](images/pycave.drawio.svg)|
|:--:|
|*Figure: Overview of the current structure of `pycave` as applied to the modelling of thermocouples measuring a temperature field.*|

## Prototype Demonstration: `pycave`

This prototype demonstration will focus on the simplest case for sensor simulation which is point sensors used to measure a scalar field. For this purpose a temperature field measurements with thermocouples was chosen. In the future it is intended that `pycave` will be extended to simulate more complex sensors such as cameras and more complex fields such as vector (e.g. displacement) or tensor fields (e.g. strain).

Two MOOSE thermal simulations were constructed and run to demonstrate the functionality of the `pycave`. The input files for these simulations and associated output can be found [here](https://github.com/Applied-Materials-Technology/pycave/tree/main/data). The first simulation is based on a MOOSE tutorial problem analysing the thermal field in a 2D plate where the temperature is held constant on the left hand edge and then increased on the right hand edge with a user specified function. The second simulation is a 3D thermal model of a divertor monoblock armour component that includes three materials with temperature dependent material properties (tungsten, copper, copper-chromium-zirconium).

Here we use the code from the [first example](https://github.com/Applied-Materials-Technology/pycave/blob/main/examples/ex1_2d_thermcouples.py) in the repository to demonstrate the use of `pycave`. The first step is to import dependencies and use `mooseherder` to read the output exodus file for the simulation of interest:

```python
from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pycave

data_path = Path('data/plate_2d_thermal_out.e')
data_reader = mh.ExodusReader(data_path)
sim_data = data_reader.read_all_sim_data()
```

Now we need to construct the first input which is a `Field` object that will be used to interpolate the simulation data to the desired sensor positions.

```python
spat_dims = 2
field_name = 'temperature'
t_field = pycave.Field(sim_data,field_name,spat_dims)
```

The second required input is the locations of the sensors. Here we use a helper function from `pycave` to generate a uniform grid of sensor positions excluding edges. The sensor positions can also be manually specified where each row of the position `numpy` array is a sensor with position [x,y,z].

```python
n_sens = (3,2,1)
x_lims = (0.0,2.0)
y_lims = (0.0,1.0)
z_lims = (0.0,0.0)
sens_pos = pycave.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
```

Having created a `Field` object and the desired sensors positions a `ThermocoupleArray` can now be created. We also generate the standard uncertainty functions by specifying their parameters. The standard systematic error function is a randomly generated constant offset from a uniform distribution where the high and low bounds are specified. The standard random error function is a sampled from normal distribution at each sample time where the standard deviation is specified.

```python
tc_array = pycave.ThermocoupleArray(sens_pos,t_field)

tc_array.set_uniform_systematic_err_func(low=-10.0,high=10.0)
tc_array.set_normal_random_err_func(std_dev=5.0)
```

Custom functions for generating the systematic and random errors can be specified using the `ThermocoupleArray.set_custom_systematic_err_func()` and `ThermocoupleArray.set_custom_random_err_func()` functions. For this case the functions must take a single `size` parameter which will be the size of the array that will be returned which must be the same size as the measurement array. The measurement array has dimensions of [number of sensors, number of sample times]. The custom error function must return a `numpy` array of this size.

The measurements are constructed using: measurement = truth + systematic error + random error. Simulated measurements from the `ThermocoupleArray` can then be generated using:

```python
measurements = tc_array.get_measurements()
```

The measurements are returned as a `numpy` array with dimensions of [number of sensors, number of sample times].


```python
pv_sens = tc_array.get_visualiser()
pv_sim = t_field.get_visualiser()
pv_plot = pycave.plot_sensors(pv_sim,pv_sens,field_name)
pv_plot.add_scalar_bar('Temperature, T [degC]')
pv_plot.show()
```

|![fig_2dplate_pyvista](images/plate_thermal_2d_sim_view.svg)|
|:--:|
|*Figure: Simple 2D thermal plate visualised with pyvista showing thermocouple location over the simulated temperature field for the last time step.*|

```python3
(fig,ax) = tc_array.plot_time_traces(plot_truth=True)
plt.show()
```

|![fig_2dplate_traces](images/plate_thermal_2d_traces.png)|
|:--:|
|*Figure: Simulated thermocouple traces for the 2D plate model with sensor locations shown in the previous figure. The simulated traces use dashed lines with crosses abd include systematic and random error models. The solid lines are the ground truth taken from the simulation.* |


|![fig_3dmonoblock_pyvista](images/monoblock_thermal_sim_view.svg)|
|:--:|
|*Figure: Monoblock 3D thermal model visualised with pyvista showing thermocouple location over the simulated temperature field for the last time step.*|


|![fig_3dmonoblock_traces](images/monoblock_thermal_traces.png)|
|:--:|
|*Figure: Simulated thermocouple traces for the 3D monoblock model with sensor locations shown in the previous figure. The simulated traces use dashed lines with crosses and include the systematic and random error models. The solid lines are the ground truth taken from the simulation.* |


## Software Specification


**Inputs**

**Outputs**


## Digital-Physical Interface Plan for Financial Year 2024-2025


### Resources


### Deliverables for Financial Year 2024-2025


