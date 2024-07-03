# Notes: `pyvale` developement

## TODO: `pyvale`

**General**
- Future: Allow user to specify sideset to locate sensors

**Examples**
- Restructure examples to make things a bit easier to understand
- Controlling plot options
- Setting sensor temporal sampling
- Post systematic error setting: digitisation and saturation
- The random error library
- The systematic error library pre and post

**Systematic error handling**
- Add spatial and temporal averaging errors:
    - Based on the `Field` object
- Add positioning errors to the sensors
    - Based on the `Field` object
- Add calibration error

**Future: Experiment Simulation & Workflow**
- Need an simulated experiment generator with mooseherder
    - Build monoblock models of increasing fidelity (thermal -> thermo-mech, single value mat props -> temp dependence)
    - Monte Carlo or Latin Hypercube sampling
    - Experiments that just perturb one parameter
        - Material Properties
        - Load and BCs
        - Geometry
    - Look at sensitivity maps? - how does this work with a geometric perturbation
    - Start with the purely thermal case with thermocouples


## Python coding pinciples:
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

## `pyvale` architecture
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



## Sensors

A sensor should have:
- A spatial measurement geometry: point, line, area, volume
- A measurement position, centroid of the measurement geometry
- A measurement point/area/line/volume
- A measurement frequency in time
- A calibration curve
- A digitisation

### Thermocouples
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

