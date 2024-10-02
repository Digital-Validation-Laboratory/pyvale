# Notes: `pyvale` developement

## TODO: `pyvale`
NOTE: spatial averaging with rectangle or quadrature makes assumptions about sensor orientation - looks like it assumes XY orientations only. Check this.

- TODO: EXAMPLES
    - Example showing area averaging as ground truth

- TODO: Experiment generator/ runner
    - TODO: Allow user to extract all sources of error for each experiment, need to dig out of `ErrorIntegrator`
    - TODO: Create example connecting to `mooseherder`, assume user provides `mooseherder` like array of simulations
    - TODO: Increase plotting capabilities to compare over simulations as well as all sensors on experiments

- TODO: visualisation tools for:
    - TODO: presentation animations - create pyvista animation synced to matplotlib traces
    - TODO: visualisation of perturbed time / sensor locations
    - TODO: experiment - allow extraction of different conditions for comparison plots

- TODO: Calibration errors

- TODO: Field based errors:
    - HALF DONE: Spatial averaging error
        - Set an integration area
        - Set a weighting function
    - TODO: Temporal averaging error
        - Set an integration time
        - Set a weighting function
    - **TODO Allow Gauss Quad as Truth with other as Err**
    - TODO: Allow Gauss Quad with position and temporal drift

- IMAGE DEF: allow upsampled image to be generated once and retained.

Gauss Quadrature for the Unit Disc
http://www.holoborodko.com/pavel/numerical-methods/numerical-integration/cubature-formulas-for-the-unit-disk/

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
where V is voltage in units of millivolts and T is temperature in degrees C.

- Thermocouple amplifier card:
https://www.ni.com/docs/en-US/bundle/ni-9213-specs/page/specs.html

- App note on thermocouple amplifier chip:
https://www.analog.com/en/resources/app-notes/an-1087.html

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

## Numba
https://www.youtube.com/watch?v=6oXedk2tGfk

## Gauss Quadrature

## Gauss Quadrature: Change of Interval
https://stackoverflow.com/questions/33457880/different-intervals-for-gauss-legendre-quadrature-in-numpy

To change the interval, translate the x values from [-1, 1] to [a, b] using, say,

t = 0.5*(x + 1)*(b - a) + a

and then scale the quadrature formula by (b - a)/2:

gauss = sum(w * f(t)) * 0.5*(b - a)
