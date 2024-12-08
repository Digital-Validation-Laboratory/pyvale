# Notes: `pyvale` developement

## TODO: `pyvale`
- TESTING/FEATURE EXAMPLES:
    - Chaining field errors
    - Area averaging
    - Sensor angles
    - Visualisation tools
    - Camera basic

- BUGS!
    - Higher order mesh node numbering conversion from Exodus to VTK!
    https://github.com/Applied-Materials-Technology/pycoatl/blob/main/src/pycoatl/spatialdata/importsimdata.py
    - Spatial averaging with rectangle or quadrature makes assumptions about sensor orientation - looks like it assumes XY orientations only. Check this.
    - Revert to python3.11 for compatibility with blender python

- TODO PRIORITY:
    - Create option to specify single rotation for all point sensors - links to camerabasic
        - Build Rory's cimple DIC strain filter on top of the basic camera
    - Visualisation tools for perturbed field errors:
        - Angle
    - Visualisation tools for animating sensor traces
    - Visualisation tools for subplots of multiple sensors?
    - Finish basic camera

- TODO: EXAMPLES
    - Example showing calibration errors
    - Example showing area averaging as ground truth
        - With and without area averaging as follow up error
    - Example showing field error chain with other errors and extraction of perturbed sensor data
    - Example showing a basic camera

- TODO: ErrorIntegrator
    - Simplify the memory efficient and non-memory efficient options

- TODO: visualisation tools for:
    - TODO: remove methods from sensor descriptor data class
    - TODO: presentation animations - animate traces
    - TODO: experiment - allow extraction of different conditions for comparison plots
    - TODO: visualise all components of vector / tensor field
        - See this example for subplots: https://docs.pyvista.org/examples/02-plot/cmap#sphx-glr-examples-02-plot-cmap-py

- TODO: Field based errors:
    - TODO: Temporal averaging error
        - Set an integration time
        - Set a weighting function

- IMAGE DEF: allow upsampled image to be generated once and retained.

- CAMERAS:
    - Need CameraData class
    - Create 'CameraBasic' class or Simple?
    - Create 'CameraRayTracing' class
    - Create 'CameraIRThermo'
    - Create 'CameraDIC2D'
    - Create 'CameraDICStereo'
    - Allow field class to do a single field rotation into camera coords on creation and store this
    - Allow all sensors to have

- TESTING:
    - Need to check rotations are consistent

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

Gauss Quadrature for the Unit Disc
http://www.holoborodko.com/pavel/numerical-methods/numerical-integration/cubature-formulas-for-the-unit-disk/

## Pyvista Cameras
Tested on monoblock sim:
cpos = xy
[(0.0, 16.0, 90.80825912395183),
    (0.0, 16.0, 5.5),
    (0.0, 1.0, 0.0)]

then, azimuth = 45
[(60.32204851776551, 16.0, 65.8220485177655),
(0.0, 16.0, 5.5),
(0.0, 1.0, 0.0)]

then, zoom = 0.5
[(60.32204851776551, 16.0, 65.8220485177655),
(0.0, 16.0, 5.5),
(0.0, 1.0, 0.0)]

Start with xy then azimuth 90
[(85.30825912395183, 16.0, 5.5000000000000195),
(0.0, 16.0, 5.5),
(0.0, 1.0, 0.0)]

