# Digital-Physical Interface Tools: `pyvale` a Digital Validation Engine

Lloyd Fletcher, Adel Tayeb, Alex Marsh.<br>

Applied Materials Technology Group, Fusion Technology Division,<br>
United Kingdom Atomic Energy Authority (UKAEA).

## Introduction for Financial Year 2024-2025

The deliverables for this project are outlined below as in the previous report:

## Deliverables for Financial Year 24-25

The scope of the deliverables for this project will be adjusted to take advantage of any synergies with other research projects throughout UKAEA such as digital shadow/twin work in the Advanced Engineering Simulation group or as part of EPSRC Key Challenge 4 on digital qualification. An initial proposal for core deliverables in the next financial year is given below.

**Core Deliverables:**

1. **COMPLETE**: Enhanced uncertainty function generation for random errors, with a focus on point sensors, including:
    - **COMPLETE**: Specification of noise as a function/percentage of sensor measurement value.
2. Enhanced uncertainty function generation for systematic errors, with a focus on point sensors, including:
    - **COMPLETE**: Calibration errors
    - **COMPLETE**: Digitisation errors
    - *In-progress*: Positioning errors
    - *In-progress*: Spatial averaging errors
    - *In-progress*: Temporal averaging errors
    - **COMPLETE**: Ability to collapse all of the above into a single function
3. Developement of library sensor models to include:
    - **COMPLETE**: Measurement of a vector field. *TODO*: accounting for sensor orientation
    - **EXTENSION/COMPLETE**: Measurement of tensor fields without support for sensor orientation
    - Developement of simplified camera sensor models for:
        - Infrared cameras measuring temperature fields
        - **COMPLETE**: Digital image correlation measuring displacement field on a surface
4. *In-progress*: A toolbox for calculating validation metrics from sensor data (simulated or real).
5. Software tests using `pytest` for point sensor functionality after completion of the additional features.
6. Automated documentation generation and worked examples using the following test cases:
    - Thermo-mechanical analysis of a simple 2D plate
    - Thermo-mechanical analysis of a 3D divertor monoblock model

**Extension Deliverables:**
- An application of `pyvale` to optimise placement of neutronics sensors for LIBRTI. Set as extension as dependent on provision of a relevant neutronics simulation.
- *In-progress*: A toolbox for simulation parameter calibration using optimisers from the multi-objective optimisation library `pymoo`.
- A journal article in SoftwareX detailing the implementation of the first version of `pyvale`.
- A journal article detailing the application of `pyvale` to the simulations and experimental data generated as part of the EPSRC Key Challenge 4 'simple test case'.