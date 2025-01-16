# `pyvale` planning

I have conceptually split `pyvale` into three seperate parts:
1. Sensor simulation engine & uncertainty quantification
2. Experimental design & sensor placement optimisation
3. Simulation calibration & validation

I think the bulk of the work this year will be on "1. Sensor simulation engine" this year. Specifically on how we implement cameras. There are two main approaches to this:

1. Rasterisation
2. Ray tracing
3. DIC engine

- SoftwareX

There are also several ways we can implement the above:

1. Write our own graphics engine in:
    - Python / numpy
    - Python but compiled with Numba or similar
    - Cython
    - C and link it in to python using Cython
    - Pushing things onto the GPU
2. Use Blender:
    - Has in-built rasterisation and ray tracing
    - Might be overkill for our simplified scenes where we are more interested in projecting an accurate physics simulation onto an image than hyper realistic images.

Challenges:
1. Dealing with finite element meshes where we have quads and higher order elements(quadratic elements) rather than linear triangle meshes which are standard in computer graphics.
2. Getting bogged down in the various options above.

Questions:


**Pre-reading**
1. [Fellowship case for support](https://ukaeauk-my.sharepoint.com/:b:/g/personal/lloyd_fletcher_ukaea_uk/EcEctHw6whJDtgNPGusCGzIBG59EQnJxU_H7ZGmpx6mD2A?e=vR8mZv)
2. [Project justification document](https://ukaeauk-my.sharepoint.com/:w:/g/personal/adel_tayeb_ukaea_uk/EXXWTVwCRV9Njyp8NZBmwHYBdepyaOZlEL1CZmLFgxqi9g?e=Ijaytz) for the experimental part of the project.
3. `pyvale` [report](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/reports/report_fy2324.md) from last financial year.
4. Some recent research seminars I have delivered with information on `pyvale` [here](https://ukaeauk-my.sharepoint.com/:p:/g/personal/lloyd_fletcher_ukaea_uk/EcKOtd_2sEVBpodjaqMcxa8BdeItwBpcbG_1RNWDDnt7Iw?e=3UruqZ) and [here](https://ukaeauk-my.sharepoint.com/:p:/g/personal/lloyd_fletcher_ukaea_uk/EWmTKcCdBmJKnyFO6wZrH0MBqBwKDVu2k8IcQ4chy_O_Qw?e=9ikDLx).
5. [Project](https://github.com/orgs/Computer-Aided-Validation-Laboratory/projects/1/views/1) page for `pyvale` software developement. Here we will track tasks on the project - feel free to add to this as we go!

## Planning Notes
- `pyvale`
    - Logo?
    - Project website
- Coding and software engineering:
    - Python coding principles
    - Auto generated documentation / docstrings
    - Tutorials with Jupyter notebooks :(
    - Continuous integration, FUTURE?