# 2D DIC using Blender

## What is Blender and how is it used
Blender is an open source computer graphics software. 
It can be used to render images through both rasterisation and rendering.
Blender can be automated using the blender-python API `bpy`.


## How is the mesh imported
Information from the SimData object can be extracted and imported as a mesh - and subsequently used to deform the object.
Blender only reads surface meshes, so any 3D meshes must be skinned prior to importing.
The SimData mesh must be converted to .obj format to be imported into Blender - as it supports both quad and triangle meshes. 

## Defining camera/lighting parameters
The intrinsic lighting and camera parameters can be precisely defined. 
The camera currently being modelled is an AV Alvium 1800 U-507:
- Pixel dimensions: 2452 x 2056
- Pixel size: 3.45 um

It should be noted that when calculating the Field Of View, Blender uses a slightly simplified pinhole camera model, so the FOV given by Blender is slightly different than expected. 
Four types of lighting can be used:
- Point 
- Sun
- Spot 
- Area

A point source light is currently being used, with an `energy` of ...........

These parameters can be altered to accurately reflect the experimental setup being used. 

## Rigid body motion images
A set of images of an object with a speckle pattern applied, undergoing rigid body motion were produced. 
In-plane rigid body motion between 0 and 1 pixel was applied to the object, and it was imaged.

![Example rigid body motion image](./rendered_images/RBM_x/rigid_body_motion_x_9.tiff)
*An example rigid body motion image*

These images were run through MatchID to compare the MatchID calculated displacement with the imposed displacement  

|Imposed displacement | MatchID displacement |
| :-----------------: | :------------------: |
| 0.1                 |                      |
| 0.2                 |
| 0.3                 |
| 0.4                 |
| 0.5                 |
| 0.6                 |
| 0.7                 |
| 0.8                 |
| 0.9                 |
| 1.0                 |


## Deformation images
The same mesh object was used to test deformation using Blender.  
The displacements calculated from a MOOSE simulation were taken from the SimData object and applied to the part at subsequent timesteps. 

The ray tracing render engine Cycles was used to render the images.  

![Example deformation image](./rendered_images/case23_deformed_cycles/def_sim_data_10.tiff)
*An example deformed image*  

These images were run through MatchID to compare the calculated displacements to those imposed on the part.  
The initial image was also run through an image deformation with the simulation, to make the comparison more accurate to the use of Blender as opposed MatchID's introduction of error.  

![Example image deformation image](./rendered_images/case23_image_deformation/defimage_0010.tiff)
*An example image deformation image*  

#### Comparison of displacements
|Timestep |Simulation |Blender |Image deformation |
|:---:|---|---|---|
| 1 | 4.45283025e-15 |   |   |
| 2 |   |   |   |
| 3 |   |   |   |
| 4 |
| 5 |
| 6 |
| 7 |
| 8 |
| 9 |
| 10|


