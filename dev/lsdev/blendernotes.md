# Blender notes

## Creating a scene
Refer to [speckle](https://github.com/AleksanderMarek/speckle)

Has information about how to load from FE mesh and function about deforming speckle pattern

## Rendering images
Refer to [blender-sdg](https://github.com/federicoarenasl/blender-sdg)

Also this [Synthetic_Data_Blender](https://github.com/matthieu-sgi/Synthetic_Data_Blender/tree/main) - not as good

## Running Blender from python
Refer to [bpy-gallery](https://github.com/kolibril13/bpy-gallery?tab=readme-ov-file)

Can install bpy from PyPi and run Blender headless

To visualise renders, save renders and then open images using Image

In order to use bpy as python module, need to run python on 3.11.*



## Loading mesh into Blender
Best way is to probably use [`mesh.from_pydata(vertices, edges, faces, shade_flat=True)`](https://docs.blender.org/api/current/bpy.types.Mesh.html#bpy.types.Mesh.from_pydata)

Can try 


