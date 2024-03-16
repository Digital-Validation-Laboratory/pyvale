# NOTES: gmsh

OpenCASCADE Tutorials: 16,18,19,20

## Geometry

**New**
Analogously to ‘newp’, the special
variables ‘newc’, ‘newcl, ‘news’, ‘newsl’ and ‘newv’ select new curve, curve loop, surface, surface loop and volume tags.

## Controlling Mesh Options

**Control min/max mesh size**
Mesh.MeshSizeMin = 0.001;
Mesh.MeshSizeMax = 0.3;

**Control the mesh algorithm**
Mesh.Algorithm = #

**Creating higher order meshes**
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;

**Mesh only part of the model**
- Note change the volume number to make visible
Hide {:}
Recursive Show { Volume{129}; }
Mesh.MeshOnlyVisible=1;