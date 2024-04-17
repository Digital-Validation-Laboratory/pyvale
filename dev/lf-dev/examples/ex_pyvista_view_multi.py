
import numpy as
import pyvista as pv

p = pv.Plotter(window_size=[1000, 1000])
for ind, solid in enumerate(solids):
    # only use smooth shading for the teapot
    smooth_shading = ind == len(solids) - 1
    p.add_mesh(
        solid, color='silver', smooth_shading=smooth_shading, specular=1.0, specular_power=10
    )
p.view_vector((5.0, 2, 3))
p.add_floor('-z', lighting=True, color='lightblue', pad=1.0)
p.enable_shadows()
p.show()