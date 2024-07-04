import numpy as np
import pyvista

point_cloud = np.random.random((100, 3))
pdata = pyvista.PolyData(point_cloud)
#pdata['orig_sphere'] = np.arange(100)

# create many spheres from the point cloud
sphere = pyvista.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
pc = pdata.glyph(scale=False, geom=sphere, orient=False)
pc.plot(cmap='Reds')