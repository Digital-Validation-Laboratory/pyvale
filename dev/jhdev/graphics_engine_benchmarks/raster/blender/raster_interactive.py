import bpy
import numpy as np
from scipy.spatial.transform import Rotation
import mathutils
import time


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()  # Clear the scene

rot_axis = "x"
phi_y_degs = -45
theta_x_degs = -45
psi_z_degs = 0.0
phi_y_rads = np.deg2rad(phi_y_degs)
theta_x_rads = np.deg2rad(theta_x_degs)

# Hard coded coordinates.
roi_pos_world = [1.14642740e-02, 1.25546830e+01, 4.91564283e-02]

cam_num_px = np.array([2464, 2056], dtype=np.int32)
pixel_size = np.array([3.45e-3, 3.45e-3])  # in millimeters
focal_length_mm = 25.0
imaging_rad = 100.0

# Camera position calculations
xx, yy = 0, 1
cam_pos_world = np.array([
roi_pos_world[xx],
roi_pos_world[yy] - imaging_rad * np.sin(theta_x_rads),
imaging_rad * np.cos(theta_x_rads)
])

cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, theta_x_degs], degrees=True)

# Set up scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()  # Clear the scene

# Import the STL file
bpy.ops.wm.stl_import(filepath="/home/kc4736/pyvale/dev/jhdev/blender/rastermeshbenchmarks/case21_m1_out.stl")

# Add a light source
bpy.ops.object.light_add(type='POINT', location=(20.0, 20.0, 20.0))
light = bpy.context.object

# Calculate sensor size (width and height)
sensor_width = cam_num_px[0] * pixel_size[0]
sensor_height = cam_num_px[1] * pixel_size[1]

# Add a camera
camera_data = bpy.data.cameras.new(name="test_cam")
camera = bpy.data.objects.new("test_cam", camera_data)
bpy.context.collection.objects.link(camera)
camera.location = cam_pos_world

rotation_matrix = cam_rot.as_matrix()
camera.matrix_world = mathutils.Matrix((
(rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], cam_pos_world[0]),
(rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], cam_pos_world[1]),
(rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], cam_pos_world[2]),
(0, 0, 0, 1)
))

 # Set camera properties
camera.data.lens = focal_length_mm  # Set focal length
camera.data.sensor_width = sensor_width # Convert to millimeters for Blender
camera.data.sensor_height = sensor_height  # Convert to millimeters for Blender
print("Added Camera")

# create a scene 
scene = bpy.context.scene
scene.camera = camera
scene.render.resolution_x = cam_num_px[0]
scene.render.resolution_y = cam_num_px[1]
scene.render.pixel_aspect_x = pixel_size[0]
scene.render.pixel_aspect_y = pixel_size[1]
scene.eevee.taa_render_samples = 1

# Render the scene to an image
print("raster starting")
bpy.context.scene.render.filepath = "/home/kc4736/pyvale/dev/jhdev/blender/rastermeshbenchmarks/case21_m1_out_interactive.png"
bpy.ops.render.render(write_still=True)
print("raster ending")
