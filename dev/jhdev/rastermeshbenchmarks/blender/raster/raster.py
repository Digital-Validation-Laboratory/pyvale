'''
================================================================================
Example: Blender raster example

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale
import pyvista as pv
import bpy
import mathutils
import numpy as np
import time


def main() -> None:

    ##################
    # Testing Blender
    ##################

    time_import_start = time.perf_counter()
    # 3D cylinder, mechanical, tets
    data_path = Path("dev/lfdev/rastermeshbenchmarks")
    data_path = data_path / "case21_m5_out.e"

    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_keys = tuple(sim_data.node_vars.keys())
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0


    field_key = "disp_y"
    components = ("disp_x","disp_y","disp_z")
    (pv_grid,_) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                components,
                                                spat_dim=3)
    pyvale.print_dimensions(sim_data)


    pv_surf = pv_grid.extract_surface()

    time_import_end = time.perf_counter()
    time_import = time_import_end - time_import_start
    print(f"{'Time taken (Import and create mesh):':45}" + f"{time_import:.5f}" + " [s]")

    pv_surf.save("./dev/jhdev/rastermeshbenchmarks/blender/raster/case21_m5_out.stl")


    # parameters
    rot_axis = "x"
    phi_y_degs = -45
    theta_x_degs = -45
    psi_z_degs = 0.0
    phi_y_rads = np.deg2rad(phi_y_degs)
    theta_x_rads = np.deg2rad(theta_x_degs)

    # Simulated world coordinates data (example data for sim_data.coords)
    roi_pos_world = np.mean(sim_data.coords, axis=0)
    print(roi_pos_world)

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

    # clear existing blender objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  # Clear the scene

    # Import the STL file
    bpy.ops.wm.stl_import(filepath="./dev/jhdev/rastermeshbenchmarks/blender/raster/case21_m5_out.stl")

    # Add a light source
    bpy.ops.object.light_add(type='POINT', location=(20, 40, 20))
    light = bpy.context.object
    light.data.energy = 10000.0

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


    # create a scene 
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = cam_num_px[0]
    scene.render.resolution_y = cam_num_px[1]
    scene.render.pixel_aspect_x = pixel_size[0]
    scene.render.pixel_aspect_y = pixel_size[1]
    scene.eevee.taa_render_samples = 4       # Samples for final render

   
    # Render the scene to an image
    bpy.context.scene.render.filepath = "./dev/jhdev/rastermeshbenchmarks/blender/raster/case21_m5_out.png"
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
