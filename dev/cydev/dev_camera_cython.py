
"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import pyvale
import mooseherder as mh

from camerarasterdata import CameraRasterData
# CYTHON MODULE
import camerac

def main() -> None:
    print()
    print(80*"C")
    print("CYTHON FILE:")
    print(camerac.__file__)
    print(80*"C")

    # 3D cylinder, mechanical, tets
    data_path = Path("dev/lfdev/rastermeshbenchmarks")
    data_path = data_path / "case21_m1_out.e"

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

    time_start_setup = time.perf_counter()

    pv_surf = pv_grid.extract_surface()
    faces = np.array(pv_surf.faces)

    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face+1)]
    assert np.all(nodes_per_face_vec == first_elem_nodes_per_face), \
    "Not all elements in the simdata object have the same number of nodes per element"

    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face+1))

    # Reshape the faces table and slice off the first column which is just the
    # number of nodes per element and should be the same for all elements
    pv_connect = np.reshape(faces,(num_faces,nodes_per_face+1))
    pv_connect = pv_connect[:,1:].T
    pv_coords = np.array(pv_surf.points).T

    print()
    print(80*"-")
    print("EXTRACTED SURFACE MESH DATA")
    print(pv_surf)
    print()
    print("Attached array names:")
    print(pv_surf.array_names)
    print()
    print(f"{nodes_per_face=}")
    print(f"{num_faces=}")
    print()
    print("NOTE: shape needs to be coord/node_per_elem first.")
    print(f"{pv_coords.shape=}")
    print(f"{pv_connect.shape=}")
    print()
    print(f"{pv_surf[components[0]].shape=}")
    print()
    print(80*"-")

    #===========================================================================
    # Create Camera and World Parameters
    (xx,yy,zz,ww) = (0,1,2,3)

    #shape=(3,num_coords)
    coords_world = pv_coords
    coords_count = coords_world.shape[1]
    # shape=(4,num_nodes)
    coords_world= np.vstack((coords_world,np.ones((1,coords_count))))
    # shape=(nodes_per_elem,num_elems)
    connectivity = pv_connect
    elem_count = connectivity.shape[1]
    nodes_per_elem = connectivity.shape[0]
    # shape=(num_nodes,num_time_steps)
    field_array = np.array(pv_surf[field_key])
    frame_to_render: int = -1

    rot_axis: str = "x"
    phi_y_degs: float = -45
    theta_x_degs: float = -45
    psi_z_degs: float = 0.0

    phi_y_rads: float = phi_y_degs * np.pi/180.0
    theta_x_rads: float = theta_x_degs * np.pi/180.0

    # Set this to 0.0 to get some of the plate outside the FOV
    roi_pos_world = np.mean(sim_data.coords,axis=0)

    # Number of divisions (subsamples) for each pixel for anti-aliasing
    sub_samp: int = 2

    cam_type = "Test"
    if cam_type == "AV507":
        cam_num_px = np.array([2464,2056],dtype=np.int32)
        pixel_size = np.array([3.45e-3,3.45e-3]) # in millimeters!
        focal_leng: float = 25.0

        imaging_rad: float = 150.0 # Not needed for camera data, just for cam pos below
    else:
        cam_num_px = np.array([510,260],dtype=np.int32)
        pixel_size = np.array([10.0e-3,10.0e-3])
        focal_leng: float = 25.0

        imaging_rad: float = 300.0 # Not needed for camera data, just for cam pos below

    if rot_axis == "y":
        cam_pos_world = np.array([roi_pos_world[xx] + imaging_rad*np.sin(phi_y_rads),
                                  roi_pos_world[yy],
                                  imaging_rad*np.cos(phi_y_rads)])

        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, phi_y_degs, 0], degrees=True)
    elif rot_axis == "x":
        cam_pos_world = np.array([roi_pos_world[xx] ,
                                  roi_pos_world[yy] - imaging_rad*np.sin(theta_x_rads),
                                  imaging_rad*np.cos(theta_x_rads)])

        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, theta_x_degs], degrees=True)

    else:
        cam_pos_world = np.array([roi_pos_world[xx],
                                  roi_pos_world[yy],
                                  imaging_rad])
        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, 0], degrees=True)

    #---------------------------------------------------------------------------
    # RASTERISATION START
    cam_data = CameraRasterData(num_pixels=cam_num_px,
                                pixel_size=pixel_size,
                                pos_world=cam_pos_world,
                                rot_world=cam_rot,
                                roi_center_world=roi_pos_world,
                                focal_length=focal_leng,
                                sub_samp=sub_samp)

    # TODO

if __name__ == "__main__":
    main()
