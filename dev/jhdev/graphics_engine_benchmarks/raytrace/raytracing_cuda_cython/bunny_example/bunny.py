"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import time
from pathlib import Path
from multiprocessing.pool import Pool
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import sys
import os

import mooseherder as mh
import pyvale

from rasterizer import CameraRasterData, Rasteriser


sys.path.append("../")
import cython_interface

def main() -> None:

    (xx,yy,zz,ww) = (0,1,2,3)


    # mesh = pv.read("../../../inputs/Stanford_Bunny_sample.stl")

    mesh = pv.Sphere()


    time_start_setup = time.perf_counter()

    pv_surf = mesh.extract_surface()
    faces = np.array(pv_surf.faces)

    ic(mesh)
    ic(pv_surf)
    # pv_surf.plot(show_edges=True, color="lightblue")

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

    #shape=(3,num_coords)
    coords_world = pv_coords
    coords_count = coords_world.shape[1]


    # shape=(4,num_nodes)
    coords_world= np.vstack((coords_world,np.ones((1,coords_count))))

    # shape=(nodes_per_elem,num_elems)
    connectivity = pv_connect
    elem_count = connectivity.shape[1]


    # CAMERA INFO
    cam_num_px = np.array([100,100],dtype=np.int32)
    # pixel_size = np.array([3.45e-3,3.45e-3]) # in millimeters!
    pixel_size = np.array([3.45e-2,3.45e-2]) # in millimeters!
    focal_leng: float = 1.0

    cam_pos_world = np.array([0.0, 0.0, 10.0])
    # cam_rot = np.array([[ 0.9703, 0.0984, -0.2210], [-0.2419, 0.3947, -0.8864], [-0.0000, 0.9135,  0.4067]])
    cam_rot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0 ], [0.0, 0.0, 1.0]])

    ic(cam_pos_world)
    ic(cam_rot)

    # dummy variables that aren't actually get used
    roi_pos_world = np.array([0.0, 0.0, 0.0])
    sub_samp = 1.0
    
        # RASTERISATION START
    cam_data = CameraRasterData(num_pixels=cam_num_px,
                                pixel_size=pixel_size,
                                pos_world=cam_pos_world,
                                rot_world=cam_rot,
                                roi_center_world=roi_pos_world,
                                focal_length=focal_leng,
                                sub_samp=sub_samp)
    

    ic(cam_data)


    # Compute camera focal point and up direction =---------------------------

    # cam_focus = cam_rot[:, 0]  # Camera looks along the 3rd column
    # cam_up = cam_rot[:, 1]  # Camera's up direction is the 2nd column
    # plotter = pv.Plotter()
    # plotter.add_mesh(pv_surf, show_edges=True, color="lightblue")
    # plotter.camera.position = cam_pos_world.tolist()
    # plotter.camera.focal_point = cam_focus.tolist()
    # plotter.camera.up = cam_up.tolist()
    # plotter.show()
    # ------------------------------------------------------------------------


    back_face_mask = Rasteriser.back_face_removal_mask(cam_data,
                                                        coords_world,
                                                        connectivity)

    coords_world = coords_world[:,connectivity]
    coords_world = np.swapaxes(coords_world,0,1)
    coords_world = coords_world[:,:,back_face_mask]
    coords_world = coords_world[:,:-1,:]
    coords_world = coords_world * 11



    ## testing to see if I can add a floor to the image
    # ic(coords_world.shape)
    # floor = np.array([[-50.0, -50.0, 0.0], [50.0, 0.0, 0.0], [0.0, 100.0,  0.0]])
    # expanded_array_3x3 = floor[:, :, np.newaxis]
    # new_array = np.concatenate((coords_world, expanded_array_3x3), axis=2)
    # ic(new_array.shape)

    # ic(floor[1,0])
    # ic(floor[1,1])
    # ic(floor[1,2])

    # checking backface removal --------------------------------------------
    # x = new_array[2, 0, :]
    # y = new_array[2, 1, :]
    # z = new_array[2, 2, :]
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # plt.show()
    # ------------------------------------------------------------------------


    time_start_loop = time.perf_counter()
    depth_buffer = cython_interface.call_gpu_raytrace(
        focal_leng,
        pixel_size[0],
        cam_num_px[0],
        cam_num_px[1],
        cam_pos_world,
        cam_rot,
        coords_world.flatten()) 
    time_end_loop = time.perf_counter()
    time_cpp_loop = time_end_loop - time_start_loop

    
    # plt.figure()
    # plt.pcolor(depth_buffer)
    # plt.show()


    # image_buffer = np.full(cam_data.sub_samp*cam_data.num_pixels,0.0).T

    # ic(depth_buffer.shape)
    # ic(depth_buffer)

    # plt.figure()
    # plt.pcolor(depth_buffer)
    # plt.show()

    # print()
    # print(80*"=")
    # print("RASTER LOOP END")
    # print(80*"=")
    # print()
    # print(80*"=")
    # print("PERFORMANCE TIMERS")
    # print(f"Total elements:    {connectivity.shape[1]}")
    # print()
    # print(f"Setup time = {time_end_setup-time_start_setup} seconds")
    # print(f"C++. Loop time  = {time_cpp_loop} seconds")
    # print(80*"=")

    # #===========================================================================
    # PLOTTING
    plot_on = True
    depth_to_plot = np.copy(depth_buffer)
    depth_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    if plot_on:
        plot_opts = pyvale.PlotOptsGeneral()

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(depth_to_plot,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title("Depth buffer",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        plt.show()

if __name__ == "__main__":
    main()
