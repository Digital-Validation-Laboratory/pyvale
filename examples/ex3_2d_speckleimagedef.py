"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================

IMAGE DEFORMATION

This program takes an input image and deforms it using the displacement field
in the fe_data object. The first block of code loads the input image and the
pickled FE data which will be used to deform the image.

The second block of code uses Speckle Image Tools (sit) to create the image
deformation options class and the camera class which control the behaviour of
the image deformation routine.

The FE displacement data and nodal locations are given in 'm' where as the
image to be deformed exists in pixel coordinates. The conversion between these
is controlled by the parameter in the camera class camera.m_per_px. The user
must also locate the sample within the cameras FOV which is controlled by the
Region Of Interest (roi) parameters in the camera object.
"""

import os
import time
import tkinter
from tkinter import filedialog
import pickle
from pprint import pprint
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplim
from PIL import Image

import pyvale.imagesim.imagedef as sit

def main() -> None:
    # LOAD DATA
    print()
    print('------------------------------------------------------------------')
    print('SYNTHETIC IMAGE DEFORMATION')
    print('------------------------------------------------------------------')
    # Setup up the root window of the GUI and hide it
    root = tkinter.Tk()
    root.wm_attributes('-topmost', 1)
    root.withdraw()

    # Gets the directory of the current script file
    cwd = Path.cwd()
    print("Current working directory:")
    print(cwd)

    #------------------------------------------------------------------------------
    # Get path and file name of the synethetic speckle image
    hardCodePath = True
    if not hardCodePath:
        # Pop a file dialog and get the file name and path
        full_path = filedialog.askopenfilename(parent=root,
                                            initialdir=cwd,
                                            title="Select speckle image file")
        im_path, im_file = os.path.split(full_path)
        im_path = Path(im_path)
    else:
        im_path = Path('data/speckleimages')
        #im_file = 'OptimisedSpeckle_500_500_width3.0_16bit_GBlur1.tiff'
        im_file = 'OptimisedSpeckle_2464_2056_width5.0_8bit_GBlur1.tiff'


    print('\nLoading speckle image from path:')
    print('{}'.format(im_path))

    # Load synthetic speckle image to mask
    input_im = mplim.imread(im_path / im_file)
    input_im = input_im.astype(float)
    # If we have RGB then get rid of it
    if input_im.ndim > 2:
        input_im = input_im[:,:,0]

    #---------------------------------------------------------------------------
    # Load FE data
    hardCodePath = True
    if not hardCodePath:
        # Pop a file dialog and get the file name and path
        full_path = filedialog.askopenfilename(parent=root,
                                            initialdir=cwd,
                                            title="Select FE data pickle")
        fe_path, fe_file = os.path.split(full_path)
        fe_path = Path(fe_path)
    else:
        fe_path = Path('scripts/imdef_cases/imdefcase7_RampRigidBodyMotion_1_0px')
        fe_file = 'fe_data.pkl'

    print('\nLoading pickled FE data from path:')
    print('{}'.format(fe_path))

    tic = time.time()
    with open(fe_path / fe_file,'rb') as fe_load_file:
        fe_data = pickle.load(fe_load_file)
    toc = time.time()

    print('Loading FE data pickle took {:.4f} seconds'.format(toc-tic))

    print('')
    print('FE Data:')
    pprint(vars(fe_data))
    print('')

    #---------------------------------------------------------------------------
    # COSMETIC VARIABLES
    # Flag for plotting diagnostic figures
    plot_diags = True
    # Colour map for images and vector fields
    im_cmap = 'gray'
    v_cmap = 'plasma'

    # INIT IMAGE DEF OPTIONS AND CAMERA
    print('')
    print('------------------------------------------------------------------')
    print('INIT. IMAGE DEF. OPTIONS AND CAMERA')
    print('')

    #------------------------------------------------------------------------------
    # CREATE IMAGE DEF OPTS
    id_opts = sit.ImageDefOpts()

    # If the input image is just a pattern then the image needs to be masked to
    # show just the sample geometry.
    id_opts.mask_input_image = False
    # Set this to True for holes and notches and False for a rectangle
    id_opts.complex_geom = False

    # If the input image is much larger than needed it can also be cropped to
    # increase computational speed.
    id_opts.crop_on = False
    id_opts.crop_px = np.array([400,250])

    # Calculates the m/px value based on fitting the specimen/ROI within the camera
    # FOV and leaving a set number of pixels as a border on the longest edge
    id_opts.calc_res_from_fe = True
    id_opts.calc_res_border_px = 10

    # Set this to true to create an undeformed masked image
    id_opts.add_static_frame = True

    print('Image Def Opts:')
    pprint(vars(id_opts))
    print('')

    #------------------------------------------------------------------------------
    # CREATE CAMERA OBJECT
    # Create a default camera object
    camera = sit.CameraData()
    # Need to set the number of pixels in [X,Y], the bit depth and the m/px

    # Assume the camera has the same number of pixels as the input image unless we
    # are going to crop/mask the input image
    (xi,yi) = (0,1) # Indices to make code more readable
    pixels = np.array([input_im.shape[1],input_im.shape[0]])
    if id_opts.crop_on:
        if id_opts.crop_px[xi] > 0:
            pixels[xi] = id_opts.crop_px[xi]
        if id_opts.crop_px[yi] > 0:
            pixels[yi] = id_opts.crop_px[yi]

    # Based on the max grey level work out what the bit depth of the image is
    bits = 8
    if max(input_im.flatten()) > (2**8):
        bits = 16

    # Assume 1mm/px to start with, can update this to fit FE data within the FOV
    # using the id_opts above. Or set this manually.
    default_res = 1.0e-3 # Overwritten by id_opts.calc_res_from_fe = True

    # Set the core camera parameters based on the specified options
    camera.num_px = pixels
    camera.bits = bits
    camera.m_per_px = default_res

    # Centers the specimen (ROI) in the FOV along the [X,Y] axis, if true the
    # camera.roi_loc parameters is set automatically and cannot be overidden
    camera.roi_cent = [True,True]
    # Can manually set the ROI location by setting the above to false and setting
    # the camera.roi_loc as the distance from the origin to the bottom left
    # corner of the sample [X,Y]: camera.roi_loc = np.array([1e-3,1e-3])

    # Default ROI is the whole FOV but we want to set this to be based on the
    # furthest nodes, this is set in FE units 'meters' and does not change FOV
    camera.roi_len = sit.calc_roi_from_nodes(camera,fe_data.nodes)

    # If we are masking an image we might want to set an optimal resolution based
    # on leaving a specified number of pixels free on each image edge, this will
    # change the FOV in 'meters'
    if id_opts.calc_res_from_fe:
        camera.m_per_px = sit.calc_res_from_nodes(camera,fe_data.nodes,
                                                id_opts.calc_res_border_px)

    # Default ROI is the whole FOV but we want to set this to be based on the
    # furthest nodes, this is set in FE units 'meters' and does not change FOV
    camera.roi_len = sit.calc_roi_from_nodes(camera,fe_data.nodes)

    print('Camera:')
    pprint(vars(camera))
    print('')

    # PRE-PROCESSING
    print('')
    print('------------------------------------------------------------------')
    print('IMAGE AND DATA PRE-PROCESSING')
    print('')

    # Show the image before any pre-processing
    if plot_diags:
        fig, ax = plt.subplots()
        cset = plt.imshow(input_im,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Raw Input Image',fontsize=12)
        cbar = fig.colorbar(cset)

    # 1) ADD ZERO DISP FRAME TO FE DATA
    if id_opts.add_static_frame:
        fe_data.disp.x = np.hstack((np.zeros((fe_data.disp.x.shape[0],1)),
                                    fe_data.disp.x))
        fe_data.disp.y = np.hstack((np.zeros((fe_data.disp.x.shape[0],1)),
                                    fe_data.disp.y))

    # 2) CROP IMAGE
    input_im = sit.crop_image(camera,input_im)

    # 3) MASK IMAGE
    image_mask = None
    if id_opts.mask_input_image:
        print('Image masking turned on, masking image...')
        tic = time.time()
        [input_im,image_mask] = sit.mask_image_with_fe(camera, input_im,
                                                    fe_data.nodes)
        toc = time.time()
        print('Masking image took {:.4f} seconds'.format(toc-tic))

    # Only need to do this if there are holes and notches
    if id_opts.complex_geom and not id_opts.mask_input_image:
        print('Complex geometry is turned on.')
        print('Finding image mask by finding the alpha-shape of the FE nodes.')
        tic = time.time()
        image_mask = sit.get_image_mask(camera, fe_data.nodes, 1)
        toc = time.time()
        print('Calculating image mask took {:.4f} seconds'.format(toc-tic))

    if image_mask is None:
        image_mask = np.ones([camera.num_px[yi],camera.num_px[xi]])

    if plot_diags:
        fig, ax = plt.subplots()
        cset = plt.imshow(input_im,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Pre-processed Image',fontsize=12)
        cbar = fig.colorbar(cset)

    # GENERATE UPSAMPLED IMAGE
    print('')
    print('------------------------------------------------------------------')
    print('GENERATE UPSAMPLED IMAGE')
    print('')

    print('Upsampling input image with a {}x{} subpixel'.format(id_opts.subsample,
                                                                id_opts.subsample))
    tic = time.time()
    upsampled_image = sit.upsample_image(camera,id_opts,input_im)
    toc = time.time()
    print('Upsampling image with I2D took {:.4f} seconds'.format(toc-tic))

    # DEFORM IMAGES
    print('')
    print('------------------------------------------------------------------')
    print('DEFORMING IMAGES')

    # If there is only one frame we can't call shape
    if fe_data.disp.x.ndim == 1:
        num_frames = 1
    else:
        num_frames = fe_data.disp.x.shape[1]

    ticl = time.time()
    for ff in range(num_frames):
        #--------------------------------------------------------------------------
        ticf = time.time()
        print('')
        print('DEFORMING FRAME: {}'.format(ff))

        [def_image,def_image_subpx,subpx_disp_x,subpx_disp_y,def_mask] = sit.deform_image(
            upsampled_image,camera,id_opts,
            np.array([fe_data.nodes.loc_x,fe_data.nodes.loc_y]),
            np.array([fe_data.disp.x[:,ff],fe_data.disp.y[:,ff]]),
            image_mask=image_mask,print_on=True)

        #--------------------------------------------------------------------------
        # SAVE IMAGE
        save_path = fe_path / 'deformed_images'

        # Need to flip image so coords are top left with Y down
        save_image = def_image[::-1,:]

        if not save_path.is_dir():
            save_path.mkdir()

        im_num = sit.get_image_num_str(ff,3)
        save_file = save_path / str('defimage_'+im_num+'.tiff')
        im = Image.fromarray(save_image.astype(np.uint16))
        im.save(save_file)

        tocf = time.time()
        print('DEFORMING FRAME: {} took {:.4f} seconds'.format(ff,tocf-ticf))

    tocl = time.time()
    print('')
    print('==================================================')
    print('Deforming all images took {:.4f}'.format(tocl-ticl))
    print('==================================================')

    # DIAGNOSTIC FIGURES
    print('')
    print('--------------------------------------------------------------------')
    print('PLOTTING DIAGNOSTIC FIGURES')

    if plot_diags:
        # Get grid of pixel centroid locations
        [px_vec_xm,px_vec_ym] = sit.get_pixel_vec(camera)
        [px_grid_xm,px_grid_ym] = sit.get_pixel_grid(camera)

        # Get grid of sub-pixel centroid locations
        [subpx_vec_xm,subpx_vec_ym] = sit.get_subpixel_vec(camera, id_opts.subsample)
        [subpx_grid_xm,subpx_grid_ym] = sit.get_subpixel_grid(camera, id_opts.subsample)

        #--------------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(input_im,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Input Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #--------------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(upsampled_image,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Upsampled Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #--------------------------------------------------------------------------
        if id_opts.complex_geom:
            fig, ax = plt.subplots()
            cset = plt.imshow(image_mask,cmap=plt.get_cmap(im_cmap),origin='lower')
            ax.set_aspect('equal','box')
            ax.set_title('Undef. Image Mask',fontsize=12)
            cbar = fig.colorbar(cset)

            fig, ax = plt.subplots()
            cset = plt.imshow(def_mask,cmap=plt.get_cmap(im_cmap),origin='lower')
            ax.set_aspect('equal','box')
            ax.set_title('Def. Mask',fontsize=12)
            cbar = fig.colorbar(cset)

        #--------------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(def_image_subpx,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Subpx Def. Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #--------------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(def_image,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Def. Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #--------------------------------------------------------------------------
        title_str = 'Sub Pixel Disp X'
        fig, ax = plt.subplots()
        fig.set_dpi(300)
        ext = tuple(np.array([subpx_grid_xm.min(),subpx_grid_xm.max(),
                        subpx_grid_ym.min(),subpx_grid_ym.max()])*10**3)
        cset = plt.imshow(subpx_disp_x,
                        aspect='auto',interpolation='none',
                        origin='lower',cmap=plt.get_cmap(v_cmap),
                        extent=ext)
        ax.set_aspect('equal','box')
        ax.set_title(title_str,fontsize=12)
        ax.set_xlabel('x [mm]',fontsize=12)
        ax.set_ylabel('y [mm]',fontsize=12)
        cbar = fig.colorbar(cset)
        plt.show()

        #--------------------------------------------------------------------------
        title_str = 'Sub Pixel Disp Y'
        fig, ax = plt.subplots()
        fig.set_dpi(300)
        ext = tuple(np.array([subpx_grid_xm.min(),subpx_grid_xm.max(),
                        subpx_grid_ym.min(),subpx_grid_ym.max()])*10**3)
        cset = plt.imshow(subpx_disp_y,
                        aspect='auto',interpolation='none',
                        origin='lower',cmap=plt.get_cmap(v_cmap),
                        extent=ext)
        ax.set_aspect('equal','box')
        ax.set_title(title_str,fontsize=12)
        ax.set_xlabel('x [mm]',fontsize=12)
        ax.set_ylabel('y [mm]',fontsize=12)
        cbar = fig.colorbar(cset)
        plt.show()

    # COMPLETE
    print('')
    print('--------------------------------------------------------------------')
    print('COMPLETE\n')

if __name__ == "__main__":
    main()

