"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================

IMAGE DEFORMATION

This program takes an input image and deforms it using the displacement field
in the sim_data object. The first block of code loads the input image and the
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

import time
import pickle
from pprint import pprint
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplim
from PIL import Image

from mooseherder import ExodusReader

from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradata import CameraData
import pyvale.imagesim.imagedefdiags as idd
import pyvale.imagesim.imagedef as sid


def main() -> None:
    # LOAD DATA
    print()
    print('-'*80)
    print('PYVALE EXAMPLE: IMAGE DEFORMATION 2D')
    print('-'*80)
    # Gets the directory of the current script file
    cwd = Path.cwd()
    print("Current working directory:")
    print(cwd)

    #------------------------------------------------------------------------------
    # Get path and file name of the synethetic speckle image
    im_path = Path('data/speckleimages')
    im_file = 'OptimisedSpeckle_500_500_width3.0_16bit_GBlur1.tiff'
    #im_file = 'OptimisedSpeckle_2464_2056_width5.0_8bit_GBlur1.tiff'

    print('\nLoading speckle image from path:')
    print(im_path)

    # Load synthetic speckle image to mask
    input_im = mplim.imread(im_path / im_file)
    input_im = input_im.astype(float)
    # If we have RGB then get rid of it
    if input_im.ndim > 2:
        input_im = input_im[:,:,0]

    #---------------------------------------------------------------------------
    # Load simulation data - expects a mooseherder.SimData object
    read_exodus = True

    sim_path = Path.cwd()
    if read_exodus:
        case_str = 'case14'
        sim_path = Path(f'simcases/{case_str}')
        #sim_file = 'case02_out.e'
        sim_file = f'{case_str}_out.e'

        print('\nLoading SimData from exodus in path:')
        print(sim_path)

        exodus_reader = ExodusReader(sim_path / sim_file)
        sim_data = exodus_reader.read_all_sim_data()

    else:
        sim_path = Path('scripts/imdef_cases/imdefcase8_RampRigidBodyMotion_5_0px')
        sim_file = 'sim_data.pkl'

        print('\nLoading pickled SimData from path:')
        print(sim_path)

        with open(sim_path / sim_file,'rb') as sim_load_file:
            sim_data = pickle.load(sim_load_file)


    print('')
    print('SimData:')
    pprint(vars(sim_data))
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
    print('-'*80)
    print('INIT. IMAGE DEF. OPTIONS AND CAMERA')
    print('')

    #---------------------------------------------------------------------------
    # CREATE IMAGE DEF OPTS
    id_opts = ImageDefOpts()

    # If the input image is just a pattern then the image needs to be masked to
    # show just the sample geometry.
    id_opts.mask_input_image = True
    # Set this to True for holes and notches and False for a rectangle
    id_opts.complex_geom = True

    # If the input image is much larger than needed it can also be cropped to
    # increase computational speed.
    id_opts.crop_on = False
    id_opts.crop_px = np.array([400,250])

    # Calculates the m/px value based on fitting the specimen/ROI within the camera
    # FOV and leaving a set number of pixels as a border on the longest edge
    id_opts.calc_res_from_fe = True
    id_opts.calc_res_border_px = 10

    # Set this to true to create an undeformed masked image
    if read_exodus:
        id_opts.add_static_frame = False
    else:
        id_opts.add_static_frame = True

    print('Image Def Opts:')
    pprint(vars(id_opts))
    print('')

    #---------------------------------------------------------------------------
    # CREATE CAMERA OBJECT
    # Create a default camera object
    camera = CameraData()
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
    camera.roi_cent = (True,True)
    # Can manually set the ROI location by setting the above to false and setting
    # the camera.roi_loc as the distance from the origin to the bottom left
    # corner of the sample [X,Y]: camera.roi_loc = np.array([1e-3,1e-3])

    # Default ROI is the whole FOV but we want to set this to be based on the
    # furthest nodes, this is set in FE units 'meters' and does not change FOV
    camera.roi_len = sid.calc_roi_from_nodes(camera,sim_data.coords)

    # If we are masking an image we might want to set an optimal resolution based
    # on leaving a specified number of pixels free on each image edge, this will
    # change the FOV in 'meters'
    if id_opts.calc_res_from_fe:
        camera.m_per_px = sid.calc_res_from_nodes(camera,sim_data.coords,
                                                id_opts.calc_res_border_px)

    # Default ROI is the whole FOV but we want to set this to be based on the
    # furthest nodes, this is set in FE units 'meters' and does not change FOV
    camera.roi_len = sid.calc_roi_from_nodes(camera,sim_data.coords)

    print('Camera:')
    pprint(vars(camera))
    print('')

    #---------------------------------------------------------------------------
    # PRE-PROCESSING
    print('')
    print('-'*80)
    print('IMAGE AND DATA PRE-PROCESSING')
    print('')

    if plot_diags:
        idd.plot_diag_image('Raw input image',input_im,im_cmap)

    if id_opts.add_static_frame:
        num_nodes = sim_data.coords.shape[0]
        sim_data.node_vars['disp_x'] = np.hstack((np.zeros((num_nodes,1)),
                                    sim_data.node_vars['disp_x']))
        sim_data.node_vars['disp_y'] = np.hstack((np.zeros((num_nodes,1)),
                                    sim_data.node_vars['disp_y']))

    input_im = sid.crop_image(camera,input_im)

    image_mask = None
    if id_opts.mask_input_image:
        print('Image masking turned on, masking image...')
        tic = time.perf_counter()
        (input_im,image_mask) = sid.mask_image_with_fe(camera, input_im,
                                                       sim_data.coords)
        toc = time.perf_counter()
        print(f'Masking image took {toc-tic:.4f} seconds')

    # Only need to do this if there are holes and notches
    # TODO: use the FE mesh and connectivity to avoid alphashape!
    if id_opts.complex_geom and not id_opts.mask_input_image:
        print('Complex geometry is turned on.')
        print('Finding image mask by finding the alpha-shape of the FE nodes.')
        tic = time.perf_counter()
        image_mask = sid.get_image_mask(camera, sim_data.coords, 1)
        toc = time.perf_counter()
        print(f'Calculating image mask took {toc-tic:.4f} seconds')

    if image_mask is None:
        image_mask = np.ones([camera.num_px[yi],camera.num_px[xi]])

    if plot_diags:
        idd.plot_diag_image('Pre-processed image', input_im, im_cmap)

    #---------------------------------------------------------------------------
    # GENERATE UPSAMPLED IMAGE
    print('')
    print('-'*80)
    print('GENERATE UPSAMPLED IMAGE')
    print('')

    print(f'Upsampling input image with a {id_opts.subsample}x{id_opts.subsample} subpixel')
    tic = time.perf_counter()
    upsampled_image = sid.upsample_image(camera,id_opts,input_im)
    toc = time.perf_counter()
    print(f'Upsampling image with I2D took {toc-tic:.4f} seconds')


    #---------------------------------------------------------------------------
    # DEFORM IMAGES
    print('')
    print('-'*80)
    print('DEFORMING IMAGES')

    # If there is only one frame we can't call shape
    if sim_data.node_vars['disp_x'].ndim == 1:
        num_frames = 1
    else:
        num_frames = sim_data.node_vars['disp_x'].shape[1]

    ticl = time.perf_counter()
    for ff in range(num_frames):
        #-----------------------------------------------------------------------
        ticf = time.perf_counter()
        print('')
        print(f'DEFORMING FRAME: {ff}')

        disp_input = np.array([sim_data.node_vars['disp_x'][:,ff],
                               sim_data.node_vars['disp_y'][:,ff]]).T

        (def_image,def_image_subpx,subpx_disp_x,subpx_disp_y,def_mask) = sid.deform_image(
                upsampled_image,
                camera,
                id_opts,
                sim_data.coords,
                disp_input,
                image_mask=image_mask,
                print_on=True)

        #-----------------------------------------------------------------------
        # SAVE IMAGE
        save_path = sim_path / 'deformed_images'

        # Need to flip image so coords are top left with Y down
        save_image = def_image[::-1,:]

        if not save_path.is_dir():
            save_path.mkdir()

        im_num = sid.get_image_num_str(ff,3,0)
        save_file = save_path / str('defimage_'+im_num+'.tiff')
        im = Image.fromarray(save_image.astype(np.uint16))
        im.save(save_file)

        tocf = time.perf_counter()
        print(f'DEFORMING FRAME: {ff} took {tocf-ticf:.4f} seconds')

    tocl = time.perf_counter()
    print('')
    print('==================================================')
    print(f'Deforming all images took {tocl-ticl:.4f}')
    print('==================================================')

    #---------------------------------------------------------------------------
    # DIAGNOSTIC FIGURES
    if plot_diags:
        print('')
        print('--------------------------------------------------------------------')
        print('PLOTTING DIAGNOSTIC FIGURES')

        # Get grid of pixel centroid locations
        #(px_vec_xm,px_vec_ym) = sid.get_pixel_vec_in_m(camera)
        #(px_grid_xm,px_grid_ym) = sid.get_pixel_grid_in_m(camera)

        # Get grid of sub-pixel centroid locations
        #(subpx_vec_xm,subpx_vec_ym) = sid.get_subpixel_vec(camera, id_opts.subsample)
        (subpx_grid_xm,subpx_grid_ym) = sid.get_subpixel_grid(camera, id_opts.subsample)

        #-----------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(input_im,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Input Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #-----------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(upsampled_image,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Upsampled Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #-----------------------------------------------------------------------
        if id_opts.complex_geom:
            fig, ax = plt.subplots()
            cset = plt.imshow(image_mask,cmap=plt.get_cmap(im_cmap),origin='lower')
            ax.set_aspect('equal','box')
            ax.set_title('Undef. Image Mask',fontsize=12)
            cbar = fig.colorbar(cset)

            if def_mask is not None:
                fig, ax = plt.subplots()
                cset = plt.imshow(def_mask,cmap=plt.get_cmap(im_cmap),origin='lower')
                ax.set_aspect('equal','box')
                ax.set_title('Def. Mask',fontsize=12)
                cbar = fig.colorbar(cset)

        #-----------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(def_image_subpx,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Subpx Def. Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #-----------------------------------------------------------------------
        fig, ax = plt.subplots()
        cset = plt.imshow(def_image,cmap=plt.get_cmap(im_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Def. Image',fontsize=12)
        cbar = fig.colorbar(cset)

        #-----------------------------------------------------------------------
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

        #-----------------------------------------------------------------------
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

        #-----------------------------------------------------------------------
        plt.show()

    # COMPLETE
    print('')
    print('-'*80)
    print('COMPLETE\n')

if __name__ == "__main__":
    main()

