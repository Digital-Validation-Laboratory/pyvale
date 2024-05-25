'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import time
import warnings
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy import ndimage

from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradata import CameraData
from pyvale.imagesim.alphashape import alphashape

# Constants
(XI,YI) = (0,1)

def get_pixel_vec_in_m(camera: CameraData) -> tuple[np.ndarray,np.ndarray]:

    mppx = camera.m_per_px
    px_vec_xm = np.arange(mppx/2,camera.fov[XI],mppx)
    px_vec_ym = np.arange(mppx/2,camera.fov[YI],mppx)
    px_vec_ym = px_vec_ym[::-1] # flip
    return (px_vec_xm,px_vec_ym)


def get_pixel_grid_in_m(camera: CameraData) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_xm,px_vec_ym) = get_pixel_vec_in_m(camera)
    (px_grid_xm,px_grid_ym) = np.meshgrid(px_vec_xm,px_vec_ym)
    return (px_grid_xm,px_grid_ym)


def get_pixel_vec_in_px(camera: CameraData) -> tuple[np.ndarray,np.ndarray]:

    px_vec_x = np.arange(0,camera.num_px[XI],1)
    px_vec_y = np.arange(0,camera.num_px[YI],1)
    px_vec_y = px_vec_y[::-1] # flip
    return (px_vec_x,px_vec_y)


def get_pixel_grid_in_px(camera: CameraData) -> tuple[np.ndarray,np.ndarray]:
    (px_vec_x,px_vec_y) = get_pixel_vec_in_px(camera)
    (px_grid_x,px_grid_y) = np.meshgrid(px_vec_x,px_vec_y)
    return (px_grid_x,px_grid_y)


def get_subpixel_vec(camera: CameraData, subsample: int = 3
                     ) -> tuple[np.ndarray,np.ndarray]:

    mppx = camera.m_per_px
    subpx_vec_xm = np.arange(mppx/(2*subsample),camera.fov[XI],mppx/subsample)
    subpx_vec_ym = np.arange(mppx/(2*subsample),camera.fov[YI],mppx/subsample)
    subpx_vec_ym = subpx_vec_ym[::-1] #flip
    return (subpx_vec_xm,subpx_vec_ym)


def get_subpixel_grid(camera: CameraData, subsample: int = 3
                     ) -> tuple[np.ndarray,np.ndarray]:
    (subpx_vec_xm,subpx_vec_ym) = get_subpixel_vec(camera,subsample)
    (subpx_grid_xm,subpx_grid_ym) = np.meshgrid(subpx_vec_xm,subpx_vec_ym)
    return (subpx_grid_xm,subpx_grid_ym)


def get_roi_node_vec(camera: CameraData) -> tuple[np.ndarray,np.ndarray]:

    node_vec_x = np.arange(0+camera.roi_loc[XI],
                           camera.roi_len[XI]+camera.roi_loc[XI]+camera.m_per_px/2,
                           camera.m_per_px)
    node_vec_y = np.arange(0+camera.roi_loc[YI],
                           camera.roi_len[YI]+camera.roi_loc[YI]+camera.m_per_px/2,
                           camera.m_per_px)
    node_vec_y = node_vec_y[::-1] # flipud
    return (node_vec_x,node_vec_y)


def get_roi_node_grid(camera: CameraData) -> tuple[np.ndarray,np.ndarray]:
    (node_vec_x,node_vec_y) = get_roi_node_vec(camera)
    (node_grid_x,node_grid_y) = np.meshgrid(node_vec_x,node_vec_y)
    return (node_grid_x,node_grid_y)


# TODO: nodes need to be changed to SimData
def calc_roi_from_nodes(camera: CameraData, nodes: np.ndarray):

    roi_len_x = np.max(nodes[:,XI]) - np.min(nodes[:,XI])
    roi_len_y = np.max(nodes[:,YI]) - np.min(nodes[:,YI])
    roi_len = np.array([roi_len_x,roi_len_y])
    if roi_len[XI] > camera.fov[XI] or roi_len[YI] > camera.fov[YI]:
        warnings.warn('ROI is larger than the cameras FOV')
    return roi_len


# TODO: nodes need to changed to SimData
def calc_res_from_nodes(camera,nodes,border_px):

    roi_len_x_m = np.max(nodes[:,XI]) - np.min(nodes[:,XI])
    roi_len_y_m = np.max(nodes[:,YI]) - np.min(nodes[:,YI])

    roi_len_x_px = camera.num_px[XI] - 2*border_px
    roi_len_y_px = camera.num_px[YI] - 2*border_px

    if roi_len_x_m > roi_len_y_m:
        m_per_px = roi_len_x_m/roi_len_x_px
    else:
        m_per_px = roi_len_y_m/roi_len_y_px

    return m_per_px


def norm_dynamic_range(in_image: np.ndarray, bits: int) -> np.ndarray:

    if bits > 8 and bits < 16:
        ret_image  = ((2**16)/(2**bits))*in_image
    elif bits < 8:
        raise ValueError('Camera cannot have less than an 8 bit dynamic range\n.')

    return ret_image


def get_image_num_str(im_num,width,cam_num=-1):
    num_str = str(im_num)
    num_str = num_str.zfill(width)

    if cam_num >= 0:
        num_str = num_str+'_'+str(cam_num)

    return num_str


def crop_image(camera: CameraData, image: np.ndarray) -> np.ndarray:

    # If the loaded image is larger than required  then crop based on the camera
    if camera.num_px[XI] > image.shape[1]:
        raise ValueError('Cannot crop image: Number of pixels in camera class is larger than in the loaded image\n.')
    elif camera.num_px[XI] < image.shape[1]:
        image = image[:,:camera.num_px[XI]]

    if  camera.num_px[YI] > image.shape[0]:
        raise ValueError('Cannot crop image: Number of pixels in camera class is larger than in the loaded image\n.')
    elif camera.num_px[YI] < image.shape[0]:
         image = image[:camera.num_px[YI],:]

    return image


# TODO: nodes
def mask_image_with_fe(camera: CameraData, image: np.ndarray, nodes: np.ndarray
                       ) -> tuple[np.ndarray,np.ndarray]:

    # Create a mesh of pixel centroid locations
    (px_x_m,px_y_m) = get_pixel_grid_in_m(camera)

    # If the loaded image is larger than required then crop based on the camera
    image = crop_image(camera,image)

    # Specify the size of the ROI based on the farthest node on each aXIs
    # camera.roi_len = calc_roi_from_nodes(camera, nodes)

    # Convert to np array for compatibility with new alpha shape function
    points = np.array((nodes[:,XI]+camera.roi_loc[XI],
                       nodes[:,YI]+camera.roi_loc[YI])).T

    # Calculate the element edge length to use as the alpha radius
    elem_edge = 2*np.max(np.diff(np.sort(nodes[:,XI])))
    alpha = elem_edge

    # Find the alpha shape based on the list of nodal points
    # Returns a shapely polygon - use 'within' to find points inside
    a_shape = alphashape(points, alpha, only_outer=True)

    # Create an array of nans to fill as the specimen image
    spec_im = np.empty(image.shape)
    spec_im[:] = np.nan
    im_mask = np.zeros([camera.num_px[YI],camera.num_px[XI]])

    # Fill the image based on the pixel being within the polygon (alpha-shape)
    # If pixel is not within the specimen set to background default colour
    for yy in range(spec_im.shape[0]):
        for xx in range(spec_im.shape[1]):
            if a_shape.contains(Point(px_x_m[yy,xx],px_y_m[yy,xx])):
                spec_im[yy,xx] = image[yy,xx]
                im_mask[yy,xx] = 1
            else:
                spec_im[yy,xx] = camera.background
                im_mask[yy,xx] = 0

    # Because arrays fill from top down the loop above effectively flips the
    # image, so need to flip it back
    spec_im = spec_im[::-1,:]
    im_mask = im_mask[::-1,:]

    # Return the image of the specimen
    return (spec_im,im_mask)


def get_image_mask(camera: CameraData,nodes: np.ndarray, subsample: int
                   ) -> np.ndarray:

    # Create a mesh of pixel centroid locations
    if subsample > 1:
        [px_x_m,px_y_m] = get_subpixel_grid(camera,subsample)
    else:
        [px_x_m,px_y_m] = get_pixel_grid_in_m(camera)

    # Specify the size of the ROI based on the farthest node on each aXIs
    camera.roi_len = calc_roi_from_nodes(camera, nodes)

    points = np.array((nodes[:,XI]+camera.roi_loc[XI],
                       nodes[:,YI]+camera.roi_loc[YI]))

    # Calculate the element edge length to use as the alpha radius
    elem_edge = 2*np.max(np.diff(np.sort(nodes[:,XI])))
    alpha = elem_edge

    # Find the alpha shape based on the list of nodal points
    # Returns a shapely polygon - use 'within' to find points inside
    a_shape = alphashape(points, alpha, only_outer=True)

    # Create an array of nans to fill as the specimen image
    im_mask = np.zeros([px_x_m.shape[0],px_x_m.shape[1]])

    # Fill the image based on the pixel being within the polygon (alpha-shape)
    # If pixel is not within the specimen set to background default colour
    for yy in range(im_mask.shape[0]):
        for xx in range(im_mask.shape[1]):
            if a_shape.contains(Point(px_x_m[yy,xx],px_y_m[yy,xx])):
                im_mask[yy,xx] = 1
            else:
                im_mask[yy,xx] = 0

    # Because arrays fill from top down the loop above effectively flips the
    # image, so need to flip it back
    im_mask = im_mask[::-1,:]

    return im_mask


def upsample_image(camera: CameraData,
                   id_opts: ImageDefOpts,
                   input_im: np.ndarray):
    # Get grid of pixel centroid locations
    (px_vec_xm,px_vec_ym) = get_pixel_vec_in_m(camera)

    # Get grid of sub-pixel centroid locations
    [subpx_vec_xm,subpx_vec_ym] = get_subpixel_vec(camera, id_opts.subsample)

    upsampled_image_interp = interp2d(px_vec_xm, px_vec_ym, input_im,
                                      kind=id_opts.image_upsamp_interp)
    # This function will flip the image regardless of the y vector input so flip it
    # back to FE coords
    upsampled_image = upsampled_image_interp(subpx_vec_xm,subpx_vec_ym)
    upsampled_image = upsampled_image[::-1,:]

    return upsampled_image


def average_subpixel_image(subpx_image,subsample):
    conv_mask = np.ones((subsample,subsample))/(subsample**2)
    if subsample > 1:
        subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
        avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                     round(subsample/2)-1::subsample]
    else:
        subpx_image_conv = subpx_image
        avg_image = subpx_image

    return avg_image


def gen_grid_image(camera,px_per_period,contrast_amp,contrast_offset=0.5):
    [px_grid_x,px_grid_y] = get_pixel_grid_in_px(camera)

    grid_image = (2*contrast_amp*camera.dyn_range)/4 \
                    *(1+np.cos(2*np.pi*px_grid_x/px_per_period)) \
                    *(1+np.cos(2*np.pi*px_grid_y/px_per_period)) \
                    +camera.dyn_range*(contrast_offset-contrast_amp)

    return grid_image


def deform_image(upsampled_image,camera,id_opts,coords,disp,
                 image_mask=np.array([]),print_on=True):

    # Check that the image mask matches the camera if not warn the user
    if (image_mask.shape[0] != camera.num_px[YI]) or (image_mask.shape[1] != camera.num_px[XI]):
        if image_mask.size == 0:
            warnings.warn('Image mask not specified, using default mask of ones.')
        else:
            warnings.warn('Image mask size does not match camera, using default mask of ones.')
        image_mask = np.ones([camera.num_px[YI],camera.num_px[XI]])


    # Get grid of pixel centroid locations
    #[px_vec_xm,px_vec_ym] = get_pixel_vec(camera)
    (px_grid_xm,px_grid_ym) = get_pixel_grid_in_m(camera)

    # Get grid of sub-pixel centroid locations
    #[subpx_vec_xm,subpx_vec_ym] = get_subpixel_vec(camera, id_opts.subsample)
    (subpx_grid_xm,subpx_grid_ym) = get_subpixel_grid(camera, id_opts.subsample)

    #--------------------------------------------------------------------------
    # Interpolate FE displacements onto the sub-pixel grid
    if print_on:
        print('Interpolating displacement onto sub-pixel grid.')
        tic = time.perf_counter()

    # Interpolate displacements onto sub-pixel locations - nan extrapolation
    subpx_disp_x = griddata((coords[:,XI] + disp[:,XI] + camera.roi_loc[XI],
                             coords[:,YI] + disp[:,YI] + camera.roi_loc[YI]),
                            disp[:,XI],
                            (subpx_grid_xm,subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    subpx_disp_y = griddata((coords[:,XI] + disp[:,XI] + camera.roi_loc[XI],
                             coords[:,YI] + disp[:,YI] + camera.roi_loc[YI]),
                            disp[:,YI],
                            (subpx_grid_xm,subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    # Ndimage interp can't handle nans so force everything outside the specimen
    # to extrapolate outside the FOV - then use ndimage opts to control
    if id_opts.fe_extrap_outside_fov:
        subpx_disp_ext_vals = 2*camera.fov
    else:
        subpx_disp_ext_vals = (0.0,0.0)

    # Set the nans to the extrapoltion value
    subpx_disp_x[np.isnan(subpx_disp_x)] = subpx_disp_ext_vals[XI]
    subpx_disp_y[np.isnan(subpx_disp_y)] = subpx_disp_ext_vals[YI]

    if print_on:
        toc = time.perf_counter()
        print('Interpolating displacement with NaN extrap took {:.4f} seconds'.format(toc-tic))

    #--------------------------------------------------------------------------
    # Interpolate sub-pixel gray levels with ndimage toolbox
    if print_on:
        print('Deforming sub-pixel image.')
        tic = time.perf_counter()

    # Use the sub-pixel displacements to deform the image
    def_subpx_x = subpx_grid_xm-subpx_disp_x
    def_subpx_y = subpx_grid_ym-subpx_disp_y
    # Flip needed to be consistent with pixel coords of ndimage
    def_subpx_x = def_subpx_x[::-1,:]
    def_subpx_y = def_subpx_y[::-1,:]

    # NDIMAGE: IMAGE DEF
    # NOTE: need to shift to pixel centroid co-ords from nodal so -0.5 makes the
    # top left 0,0 in pixel co-ords
    def_subpx_x_in_px = def_subpx_x*(id_opts.subsample/camera.m_per_px)-0.5
    def_subpx_y_in_px = def_subpx_y*(id_opts.subsample/camera.m_per_px)-0.5
    # NOTE: prefilter needs to be on to match griddata and interp2D!
    # with prefilter on this exactly matches I2D but 10x faster!
    def_image_subpx = ndimage.map_coordinates(upsampled_image,
                                            [[def_subpx_y_in_px],
                                             [def_subpx_x_in_px]],
                                            prefilter=True,
                                            order= id_opts.image_def_order,
                                            mode= id_opts.image_def_extrap,
                                            cval= id_opts.image_def_extval)

    def_image_subpx = def_image_subpx[0,:,:].squeeze()
    if print_on:
        toc = time.perf_counter()
        print('Deforming sub-pixel image with ndimage took {:.4f} seconds'.format(toc-tic))

    if print_on:
        tic = time.perf_counter()

    def_image = average_subpixel_image(def_image_subpx,id_opts.subsample)

    if print_on:
        toc = time.perf_counter()
        print('Averaging sub-pixel imagetook {:.4f} seconds'.format(toc-tic))

    #--------------------------------------------------------------------------
    # DEFORMING IMAGE MASK
    # Only need to do this if there are holes and notches
    if id_opts.complex_geom:
        if print_on:
            print('Deforming image mask.')
            tic = time.perf_counter()

        px_disp_x = subpx_disp_x[round(id_opts.subsample/2)-1::id_opts.subsample,
                                 round(id_opts.subsample/2)-1::id_opts.subsample]
        px_disp_y = subpx_disp_y[round(id_opts.subsample/2)-1::id_opts.subsample,
                                 round(id_opts.subsample/2)-1::id_opts.subsample]
        def_px_x = px_grid_xm-px_disp_x
        def_px_y = px_grid_ym-px_disp_y
        # Flip needed to be consistent with pixel coords of ndimage
        def_px_x = def_px_x[::-1,:]
        def_px_y = def_px_y[::-1,:]

        # NDIMAGE: DEFORM IMAGE MASK
        # NOTE: need to shift to pixel centroid co-ords from nodal so -0.5 makes the
        # top left 0,0 in pixel co-ords
        def_px_x_in_px = def_px_x*(1/camera.m_per_px)-0.5
        def_px_y_in_px = def_px_y*(1/camera.m_per_px)-0.5
        # NOTE: prefilter needs to be on to match griddata and interp2D!
        # with prefilter on this exactly matches I2D but 10x faster!
        def_mask = ndimage.map_coordinates(image_mask,
                                            [[def_px_y_in_px],
                                             [def_px_x_in_px]],
                                            prefilter=True,
                                            order=2,
                                            mode='constant',
                                            cval=0)

        def_mask = def_mask[0,:,:].squeeze()
        # Use the deformed image mask to mask the deformed image
        # Mask is 0-1 with 1 being definitely inside the sample 0 outside
        def_image[def_mask<0.51] = camera.background # type: ignore

        if print_on:
            toc = time.perf_counter()
            print('Deforming image mask with ndimage took {:.4f} seconds'.format(toc-tic))

    else:
        def_mask = None

    return (def_image,def_image_subpx,subpx_disp_x,subpx_disp_y,def_mask)

