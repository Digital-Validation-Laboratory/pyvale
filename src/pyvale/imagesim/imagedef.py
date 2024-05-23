'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import time
import warnings
import numpy as np
from shapely.geometry import Point
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy import ndimage

from pyvale.imagesim.alphashape import alphashape

class CameraData:
    def __init__(self,num_px=np.array([1000,1000]),bits=8,m_per_px=1.0e-3):
        # Core params
        self._num_px = num_px
        self._bits = bits
        self._m_per_px = m_per_px

        # Calculated parameters
        self._fov = self._m_per_px*self._num_px
        self._dyn_range = 2**self._bits
        self._background = self._dyn_range/2

        # Region of Interest (ROI)
        self._roi_cent = [True,True]
        self._roi_len = self._fov
        self._roi_loc = np.array([0.0,0.0])

    @property
    def num_px(self):
        return self._num_px

    @property
    def bits(self):
        return self._bits
    @property
    def m_per_px(self):
        return self._m_per_px

    @property
    def fov(self):
        return self._fov

    @property
    def dyn_range(self):
        return self._dyn_range

    @property
    def background(self):
        return self._background

    @property
    def roi_len(self):
        return self._roi_len

    @property
    def roi_loc(self):
        return self._roi_loc

    @property
    def roi_cent(self):
        return self._roi_cent

    @num_px.setter
    def num_px(self,in_px):
        self._num_px = in_px
        self._fov = self._m_per_px*self._num_px

    @bits.setter
    def bits(self,in_bits):
        self._bits = in_bits
        self._dyn_range = 2**self._bits
        self._background = self._dyn_range*0.5

    @background.setter
    def background(self,background):
        self._background = background

    @m_per_px.setter
    def m_per_px(self,in_calib):
        self._m_per_px = in_calib
        self._fov = self._m_per_px*self._num_px

    @roi_len.setter
    def roi_len(self,in_len):
        self._roi_len = in_len
        self._cent_roi()

    @roi_loc.setter
    def roi_loc(self,in_loc):
        if sum(self._roi_cent) > 0:
            warnings.warn('ROI is centered, cannot set ROI location. Update centering flags to adjust.')
        else:
            self._roi_loc = in_loc

    @roi_cent.setter
    def roi_cent(self,in_flags):
        self._roi_cent = in_flags
        self._cent_roi()

    def _cent_roi(self):
        if self._roi_cent[0] == True:
            self._roi_loc[0] = (self._fov[0] - self._roi_len[0])/2
        if self._roi_cent[1] == True:
            self._roi_loc[1] = (self._fov[1] - self._roi_len[1])/2

class ImageDefOpts:
    def __init__(self):
        #----------------------------------------------------------------------
        # USER CONTROLLED OPTIONS
        # Set these to achieve desired behaviour

        # Use this if starting with a full speckle or grid to create an
        # an artificial image with just the specimen geom
        self.mask_input_image = True

        # Use this to crop the image to reduce computational  time, useful if
        # there are large borders around the specimen
        self.crop_on = False
        self.crop_px = np.array([1000,1000]) # only used to crop input image if above is true

        # Used to calculate ideal resolution using FE data and leaving a
        # a specified number of pixels around the border of the image
        self.calc_res_from_fe =  False
        self.calc_res_border_px = 5

        # Options to append the input image to the list or to add a zero disp
        # frame at the start of the displacement data, useful to create static
        # image with masking
        self.add_static_frame = True

        #----------------------------------------------------------------------
        # IMAGE AND DISPLACEMENT INTERPOLATION OPTIONS
        # Leave these as defaults unless advanced options are required. The
        # Following setup achieves a 1/1000th pixel accuracy using a rigid body
        # motion test analysed with the grid method and DIC. Computation time
        # is ~15 seconds per 1Mpx image on an AMD Ryzen 7, 4700U, 8 core CPU.

        # Interpolation type for using griddata to interpolate: scipy-griddata
        self.fe_interp = 'linear'
        self.fe_rescale = True
        self.fe_extrap_outside_fov = True # forces displacements outside the
        # specimen area to be padded with border values

        # Subsampling used to split each pixel in the input image
        self.subsample = 3

        # Interpolation used to upsample the input image: scipy-interp2d
        self.image_upsamp_interp = 'cubic'

        # Order of interpolant used to deform the image: scipy.ndimage.map_coords
        self.image_def_order = 3
        self.image_def_extrap = 'nearest'
        self.image_def_extval = 0.0 # only used if above is 'constant'

        # Used to deal with holes and notches - if the specimen is just a
        # rectangle this can be set to false
        self.complex_geom = True


def get_pixel_vec(camera):
    (xi,yi) = (0,1)
    mppx = camera.m_per_px
    px_vec_xm = np.arange(mppx/2,camera.fov[xi],mppx)
    px_vec_ym = np.arange(mppx/2,camera.fov[yi],mppx)
    px_vec_ym = px_vec_ym[::-1] #flip
    return [px_vec_xm,px_vec_ym]

def get_pixel_grid(camera):
    [px_vec_xm,px_vec_ym] = get_pixel_vec(camera)
    [px_grid_xm,px_grid_ym] = np.meshgrid(px_vec_xm,px_vec_ym)
    return [px_grid_xm,px_grid_ym]

def get_pixel_vec_inpx(camera):
    [xi,yi] = [0,1]
    px_vec_x = np.arange(0,camera.num_px[xi],1)
    px_vec_y = np.arange(0,camera.num_px[yi],1)
    px_vec_y = px_vec_y[::-1] #flip
    return [px_vec_x,px_vec_y]

def get_pixel_grid_inpx(camera):
    [px_vec_x,px_vec_y] = get_pixel_vec_inpx(camera)
    [px_grid_x,px_grid_y] = np.meshgrid(px_vec_x,px_vec_y)
    return [px_grid_x,px_grid_y]

def get_subpixel_vec(camera,subsample):
    [xi,yi] = [0,1]
    mppx = camera.m_per_px
    subpx_vec_xm = np.arange(mppx/(2*subsample),camera.fov[xi],mppx/subsample)
    subpx_vec_ym = np.arange(mppx/(2*subsample),camera.fov[yi],mppx/subsample)
    subpx_vec_ym = subpx_vec_ym[::-1] #flip
    return [subpx_vec_xm,subpx_vec_ym]

def get_subpixel_grid(camera,subsample):
    [subpx_vec_xm,subpx_vec_ym] = get_subpixel_vec(camera,subsample)
    [subpx_grid_xm,subpx_grid_ym] = np.meshgrid(subpx_vec_xm,subpx_vec_ym)
    return [subpx_grid_xm,subpx_grid_ym]

def get_roi_node_vec(camera):
    [xi,yi] = [0,1]
    # Create a mesh of nodal locations along the sample area covered by grid
    node_vec_x = np.arange(0+camera.roi_loc[xi],
                           camera.roi_len[xi]+camera.roi_loc[xi]+camera.m_per_px/2,
                           camera.m_per_px)
    node_vec_y = np.arange(0+camera.roi_loc[yi],
                           camera.roi_len[yi]+camera.roi_loc[yi]+camera.m_per_px/2,
                           camera.m_per_px)
    node_vec_y = node_vec_y[::-1] # flipud
    return [node_vec_x,node_vec_y]

def get_roi_node_grid(camera):
    [node_vec_x,node_vec_y] = get_roi_node_vec(camera)
    [node_grid_x,node_grid_y] = np.meshgrid(node_vec_x,node_vec_y)
    return [node_grid_x,node_grid_y]

def calc_roi_from_nodes(camera,nodes):
    [xi,yi] = [0,1]
    roi_len_x = np.max(nodes.loc_x) - np.min(nodes.loc_x)
    roi_len_y = np.max(nodes.loc_y) - np.min(nodes.loc_y)
    roi_len = np.array([roi_len_x,roi_len_y])
    if roi_len[xi] > camera.fov[xi] or roi_len[yi] > camera.fov[yi]:
        warnings.warn('ROI is larger than the cameras FOV')
    return roi_len

def calc_res_from_nodes(camera,nodes,border_px):
    [xi,yi] = [0,1]
    # Calculate ROI length based on dist between furthest nodes
    roi_len_x_m = np.max(nodes.loc_x) - np.min(nodes.loc_x)
    roi_len_y_m = np.max(nodes.loc_y) - np.min(nodes.loc_y)
    # Calculate ROI in px subtracting number of border pixels on each edge
    roi_len_x_px = camera.num_px[xi] - 2*border_px
    roi_len_y_px = camera.num_px[yi] - 2*border_px
    # Depending on which direction is largest set the resolution to fit
    if roi_len_x_m > roi_len_y_m:
        m_per_px = roi_len_x_m/roi_len_x_px
    else:
        m_per_px = roi_len_y_m/roi_len_y_px

    return m_per_px

def norm_dynamic_range(in_image,bits):
    if bits > 8 and bits < 16:
        ret_image  = ((2**16)/(2**bits))*in_image
    elif bits < 8:
        raise ValueError('Camera cannot have less than an 8 bit dynamic range\n.')

    return ret_image

def get_image_num_str(im_num,width,cam_num=-1):
    num_str = str(im_num)
    if len(num_str) < width:
        num_str = num_str.zfill(width-len(num_str)+1)

    if cam_num >= 0:
        num_str = num_str+'_'+str(cam_num)

    return num_str


def crop_image(camera,image):
    [xi,yi] = [0,1]
    # If the loaded image is larger than required then crop based on the camera
    if camera.num_px[xi] > image.shape[1]:
        raise ValueError('Cannot crop image: Number of pixels in camera class is larger than in the loaded image\n.')
    elif camera.num_px[xi] < image.shape[1]:
        image = image[:,:camera.num_px[xi]]

    if  camera.num_px[yi] > image.shape[0]:
        raise ValueError('Cannot crop image: Number of pixels in camera class is larger than in the loaded image\n.')
    elif camera.num_px[yi] < image.shape[0]:
         image = image[:camera.num_px[yi],:]

    return image


def mask_image_with_fe(camera,image,nodes):
    # Indices to make code more readable
    [xi,yi] = [0,1]

    # Create a mesh of pixel centroid locations
    [px_x_m,px_y_m] = get_pixel_grid(camera)

    # If the loaded image is larger than required then crop based on the camera
    image = crop_image(camera,image)

    # Specify the size of the ROI based on the farthest node on each axis
    #camera.roi_len = calc_roi_from_nodes(camera, nodes)

    # Create a list of nodal points to calculate the alpha shape
    points = list()
    for ii in range(nodes.nums.shape[0]):
        points.append((nodes.loc_x[ii]+camera.roi_loc[xi],
                       nodes.loc_y[ii]+camera.roi_loc[yi]))

    # Convert to np array for compatibility with new alpha shape function
    points = np.array(points)

    # Calculate the element edge length to use as the alpha radius
    elem_edge = 2*np.max(np.diff(np.sort(nodes.loc_x)))
    alpha = elem_edge

    # Find the alpha shape based on the list of nodal points
    # Returns a shapely polygon - use 'within' to find points inside
    a_shape = alphashape(points, alpha, only_outer=True)

    # Create an array of nans to fill as the specimen image
    spec_im = np.empty(image.shape)
    spec_im[:] = np.nan
    im_mask = np.zeros([camera.num_px[yi],camera.num_px[xi]])

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
    return [spec_im,im_mask]


def get_image_mask(camera,nodes,subsample):
    # Indices to make code more readable
    [xi,yi] = [0,1]

    # Create a mesh of pixel centroid locations
    if subsample > 1:
        [px_x_m,px_y_m] = get_subpixel_grid(camera,subsample)
    else:
        [px_x_m,px_y_m] = get_pixel_grid(camera)

    # Specify the size of the ROI based on the farthest node on each axis
    camera.roi_len = calc_roi_from_nodes(camera, nodes)

    # Create a list of nodal points to calculate the alpha shape
    points = list()
    for ii in range(nodes.nums.shape[0]):
        points.append((nodes.loc_x[ii]+camera.roi_loc[xi],
                       nodes.loc_y[ii]+camera.roi_loc[yi]))

    # Convert to np array for compatibility with new alpha shape function
    points = np.array(points)

    # Calculate the element edge length to use as the alpha radius
    elem_edge = 2*np.max(np.diff(np.sort(nodes.loc_x)))
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

    # Return the image of the specimen
    return im_mask


def upsample_image(camera,id_opts,input_im):
    # Get grid of pixel centroid locations
    [px_vec_xm,px_vec_ym] = get_pixel_vec(camera)

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


def deform_image(upsampled_image,camera,id_opts,coords,disp,
                 image_mask=np.array([]),print_on=True):
    # Indices to make code more readable
    (xi,yi) = (0,1)

    # Check that the image mask matches the camera if not warn the user
    if (image_mask.shape[0] != camera.num_px[yi]) or (image_mask.shape[1] != camera.num_px[xi]):
        if image_mask.size == 0:
            warnings.warn('Image mask not specified, using default mask of ones.')
        else:
            warnings.warn('Image mask size does not match camera, using default mask of ones.')
        image_mask = np.ones([camera.num_px[yi],camera.num_px[xi]])


    # Get grid of pixel centroid locations
    #[px_vec_xm,px_vec_ym] = get_pixel_vec(camera)
    (px_grid_xm,px_grid_ym) = get_pixel_grid(camera)

    # Get grid of sub-pixel centroid locations
    #[subpx_vec_xm,subpx_vec_ym] = get_subpixel_vec(camera, id_opts.subsample)
    (subpx_grid_xm,subpx_grid_ym) = get_subpixel_grid(camera, id_opts.subsample)

    #--------------------------------------------------------------------------
    # Interpolate FE displacements onto the sub-pixel grid
    if print_on:
        print('Interpolating displacement onto sub-pixel grid.')
        tic = time.perf_counter()

    # Interpolate displacements onto sub-pixel locations - nan extrapolation
    subpx_disp_x = griddata((coords[xi] + disp[xi] + camera.roi_loc[xi],
                             coords[yi] + disp[yi] + camera.roi_loc[yi]),
                            disp[xi],
                            (subpx_grid_xm,subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    subpx_disp_y = griddata((coords[xi] + disp[xi] + camera.roi_loc[xi],
                             coords[yi] + disp[yi] + camera.roi_loc[yi]),
                            disp[yi],
                            (subpx_grid_xm,subpx_grid_ym),
                            method=id_opts.fe_interp,
                            fill_value=np.nan,
                            rescale=id_opts.fe_rescale)

    # Ndimage interp can't handle nans so force everything outside the specimen
    # to extrapolate outside the FOV - then use ndimage opts to control
    if id_opts.fe_extrap_outside_fov:
        subpx_disp_ext_vals = 2*camera.fov
    else:
        subpx_disp_ext_vals = [0.0,0.0]

    # Set the nans to the extrapoltion value
    subpx_disp_x[np.isnan(subpx_disp_x)] = subpx_disp_ext_vals[xi]
    subpx_disp_y[np.isnan(subpx_disp_y)] = subpx_disp_ext_vals[yi]

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
    def_subpx_x_inpx = def_subpx_x*(id_opts.subsample/camera.m_per_px)-0.5
    def_subpx_y_inpx = def_subpx_y*(id_opts.subsample/camera.m_per_px)-0.5
    # NOTE: prefilter needs to be on to match griddata and interp2D!
    # with prefilter on this exactly matches I2D but 10x faster!
    def_image_subpx = ndimage.map_coordinates(upsampled_image,
                                            [[def_subpx_y_inpx],
                                             [def_subpx_x_inpx]],
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
        def_px_x_inpx = def_px_x*(1/camera.m_per_px)-0.5
        def_px_y_inpx = def_px_y*(1/camera.m_per_px)-0.5
        # NOTE: prefilter needs to be on to match griddata and interp2D!
        # with prefilter on this exactly matches I2D but 10x faster!
        def_mask = ndimage.map_coordinates(image_mask,
                                            [[def_px_y_inpx],
                                             [def_px_x_inpx]],
                                            prefilter=True,
                                            order=2,
                                            mode='constant',
                                            cval=0)

        def_mask = def_mask[0,:,:].squeeze()
        # Use the deformed image mask to mask the deformed image
        # Mask is 0-1 with 1 being definitely inside the sample 0 outside
        def_image[def_mask<0.51] = camera.background

        if print_on:
            toc = time.perf_counter()
            print('Deforming image mask with ndimage took {:.4f} seconds'.format(toc-tic))

    return [def_image,def_image_subpx,subpx_disp_x,subpx_disp_y,def_mask]

def gen_grid_image(camera,px_per_period,contrast_amp,contrast_offset=0.5):
    [px_grid_x,px_grid_y] = get_pixel_grid_inpx(camera)

    grid_image = (2*contrast_amp*camera.dyn_range)/4 \
                    *(1+np.cos(2*np.pi*px_grid_x/px_per_period)) \
                    *(1+np.cos(2*np.pi*px_grid_y/px_per_period)) \
                    +camera.dyn_range*(contrast_offset-contrast_amp)

    return grid_image
