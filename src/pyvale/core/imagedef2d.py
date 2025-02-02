"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage

from pyvale.core.camerarasternp import edge_function, RasteriserNP
from pyvale.core.cameradata2d import CameraData2D
from pyvale.core.cameratools import CameraTools


@dataclass(slots=True)
class ImageDefOpts:
    save_path: Path | None = None
    save_tag: str = "defimage"

    mask_input_image: bool = True

    crop_on: bool = False
    crop_px: np.ndarray | None = None # only used to crop input image if above is true

    calc_res_from_fe: bool =  False
    calc_res_border_px: int = 5

    add_static_ref: bool = False

    fe_interp: str = "linear"
    fe_rescale: bool = True
    fe_extrap_outside_fov: bool = True # forces displacements outside the
    subsample: int = 2

    image_def_order: int = 3
    image_def_extrap: str = "nearest"
    image_def_extval: float = 0.0 # only used if above is "constant"

    def_complex_geom: bool = True

    def __post_init__(self) -> None:
        if self.save_path is None:
            self.save_path = Path.cwd() / "deformed_images"


class ImageDef2D:

    @staticmethod
    def image_mask_from_sim(cam_data: CameraData2D,
                            image: np.ndarray,
                            coords: np.ndarray,
                            connectivity: np.ndarray
                            ) -> tuple[np.ndarray,np.ndarray]:

        # Here to allow for addition
        #subsample: int = cam_data.subsample
        subsample: int = 1

        coords_raster = coords - cam_data.roi_cent_world
        if coords_raster.shape[1] >= 3:
            coords_raster = coords_raster[:,:-1]

        # Coords NDC: Convert to normalised device coords in the range [-1,1]
        coords_raster[:,0] = 2*coords_raster[:,0] / cam_data.field_of_view[0]
        coords_raster[:,1] = 2*coords_raster[:,1] / cam_data.field_of_view[1]

        # Coords Raster: Covert to pixel (raster) coords
        # Shape = ([X,Y,Z],num_nodes)
        coords_raster[:,0] = (coords_raster[:,0] + 1)/2 * cam_data.pixels_count[0]
        coords_raster[:,1] = (1-coords_raster[:,1])/2 * cam_data.pixels_count[1]

        # shape=(num_elems,node_per_elem,coord[x,y])
        elem_coords = np.ascontiguousarray(coords_raster[connectivity,:])

        #shape=(num_elems,coord[x,y,z])
        elem_coord_min = np.min(elem_coords,axis=1)
        elem_coord_max = np.max(elem_coords,axis=1)

        # Check that min/max nodes are within the 4 edges of the camera image
        #shape=(4_edges_to_check,num_elems)
        crop_mask = np.zeros([elem_coords.shape[0],4],dtype=np.int8)
        crop_mask[elem_coord_min[:,0] <= (cam_data.pixels_count[0]-1), 0] = 1
        crop_mask[elem_coord_min[:,1] <= (cam_data.pixels_count[1]-1), 1] = 1
        crop_mask[elem_coord_max[:,0] >= 0, 2] = 1
        crop_mask[elem_coord_max[:,1] >= 0, 3] = 1
        crop_mask = np.sum(crop_mask,axis=1) == 4

        # Mask the element coords
        elem_coords =  np.ascontiguousarray(elem_coords[crop_mask,:,:])

        # Get only the elements that are within the FOV
        # Mask the elem coords and the max and min elem coords for processing
        elem_coord_min = elem_coord_min[crop_mask,:]
        elem_coord_max = elem_coord_max[crop_mask,:]
        num_elems_in_image = elem_coord_min.shape[0]

        # Find the indices of the bounding box that each element lies within on
        # the image, bounded by the upper and lower edges of the image
        elem_bound_boxes_inds = np.zeros([num_elems_in_image,4],dtype=np.int32)
        elem_bound_boxes_inds[:,0] = RasteriserNP.elem_bound_box_low(
                                            elem_coord_min[:,0])
        elem_bound_boxes_inds[:,1] = RasteriserNP.elem_bound_box_high(
                                            elem_coord_max[:,0],
                                            cam_data.pixels_count[0]-1)
        elem_bound_boxes_inds[:,2] = RasteriserNP.elem_bound_box_low(
                                            elem_coord_min[:,1])
        elem_bound_boxes_inds[:,3] = RasteriserNP.elem_bound_box_high(
                                            elem_coord_max[:,1],
                                            cam_data.pixels_count[1]-1)

        num_edges: int = 3
        if elem_coords.shape[1] > 3:
            num_edges = 4

        mask_buffer =  np.full(subsample*cam_data.pixels_count,0.0).T
        # Raster Loop
        for ee in range(elem_coords.shape[0]):
            # Create the subpixel coords inside the bounding box to test with the
            # edge function. Use the pixel indices of the bounding box.
            bound_subpx_x = np.arange(elem_bound_boxes_inds[ee,0],
                                      elem_bound_boxes_inds[ee,1],
                                      1/subsample) + 1/(2*subsample)
            bound_subpx_y = np.arange(elem_bound_boxes_inds[ee,2],
                                      elem_bound_boxes_inds[ee,3],
                                      1/subsample) + 1/(2*subsample)
            (bound_subpx_grid_x,bound_subpx_grid_y) = np.meshgrid(bound_subpx_x,
                                                                  bound_subpx_y)
            bound_coords_grid_shape = bound_subpx_grid_x.shape
            # shape=(coord[x,y],num_subpx_in_box)
            bound_subpx_coords_flat = np.vstack((bound_subpx_grid_x.flatten(),
                                                 bound_subpx_grid_y.flatten()))

            # Create the subpixel indices for buffer slicing later
            subpx_inds_x = np.arange(subsample*elem_bound_boxes_inds[ee,0],
                                     subsample*elem_bound_boxes_inds[ee,1])
            subpx_inds_y = np.arange(subsample*elem_bound_boxes_inds[ee,2],
                                     subsample*elem_bound_boxes_inds[ee,3])
            (subpx_inds_grid_x,subpx_inds_grid_y) = np.meshgrid(subpx_inds_x,
                                                                subpx_inds_y)

            edge = np.zeros((num_edges,bound_subpx_coords_flat.shape[1]),dtype=np.float64)

            if num_edges == 4:
                edge[0,:] = edge_function(elem_coords[ee,1,:],
                                          elem_coords[ee,2,:],
                                          bound_subpx_coords_flat)
                edge[1,:] = edge_function(elem_coords[ee,2,:],
                                          elem_coords[ee,3,:],
                                          bound_subpx_coords_flat)
                edge[2,:] = edge_function(elem_coords[ee,3,:],
                                          elem_coords[ee,0,:],
                                          bound_subpx_coords_flat)
                edge[3,:] = edge_function(elem_coords[ee,0,:],
                                          elem_coords[ee,1,:],
                                          bound_subpx_coords_flat)
            else:
                edge[0,:] = edge_function(elem_coords[ee,1,:],
                                          elem_coords[ee,2,:],
                                          bound_subpx_coords_flat)
                edge[1,:] = edge_function(elem_coords[ee,2,:],
                                          elem_coords[ee,0,:],
                                          bound_subpx_coords_flat)
                edge[2,:] = edge_function(elem_coords[ee,0,:],
                                          elem_coords[ee,1,:],
                                          bound_subpx_coords_flat)


            # Now we check where the edge function is above zero for all edges
            edge_check = np.zeros_like(edge,dtype=np.int8)
            edge_check[edge >= 0.0] = 1
            edge_check = np.sum(edge_check, axis=0)
            # Create a mask with the check, TODO check the 3 here for non triangles
            edge_mask_flat = edge_check == num_edges
            edge_mask_grid = np.reshape(edge_mask_flat,bound_coords_grid_shape)


            subpx_inds_grid_x = subpx_inds_grid_x[edge_mask_grid]
            subpx_inds_grid_y = subpx_inds_grid_y[edge_mask_grid]
            mask_buffer[subpx_inds_grid_y,subpx_inds_grid_x] += 1.0

        mask_buffer[mask_buffer>1.0] = 1.0
        image[mask_buffer<1.0] = cam_data.background
        return (image,mask_buffer)

    @staticmethod
    def upsample_image(cam_data: CameraData2D,
                       input_im: np.ndarray):
        # Get grid of pixel centroid locations
        (px_vec_xm,px_vec_ym) = CameraTools.pixel_vec_leng(cam_data.field_of_view,
                                                           cam_data.leng_per_px)

        # Get grid of sub-pixel centroid locations
        (subpx_vec_xm,subpx_vec_ym) = CameraTools.subpixel_vec_leng(
                                                        cam_data.field_of_view,
                                                        cam_data.leng_per_px,
                                                        cam_data.subsample)

        # NOTE: See Scipy transition from interp2d docs here:
        # https://scipy.github.io/devdocs/tutorial/interpolate/interp_transition_guide.html
        spline_interp = RectBivariateSpline(px_vec_xm,
                                            px_vec_ym,
                                            input_im.T)
        upsampled_image_interp = lambda x_new, y_new: spline_interp(x_new, y_new).T

        # This function will flip the image regardless of the y vector input so flip it
        # back to FE coords
        upsampled_image =  upsampled_image_interp(subpx_vec_xm,subpx_vec_ym)

        return upsampled_image


    @staticmethod
    def preprocess(cam_data: CameraData2D,
                   image_input: np.ndarray,
                   coords: np.ndarray,
                   connectivity: np.ndarray,
                   disp_x: np.ndarray,
                   disp_y: np.ndarray,
                   id_opts: ImageDefOpts,
                   print_on: bool = False
                   ) -> tuple[np.ndarray | None,
                              np.ndarray | None,
                              np.ndarray | None,
                              np.ndarray | None,
                              np.ndarray | None]:

        if print_on:
            print("\n"+"="*80)
            print("IMAGE DEF PRE-PROCESSING\n")

        if not id_opts.save_path.is_dir():
            id_opts.save_path.mkdir()

        # Make displacements a 2D column vector, allows addition of static frame
        if disp_x.ndim == 1:
            disp_x = np.atleast_2d(disp_x).T
        if disp_y.ndim == 1:
            disp_y = np.atleast_2d(disp_y).T

        if id_opts.add_static_ref:
            num_nodes = coords.shape[0] # type: ignore
            disp_x = np.hstack((np.zeros((num_nodes,1)),disp_x))
            disp_y = np.hstack((np.zeros((num_nodes,1)),disp_y))

        image_input = CameraTools.crop_image_rectangle(image_input,
                                                       cam_data.pixels_count)

        if id_opts.mask_input_image or id_opts.def_complex_geom:
            if print_on:
                print('Image masking or complex geometry on, getting image mask.')
                tic = time.perf_counter()

            (image_input,
             image_mask) = ImageDef2D.image_mask_from_sim(cam_data,
                                                          image_input,
                                                          coords,
                                                          connectivity)


            if print_on:
                toc = time.perf_counter()
                print(f'Calculating image mask took {toc-tic:.4f} seconds')
        else:
            image_mask = None


        # Image upsampling
        if print_on:
            print('\n'+'-'*80)
            print('GENERATE UPSAMPLED IMAGE\n')
            print(f'Upsampling input image with a {id_opts.subsample}x{id_opts.subsample} subpixel')
            tic = time.perf_counter()

        upsampled_image = ImageDef2D.upsample_image(cam_data,image_input)

        if print_on:
            toc = time.perf_counter()
            print(f'Upsampling image withtook {toc-tic:.4f} seconds')

        return (upsampled_image,image_mask,image_input,disp_x,disp_y)

    @staticmethod
    def deform_one_image(upsampled_image: np.ndarray,
                         cam_data: CameraData2D,
                         id_opts: ImageDefOpts,
                         coords: np.ndarray,
                         disp: np.ndarray,
                         image_mask: np.ndarray | None = None,
                         print_on: bool = True
                         ) -> tuple[np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray,
                                    np.ndarray | None]:

        if image_mask is not None:
            if (image_mask.shape[0] != cam_data.num_px[1]) or (image_mask.shape[1] != cam_data.num_px[0]):
                if image_mask.size == 0:
                    warnings.warn('Image mask not specified, using default mask of ones.')
                else:
                    warnings.warn('Image mask size does not match camera, using default mask of ones.')
                image_mask = np.ones([cam_data.num_px[1],cam_data.num_px[0]])

        # Get grid of pixel centroid locations
        (px_grid_xm,
         px_grid_ym) = CameraTools.pixel_grid_leng(cam_data.field_of_view,
                                                    cam_data.leng_per_px)
        # Get grid of sub-pixel centroid locations
        (subpx_grid_xm,
         subpx_grid_ym) = CameraTools.subpixel_grid_leng(cam_data.field_of_view,
                                                        cam_data.leng_per_px,
                                                        cam_data.subsample)

        #--------------------------------------------------------------------------
        # Interpolate FE displacements onto the sub-pixel grid
        if print_on:
            print('Interpolating displacement onto sub-pixel grid.')
            tic = time.perf_counter()

        # Interpolate displacements onto sub-pixel locations - nan extrapolation
        subpx_disp_x = griddata((coords[:,0] + disp[:,0] + cam_data.world_to_cam[0],
                                 coords[:,1] + disp[:,1] + cam_data.world_to_cam[1]),
                                 disp[:,0],
                                 (subpx_grid_xm,subpx_grid_ym),
                                 method=id_opts.fe_interp,
                                 fill_value=np.nan,
                                 rescale=id_opts.fe_rescale)

        subpx_disp_y = griddata((coords[:,0] + disp[:,0] + cam_data.world_to_cam[0],
                                 coords[:,1] + disp[:,1] + cam_data.world_to_cam[1]),
                                 disp[:,1],
                                 (subpx_grid_xm,subpx_grid_ym),
                                 method=id_opts.fe_interp,
                                 fill_value=np.nan,
                                 rescale=id_opts.fe_rescale)


