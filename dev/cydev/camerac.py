"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
import cython
from cython.parallel import prange, parallel
from cython.cimports.libc.math import floor, ceil

@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def mesh_grid_2d_para(x: cython.double[:], y: cython.double[:]
                 ) -> tuple[np.ndarray,np.ndarray]:

    x_max: cython.size_t = x.shape[0]
    y_max: cython.size_t = y.shape[0]

    x_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)
    y_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)

    x_grid_view: cython.double[:,:] = x_grid
    y_grid_view: cython.double[:,:] = y_grid

    ii: cython.size_t
    jj: cython.size_t

    with cython.nogil, parallel():
        for ii in prange(y_max):
            for jj in range(x_max):
                x_grid_view[ii,jj] = x[jj]
                y_grid_view[ii,jj] = y[jj]

    return (x_grid,y_grid)


@cython.ccall # python+C or cython.cfunc for C only
@cython.boundscheck(False) # Turn off array bounds checking
@cython.wraparound(False)  # Turn off negative indexing
@cython.cdivision(True)    # Turn off divide by zero check
def mesh_grid_2d(x: cython.double[:], y: cython.double[:]
                 ) -> tuple[np.ndarray,np.ndarray]:

    x_max: cython.size_t = x.shape[0]
    y_max: cython.size_t = y.shape[0]

    x_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)
    y_grid = np.empty(shape=(y_max, x_max), dtype=np.float64)

    x_grid_view: cython.double[:,:] = x_grid
    y_grid_view: cython.double[:,:] = y_grid

    ii: cython.size_t
    jj: cython.size_t

    for ii in range(y_max):
        for jj in range(x_max):
            x_grid_view[ii,jj] = x[jj]
            y_grid_view[ii,jj] = y[jj]

    return (x_grid,y_grid)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def vec_range_double(start: cython.double,
                     stop: cython.double,
                     step: cython.double) -> cython.double[:]:

    num_vals: cython.size_t = int(ceil((stop - start) / step))

    vec_range_np = np.empty((num_vals,),np.float64)
    vec_range: cython.double[:] = vec_range_np

    vec_range[0] = start
    ii: cython.size_t
    for ii in range(1,num_vals):
        vec_range[ii] = vec_range[ii-1] + step

    return vec_range

@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def vec_range_int(start: cython.int,
                  stop: cython.int,
                  step: cython.int = 1) -> cython.long[:]:

    num_vals: cython.size_t = int(ceil((stop - start) / step))

    vec_range_np = np.empty((num_vals,),np.int64)
    vec_range: cython.long[:] = vec_range_np

    vec_range[0] = start
    ii: cython.size_t
    for ii in range(1,num_vals):
        vec_range[ii] = vec_range[ii-1] + step

    return vec_range


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def vec_max_double(vals: cython.double[:]) -> cython.double:

    num_vals: cython.size_t = vals.shape[0]

    ii: cython.size_t = 0
    max_val: cython.double = vals[ii]

    for ii in range(1,num_vals):
        if vals[ii] > max_val:
            max_val = vals[ii]

    return max_val

@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def vec_min_double(vals: cython.double[:]) -> cython.double:

    num_vals: cython.size_t = vals.shape[0]

    ii: cython.size_t = 0
    min_val: cython.double = vals[ii]

    for ii in range(1,num_vals):
        if vals[ii] < min_val:
            min_val = vals[ii]

    return min_val


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def vec_dot_double(vec0: cython.double[:], vec1: cython.double[:]
                   ) -> cython.double:
    vec0_len: cython.size_t = vec0.shape[0]
    vec1_len: cython.size_t = vec1.shape[0]
    if vec0_len != vec1_len:
        return 0.0

    ii: cython.size_t = 0
    dot: cython.double = 0.0
    for ii in range(vec0_len):
        dot += vec0[ii]*vec1[ii]

    return dot



@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def bound_index_min(min_val: cython.double) -> cython.int:
    min_ind: cython.int = int(floor(min_val))
    if min_ind < 0:
        min_ind = 0
    return min_ind


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def bound_index_max(max_val: cython.double,
                    num_pixels: cython.int) -> cython.int:
    max_ind: cython.int = int(ceil(max_val))
    if max_ind > (num_pixels-1):
        max_ind = (num_pixels-1)
    return max_ind


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def mult_mat44_by_vec3(mat44: cython.double[:,:], vec3: cython.double[:]
                       ) -> cython.double[:]:

    coord_view = np.empty((4,),dtype=np.float64)
    coord_slice: cython.double[:] = coord_view

    coord_slice[0] = mat44[0,0]*vec3[0] + mat44[0,1]*vec3[1] + mat44[0,2]*vec3[2] + mat44[0,3]
    coord_slice[1] = mat44[1,0]*vec3[0] + mat44[1,1]*vec3[1] + mat44[1,2]*vec3[2] + mat44[1,3]
    coord_slice[2] = mat44[2,0]*vec3[0] + mat44[2,1]*vec3[1] + mat44[2,2]*vec3[2] + mat44[2,3]
    coord_slice[3] = mat44[3,0]*vec3[0] + mat44[3,1]*vec3[1] + mat44[3,2]*vec3[2] + mat44[3,3]

    return coord_slice


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
def world_to_raster_coords(coords_world: cython.double[:],
                           world_to_cam_mat: cython.double[:,:],
                           image_dist: cython.double,
                           image_dims: cython.double[:],
                           num_pixels: cython.int[:]
                           ) -> cython.double[:]:
    xx: cython.size_t = 0
    yy: cython.size_t = 1
    zz: cython.size_t = 2
    ww: cython.size_t = 3

    coords_raster = mult_mat44_by_vec3(world_to_cam_mat,coords_world)

    coords_raster[xx] = coords_raster[xx] / coords_raster[ww]
    coords_raster[yy] = coords_raster[yy] / coords_raster[ww]
    coords_raster[zz] = coords_raster[zz] / coords_raster[ww]

    coords_raster[xx] = (image_dist * coords_raster[xx]
                        / -coords_raster[zz])
    coords_raster[yy] = (image_dist * coords_raster[yy]
                        / -coords_raster[zz])

    coords_raster[xx] = 2*coords_raster[xx] / image_dims[xx]
    coords_raster[yy] = 2*coords_raster[yy] / image_dims[yy]

    coords_raster[xx] = (coords_raster[xx] + 1)/2 * num_pixels[xx]
    coords_raster[yy] = (1-coords_raster[yy])/2 * num_pixels[yy]
    coords_raster[zz] = -coords_raster[zz]

    return coords_raster


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def edge_function(vert_0: cython.double[:],
                  vert_1: cython.double[:],
                  vert_2: cython.double[:]) -> cython.double:
    edge_fun = ((vert_2[0] - vert_0[0]) * (vert_1[1] - vert_0[1])
              - (vert_2[1] - vert_0[1]) * (vert_1[0] - vert_0[0]))
    return edge_fun


@cython.ccall
#@cython.boundscheck(False)
@cython.wraparound(False)
def raster_loop(field_to_render: cython.double[:,:],
                elem_world_coords: cython.double[:,:,:],
                world_to_cam_mat: cython.double[:,:],
                num_pixels: cython.int[:],
                image_dims: cython.double[:],
                image_dist: cython.double,
                sub_samp: cython.int,
                ) -> tuple[np.ndarray,np.ndarray]:

    xx: cython.size_t = 0
    yy: cython.size_t = 1
    zz: cython.size_t = 2
    nodes_per_elem: cython.size_t = elem_world_coords.shape[0]

    elem_count: cython.size_t = elem_world_coords.shape[2]
    # elem_count: cython.size_t = 1
    elems_in_image: cython.int = 0

    sub_pixels_x: cython.int = num_pixels[0]*sub_samp
    sub_pixels_y: cython.int = num_pixels[1]*sub_samp

    depth_buffer_np = 1.0e6*np.ones((sub_pixels_y,sub_pixels_x))
    image_buffer_np = np.full((sub_pixels_y,sub_pixels_x),0.0)
    depth_buffer: cython.double[:,:] = depth_buffer_np
    image_buffer: cython.double[:,:] = image_buffer_np

    # shape=(nodes_per_elem, coord[X,Y,Z,W])
    nodes_raster_np = np.empty((nodes_per_elem,4),dtype=np.float64)
    nodes_raster: cython.double[:,:] = nodes_raster_np

    px_coord_np = np.zeros((3,),np.float64)
    px_coord: cython.double[:] = px_coord_np

    # NOTE: assumes linear elements with triangles only!
    weights_np = np.zeros((3,),np.float64)
    weights: cython.double[:] = weights_np

    # tolerance for floating point zero dot product
    tol: cython.double = 1e-9

    ee: cython.size_t = 0
    for ee in range(elem_count):
        # shape=(nodes_per_elem,coord[X,Y,Z,W])
        nodes_world: cython.double[:,:] = elem_world_coords[:,:,ee]

        nn: cython.size_t = 0
        for nn in range(nodes_per_elem):
            nodes_raster[nn,:] = world_to_raster_coords(nodes_world[nn,:],
                                                        world_to_cam_mat,
                                                        image_dist,
                                                        image_dims,
                                                        num_pixels)

        x_min: cython.double = vec_min_double(nodes_raster[:,xx])
        x_max: cython.double = vec_max_double(nodes_raster[:,xx])
        y_min: cython.double = vec_min_double(nodes_raster[:,yy])
        y_max: cython.double = vec_max_double(nodes_raster[:,yy])

        # print()
        # print(nodes_raster)
        # print()
        # print(x_min)
        # print(x_max)
        # print(y_max)
        # print(y_min)
        # print()

        if ((x_min > num_pixels[xx]-1) or (x_max < 0)
            or (y_min > num_pixels[yy]-1) or (y_max < 0)):
            #print(f"Cropping element {ee}")
            continue

        elems_in_image += 1

        xi_min: cython.size_t = bound_index_min(x_min)
        xi_max: cython.size_t = bound_index_max(x_max,num_pixels[xx])
        yi_min: cython.size_t = bound_index_min(y_min)
        yi_max: cython.size_t = bound_index_max(y_max,num_pixels[yy])

        # print()
        # print(f"{xi_min=}")
        # print(f"{xi_max=}")
        # print(f"{yi_min=}")
        # print(f"{yi_max=}")

        nn = 0
        for nn in range(nodes_per_elem):
            nodes_raster[nn,zz] = 1/nodes_raster[nn,zz]

        # shape=(nodes_per_elem,)
        nodes_field: cython.double[:] = field_to_render[:,ee]

        elem_area = edge_function(nodes_raster[0,:],
                                  nodes_raster[1,:],
                                  nodes_raster[2,:])

        # print()
        # print(f"{nodes_field.shape=}")
        # print("nodes_field = ")
        # print(np.array(nodes_field))
        # print(f"{elem_area=}")

        bound_coords_x = vec_range_double(float(xi_min),
                                          float(xi_max),
                                          1.0/float(sub_samp))
        bound_coords_y = vec_range_double(float(yi_min),
                                          float(yi_max),
                                          1.0/float(sub_samp))

        nn = 0
        num_bound_x: cython.size_t = bound_coords_x.shape[0]
        for nn in range(num_bound_x):
            bound_coords_x[nn] += 1/(2*sub_samp)

        nn = 0
        num_bound_y: cython.size_t = bound_coords_y.shape[0]
        for nn in range(num_bound_y):
            bound_coords_y[nn] += 1/(2*sub_samp)

        bound_inds_x = vec_range_int(sub_samp*xi_min,sub_samp*xi_max)
        bound_inds_y = vec_range_int(sub_samp*yi_min,sub_samp*yi_max)

        # print()
        # print(f"{bound_coords_x.shape=}")
        # print("bound_coords_x=")
        # print(np.array(bound_coords_x))
        # print()
        # print(f"{bound_inds_x.shape=}")
        # print("bound_inds_x=")
        # print(np.array(bound_inds_x))

        ii: cython.size_t = 0
        jj: cython.size_t = 0
        for jj in range(num_bound_y):
            for ii in range(num_bound_x):
                px_coord[xx] = bound_coords_x[ii]
                px_coord[yy] = bound_coords_y[jj]

                # TODO: fix this for non-triangles
                weights[0] = edge_function(nodes_raster[1,:],
                                           nodes_raster[2,:],
                                           px_coord)
                weights[1] = edge_function(nodes_raster[2,:],
                                           nodes_raster[0,:],
                                           px_coord)
                weights[2] = edge_function(nodes_raster[0,:],
                                           nodes_raster[1,:],
                                           px_coord)

                if ((weights[0] > 0.0) and (weights[1] > 0.0)
                    and (weights[2] > 0.0)):

                    weights[0] = weights[0] / elem_area
                    weights[1] = weights[1] / elem_area
                    weights[2] = weights[2] / elem_area

                    weight_dot_nodes: cython.double = vec_dot_double(weights,
                                                        nodes_raster[:,zz])
                    # Avoid a divide by zero problem here
                    if (weight_dot_nodes > tol) and (weight_dot_nodes < -tol):
                        continue

                    px_coord_z: cython.double = 1/weight_dot_nodes
                    px_field: cython.double = (vec_dot_double(nodes_field,weights)
                                               * px_coord_z)

                    if px_coord_z < depth_buffer[bound_inds_y[jj],bound_inds_x[ii]]:
                        depth_buffer[bound_inds_y[jj],bound_inds_x[ii]] = px_coord_z
                        image_buffer[bound_inds_y[jj],bound_inds_x[ii]] = px_field

    return (image_buffer_np,depth_buffer_np)



