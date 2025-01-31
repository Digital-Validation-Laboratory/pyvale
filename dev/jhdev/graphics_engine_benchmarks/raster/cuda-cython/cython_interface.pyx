cimport numpy as np
import numpy as np
import time

# import cpp libraries
from libcpp.vector cimport vector


# Declare a C++ function signature using `cdef extern`
cdef extern from "src/raster.hpp":
    void raster_gpu(int sub_samp,
        np.ndarray[np.float64_t, ndim=3] elem_raster_coords,  # Shape (3, 3, 549)
        np.ndarray[np.int32_t, ndim=2] elem_bound_box_inds,  # Shape (4, 549)
        np.ndarray[np.float64_t, ndim=1] elem_areas,         # Shape (549,)
        np.ndarray[np.float64_t, ndim=2] field_frame_divide_z,  # Shape (3, 549)
        vector[double] &image_buffer_c,
        vector[double] &depth_buffer_c)

    # vector[double] return_image_buffer()
    # vector[double] return_depth_buffer()



# A wrapper function to call the C++ function from Python
def call_raster_gpu(int sub_samp,
        np.ndarray[np.float64_t, ndim=3] elem_raster_coords, 
        np.ndarray[np.int32_t, ndim=2] elem_bound_box_inds, 
        np.ndarray[np.float64_t, ndim=1] elem_areas, 
        np.ndarray[np.float64_t, ndim=2] field_frame_divide_z):


    cdef vector[double] image_buffer_c
    cdef vector[double] depth_buffer_c

    # call main cpp raster func
    raster_gpu(sub_samp, elem_raster_coords, elem_bound_box_inds, elem_areas, field_frame_divide_z, image_buffer_c, depth_buffer_c)


    # starting timer
    time_start_loop = time.perf_counter()


    # std::vector to np.ndarray coercion. See here for more info on syntax: 
    #       https://github.com/cython/cython/issues/4487
    #       https://stackoverflow.com/questions/59666307/convert-c-vector-to-numpy-array-in-cython-without-copying

    cdef double[::1] test1 = <double [:image_buffer_c.size()]>image_buffer_c.data()
    cdef double[::1] test2 = <double [:depth_buffer_c.size()]>depth_buffer_c.data()

    np_image_buffer = np.asarray(test1).copy()
    np_depth_buffer = np.asarray(test2).copy()

    # convert back to a 2d array for easy integration back into python code. suprisingly quick!
    image_buffer_2d = np_image_buffer.reshape(2056*sub_samp,2464*sub_samp)
    depth_buffer_2d = np_depth_buffer.reshape(2056*sub_samp,2464*sub_samp)

    #ending timer
    time_end_loop = time.perf_counter()
    time_cpp_loop = time_end_loop - time_start_loop
    print(f"{'Cython coercion of vector to np.array time':75}" + f"{time_cpp_loop:.8f}" + " [s]")

    
    return image_buffer_2d, depth_buffer_2d

