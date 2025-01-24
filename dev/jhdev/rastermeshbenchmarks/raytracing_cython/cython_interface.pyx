cimport numpy as np
import numpy as np
import time

# import cpp libraries
from libcpp.vector cimport vector


# Declare a C++ function signature using `cdef extern`
cdef extern from "raytracer.hpp":
    void raytrace(
        np.ndarray[np.float64_t, ndim=3] cam_pos_world,  # Shape (3, 3, 549)
        np.ndarray[np.int32_t, ndim=2] cam_rot,  # Shape (4, 549)
        np.ndarray[np.float64_t, ndim=1] coords_data,         # Shape (549,)
        vector[double] &image_buffer_c,
        vector[double] &depth_buffer_c)


# A wrapper function to call the C++ function from Python
def cpp_raytrace(
        np.ndarray[np.float64_t, ndim=1]  cam_pos_world,
        np.ndarray[np.float64_t, ndim=2] cam_rot,
        np.ndarray[np.float64_t, ndim=2] coords_world):


    cdef vector[double] image_buffer_c
    cdef vector[double] depth_buffer_c

    # call main cpp raster func
    cpp_raytrace(cam_pos_world, cam_rot, coords_world, image_buffer_c, depth_buffer_c)

    # std::vector to np.ndarray coercion. See here for more info on syntax: 
    #       https://github.com/cython/cython/issues/4487
    #       https://stackoverflow.com/questions/59666307/convert-c-vector-to-numpy-array-in-cython-without-copying
    cdef double[::1] test1 = <double [:image_buffer_c.size()]>image_buffer_c.data()
    cdef double[::1] test2 = <double [:depth_buffer_c.size()]>depth_buffer_c.data()

    np_image_buffer = np.asarray(test1).copy()
    np_depth_buffer = np.asarray(test2).copy()

    # convert back to a 2d array for easy integration back into python code. suprisingly quick!
    image_buffer_2d = np_image_buffer.reshape(640,640)
    depth_buffer_2d = np_depth_buffer.reshape(640,640)

    return image_buffer_2d, depth_buffer_2d

