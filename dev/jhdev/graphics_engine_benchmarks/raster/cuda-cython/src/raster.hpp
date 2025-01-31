#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>
#include <numpy/arrayobject.h>


void raster_gpu(int sub_samp,
    PyArrayObject *raster_coords,
    PyArrayObject *bound_box_inds,
    PyArrayObject *areas,
    PyArrayObject *frame_divide_z,
    std::vector<double> &image_buffer,
    std::vector<double> &depth_buffer);