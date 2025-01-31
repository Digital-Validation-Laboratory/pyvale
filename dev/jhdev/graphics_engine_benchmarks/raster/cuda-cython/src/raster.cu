#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <utility>
#include "define.hpp"
#include <cuda.h>
#include "raster.hpp"
#include <iomanip>
#include <cuda_runtime.h>
#include <numpy/arrayobject.h>




// constant memory variables.
__constant__ double focal_length_gpu;
__constant__ double spacing_gpu;
__constant__ double start_val_gpu;

__constant__ int buffer_width_gpu;
__constant__ int buffer_height_gpu;
__constant__ int num_pixels_gpu;



//////////////////////////////////
// raster_one_element
//////////////////////////////////    

__device__ void raster_one_element_gpu(double *elem_raster_coords_x, 
                                   double *elem_raster_coords_y, 
                                   double *elem_raster_coords_z, 
                                   double *field_frame_divide_z, 
                                   int    *elem_bound_box_inds, 
                                   double elem_areas, 
                                   double sub_samp, 
                                   double *image_buffer, 
                                   double *depth_buffer){



    const double x_st = elem_bound_box_inds[0] + start_val_gpu;
    const double x_en = elem_bound_box_inds[1] + start_val_gpu;
    const double y_st = elem_bound_box_inds[2] + start_val_gpu;
    const double y_en = elem_bound_box_inds[3] + start_val_gpu;
    const int num_subpx_x = (x_en - x_st) / spacing_gpu;
    const int num_subpx_y = (y_en - y_st) / spacing_gpu;

    const int subpx_x_st_ind = elem_bound_box_inds[0] * sub_samp;
    const int subpx_y_st_ind = elem_bound_box_inds[2] * sub_samp;

    const double dzy0 = elem_raster_coords_z[0] - elem_raster_coords_y[0];
    const double dxz0 = elem_raster_coords_x[0] - elem_raster_coords_z[0];
    const double dyx0 = elem_raster_coords_y[0] - elem_raster_coords_x[0];
    const double dzy1 = elem_raster_coords_z[1] - elem_raster_coords_y[1];
    const double dxz1 = elem_raster_coords_x[1] - elem_raster_coords_z[1];
    const double dyx1 = elem_raster_coords_y[1] - elem_raster_coords_x[1];

    const double inv_elem_areas = 1.0 / elem_areas;

    int subpx_y_ind = subpx_y_st_ind;
    double subpx_y = y_st;

    // loop over all the subpixels
    for (int y = 0; y < num_subpx_y; y++){

        int subpx_x_ind = subpx_x_st_ind;
        double subpx_x = elem_bound_box_inds[0] + start_val_gpu;

        for (double x = 0; x < num_subpx_x; x++){

            double edge_x = (subpx_x - elem_raster_coords_y[0]) * dzy1 - (subpx_y - elem_raster_coords_y[1]) * dzy0;
            double edge_y = (subpx_x - elem_raster_coords_z[0]) * dxz1 - (subpx_y - elem_raster_coords_z[1]) * dxz0;
            double edge_z = (subpx_x - elem_raster_coords_x[0]) * dyx1 - (subpx_y - elem_raster_coords_x[1]) * dyx0;


            int edge_check = (edge_x >= 0.0) + (edge_y >= 0.0) + (edge_z >= 0.0);
            if (edge_check == 3) {

                double interp_weights_x = edge_x * inv_elem_areas;
                double interp_weights_y = edge_y * inv_elem_areas;
                double interp_weights_z = edge_z * inv_elem_areas;

                double px_coord_z = 1.0 / (
                    elem_raster_coords_x[2] * interp_weights_x +
                    elem_raster_coords_y[2] * interp_weights_y +
                    elem_raster_coords_z[2] * interp_weights_z
                );

                double field_interp = (
                    (field_frame_divide_z[0] * interp_weights_x +
                    field_frame_divide_z[1] * interp_weights_y +
                    field_frame_divide_z[2] * interp_weights_z) * px_coord_z
                );

                int index = subpx_y_ind * (buffer_height_gpu * sub_samp) + subpx_x_ind;

                if (px_coord_z < depth_buffer[index]) {

                    // Update the depth buffer
                    depth_buffer[index] = px_coord_z;
                    image_buffer[index] = field_interp;
                }
            }

            subpx_x+=spacing_gpu;
            subpx_x_ind++;
        }   

        subpx_y+=spacing_gpu;
        subpx_y_ind++;
    }

}


////////////////////////////////////
/// kernel
////////////////////////////////////


__global__ void raster_gpu_kernel(
    int num_elems_in_scene,
    double sub_samp,
    double *elem_raster_coords_ptr,
    double *field_frame_divide_z_ptr,
    double *elem_areas_ptr,
    int *elem_bound_box_inds_ptr,
    double *image_buffer,
    double *depth_buffer){

    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elems_in_scene){

        double elem_raster_coords_x[3];
        double elem_raster_coords_y[3];
        double elem_raster_coords_z[3];
        double field_frame_divide_z[3];
        int elem_bound_box_inds[4];
        const double elem_areas = elem_areas_ptr[i];


        for (int j = 0; j < 3; j++){
            elem_raster_coords_x[j] = elem_raster_coords_ptr[(i * 3 * 3) + (0 * 3) + j];
            elem_raster_coords_y[j] = elem_raster_coords_ptr[(i * 3 * 3) + (1 * 3) + j];
            elem_raster_coords_z[j] = elem_raster_coords_ptr[(i * 3 * 3) + (2 * 3) + j];
            elem_bound_box_inds[j] = elem_bound_box_inds_ptr[j * num_elems_in_scene + i];
            field_frame_divide_z[j] = field_frame_divide_z_ptr[j * num_elems_in_scene + i];

        }
        
        elem_bound_box_inds[3] = elem_bound_box_inds_ptr[3 * num_elems_in_scene + i];

        raster_one_element_gpu(elem_raster_coords_x, elem_raster_coords_y, elem_raster_coords_z, field_frame_divide_z, elem_bound_box_inds, elem_areas, sub_samp, image_buffer, depth_buffer);
        
    }

}



void raster_gpu(int sub_samp,
            PyArrayObject *raster_coords,
            PyArrayObject *bound_box_inds,
            PyArrayObject *areas,
            PyArrayObject *frame_divide_z,
            std::vector<double> &image_buffer,
            std::vector<double> &depth_buffer){   


    double focal_length = 25.0;
    double spacing = 1.0/sub_samp;
    double start_val = 1.0/(2.0 * sub_samp);

    int buffer_width = 2056;
    int buffer_height = 2464;
    int num_pixels = buffer_width * buffer_height;


    // Get pointer for each python numpy array
    double *elem_raster_coords_ptr = (double *)PyArray_DATA(raster_coords);
    int *elem_bound_box_inds_ptr = (int *)PyArray_DATA(bound_box_inds);
    double *elem_areas_ptr = (double *)PyArray_DATA(areas);
    double *field_frame_divide_z_ptr = (double *)PyArray_DATA(frame_divide_z);
    npy_intp *ar_shape = PyArray_DIMS(areas);
    int num_elems_in_scene = ar_shape[0];

    int threads_per_block = 256;
    int n_blocks = (num_elems_in_scene + threads_per_block - 1) / threads_per_block;

    auto start_resize = std::chrono::high_resolution_clock::now();
    depth_buffer.resize(sub_samp * sub_samp * num_pixels, 1e6);
    image_buffer.resize(sub_samp * sub_samp * num_pixels, 0.0);    
    auto end_resize = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_resize = end_resize - start_resize;
    INFO_OUT("C++ image/depth buffer resize time: ", std::fixed << std::setprecision(8) << duration_resize.count() << " [s]");


    npy_intp *elem_raster_coords_dims = PyArray_DIMS(raster_coords);
    npy_intp *elem_bound_box_inds_dims = PyArray_DIMS(bound_box_inds);
    npy_intp *elem_areas_dims = PyArray_DIMS(areas);
    npy_intp *field_frame_divide_z_dims = PyArray_DIMS(frame_divide_z);
    int elem_raster_coords_size = elem_raster_coords_dims[0] * elem_raster_coords_dims[1] * elem_raster_coords_dims[2];
    int elem_bound_box_inds_size = elem_bound_box_inds_dims[0] * elem_bound_box_inds_dims[1];
    int elem_areas_size = elem_areas_dims[0];
    int field_frame_divide_z_size = field_frame_divide_z_dims[0] * field_frame_divide_z_dims[1];

    double *depth_buffer_gpu;
    double *image_buffer_gpu;
    double *elem_raster_coords_gpu;
    int *elem_bound_box_inds_gpu;
    double *elem_areas_gpu;  
    double *field_frame_divide_z_gpu;

    auto start_malloc = std::chrono::high_resolution_clock::now();
    CUDA_CALL(cudaMalloc((void**)&depth_buffer_gpu, sizeof(double) * depth_buffer.size()));
    CUDA_CALL(cudaMalloc((void**)&image_buffer_gpu, sizeof(double) * image_buffer.size()));
    CUDA_CALL(cudaMalloc((void**)&elem_raster_coords_gpu,   sizeof(double)* elem_raster_coords_size));
    CUDA_CALL(cudaMalloc((void**)&elem_bound_box_inds_gpu,  sizeof(int)* elem_bound_box_inds_size));
    CUDA_CALL(cudaMalloc((void**)&elem_areas_gpu,           sizeof(double)* elem_areas_size));
    CUDA_CALL(cudaMalloc((void**)&field_frame_divide_z_gpu,  sizeof(double)* field_frame_divide_z_size));
    auto end_malloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_malloc = end_malloc - start_malloc;
    INFO_OUT("Cuda memory allocation time: ", std::fixed << std::setprecision(8) << duration_malloc.count() << " [s]");


    auto start_memcpy = std::chrono::high_resolution_clock::now();
    CUDA_CALL(cudaMemcpy(elem_raster_coords_gpu,    elem_raster_coords_ptr,   sizeof(double) * elem_raster_coords_size,      cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(elem_bound_box_inds_gpu,   elem_bound_box_inds_ptr,  sizeof(int)    * elem_bound_box_inds_size,     cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(elem_areas_gpu,            elem_areas_ptr,           sizeof(double) * elem_areas_size,              cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(field_frame_divide_z_gpu,  field_frame_divide_z_ptr, sizeof(double) * field_frame_divide_z_size,    cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(depth_buffer_gpu,   &depth_buffer[0],  sizeof(double) * depth_buffer.size(),      cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(image_buffer_gpu,   &image_buffer[0],  sizeof(double) * image_buffer.size(),     cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpyToSymbol(*(&focal_length_gpu),  &focal_length,  sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(*(&spacing_gpu),       &spacing,       sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(*(&start_val_gpu),     &start_val,     sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(*(&buffer_width_gpu),      &buffer_width,  sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(*(&buffer_height_gpu),      &buffer_height, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(*(&num_pixels_gpu),    &num_pixels,    sizeof(int)));
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_memcpy = end_memcpy - start_memcpy;
    INFO_OUT("Cuda memory copy time: ", duration_memcpy.count() << " [s]");

    auto start_kernel = std::chrono::high_resolution_clock::now();
    raster_gpu_kernel<<<n_blocks,threads_per_block>>>(num_elems_in_scene, sub_samp, elem_raster_coords_gpu, field_frame_divide_z_gpu, elem_areas_gpu, elem_bound_box_inds_gpu, image_buffer_gpu, depth_buffer_gpu);
    CUDA_CALL(cudaDeviceSynchronize());  // Wait for the kernel to finish
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duratio_kernel = end_kernel - start_kernel;
    INFO_OUT("Cuda kernel run time: ", std::fixed << std::setprecision(8) << duratio_kernel.count() << " [s]")


    auto start_backtocpu = std::chrono::high_resolution_clock::now();
    std::cout << "depth_buffer size: " << depth_buffer.size() << std::endl;
    CUDA_CALL(cudaMemcpy(&image_buffer[0], image_buffer_gpu,  sizeof(double)* depth_buffer.size(), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&depth_buffer[0], depth_buffer_gpu,  sizeof(double)* image_buffer.size(), cudaMemcpyDeviceToHost));
    auto end_backtocpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_backtocpu = end_backtocpu - start_backtocpu;
    INFO_OUT("Cuda copying image/depth buffers back to cpu time: ", std::fixed << std::setprecision(8) << duration_backtocpu.count() << " [s]")

}





