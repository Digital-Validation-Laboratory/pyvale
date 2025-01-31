// include headers
#include <cmath>
#include <vector>
#include <chrono>
#include <utility>
#include <iomanip>
#include <iostream>
#include <cstdlib>
// #include <numpy/arrayobject.h>

// custom headers
#include "define.hpp"
// #include "raytrace.hpp"


// variables that can go in constant memory
__constant__ double focal_length_gpu;
__constant__ double pixel_size_gpu;
__constant__ int buffer_height_gpu;
__constant__ int buffer_width_gpu;
__constant__ int num_pixels_gpu;


//////////////////////////////////////////////
// Does primary array intersect with object
//////////////////////////////////////////////    

__device__ bool intersect(
    double *origin, 
    double *direction, 
    double *elem_x,
    double *elem_y,
    double *elem_z,
    double &cam_to_intersect){
    
    //////////////////////////////////
    // muller trumbore algorithm
    //////////////////////////////////

    double v0v1[3];
    v0v1[0] = elem_y[0] - elem_x[0];
    v0v1[1] = elem_y[1] - elem_x[1];
    v0v1[2] = elem_y[2] - elem_x[2];
    // printf("v0v1: %e %e %e\n", v0v1[0] ,v0v1[1] ,v0v1[2]);


    double v0v2[3];
    v0v2[0] = elem_z[0] - elem_x[0];
    v0v2[1] = elem_z[1] - elem_x[1];
    v0v2[2] = elem_z[2] - elem_x[2];
    // printf("v0v2: %e %e %e\n", v0v2[0] ,v0v2[1] ,v0v2[2]);

    double pvec[3];
    pvec[0] = direction[1] * v0v2[2] - direction[2] * v0v2[1]; 
    pvec[1] = direction[2] * v0v2[0] - direction[0] * v0v2[2];
    pvec[2] = direction[0] * v0v2[1] - direction[1] * v0v2[0]; 
    // printf("pvec: %e %e %e\n", pvec[0] ,pvec[1] ,pvec[2]);

    double tvec[3];
    tvec[0] = origin[0] - elem_x[0];
    tvec[1] = origin[1] - elem_x[1];
    tvec[2] = origin[2] - elem_x[2];
    // printf("tvec: %e %e %e\n", tvec[0] ,tvec[1] ,tvec[2]);

    double qvec[3];
    qvec[0] = tvec[1] * v0v1[2] - tvec[2] * v0v1[1]; 
    qvec[1] = tvec[2] * v0v1[0] - tvec[0] * v0v1[2];
    qvec[2] = tvec[0] * v0v1[1] - tvec[1] * v0v1[0]; 
    // printf("qvec: %e %e %e\n", qvec[0] ,qvec[1] ,qvec[2]);

    double det = v0v1[0] * pvec[0] + v0v1[1] * pvec[1] + v0v1[2] * pvec[2]; 
    // printf("det: %e\n", det);

    if (abs(det < 1.0e-8)) return false;

    double invdet = 1.0 / det;
    // printf("invdet: %e\n", invdet);


    // barycentric coordinates along the triangle edges
    double u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invdet;
    double v = (direction[0] * qvec[0] + direction[1] * qvec[1] + direction[2] * qvec[2]) * invdet;
    // printf("uv: %e %e\n", u,v);


    // check if the intersection is outside the triange. If it is, return false.
    if (u < 0 || u > 1) return false;
    if (v < 0 || u + v > 1) return false;
    
    cam_to_intersect = invdet * (v0v2[0] * qvec[0] + v0v2[1] * qvec[1] + v0v2[2] * qvec[2]);
    // printf("cam_to_intersect: %e\n", cam_to_intersect);

    return true;
}



//////////////////////////////////////////////
// raytracer GPU kernel.
//////////////////////////////////////////////    



__global__ void raytrace_gpu(
            int num_pixels,
            int num_elems_in_scene,
            double *cam_pos_world,
            double *cam_rot,
            double *coords_world,
            double *depth_buffer){
                
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pixels){     

        // get centre of each pixel in world coordinates
        int y = idx / buffer_width_gpu; // Row index
        int x = idx % buffer_width_gpu; // Column index

         // get centre of each pixel in world coordinates
        double vec[3];
        vec[0] =  (x + 0.5) * pixel_size_gpu - (buffer_width_gpu  * pixel_size_gpu) / 2.0; 
        vec[1] = -(y + 0.5) * pixel_size_gpu + (buffer_height_gpu * pixel_size_gpu) / 2.0;  
        vec[2] = -focal_length_gpu;

        // get the direction of the primary ray by multiplying with camera rotation matrix
        double direction[3];

        for (int i = 0; i < 3; ++i) {
            direction[i] = 0.0;
            for (int j = 0; j < 3; ++j) {
                direction[i] += cam_rot[i * 3 + j] * vec[j];
            }
        }

        // default the nearest object as a background value very far away.
        double nearest_intersect = 1.0e6;
        double cam_to_intersect = 1.0e6;
        double elem_x[3], elem_y[3], elem_z[3];

        // loop over all the elements to check intersection of primary array with element
        for (int elem = 0; elem < num_elems_in_scene; elem++){

            // get the coordinates of the element vertices from the master array
            for (int j = 0; j < 3; j++){  
                // pull invidielement vertices from the 'master' coordinates list
                elem_x[j] = coords_world[(3*num_elems_in_scene*0) + (j * num_elems_in_scene) + elem];
                elem_y[j] = coords_world[(3*num_elems_in_scene*1) + (j * num_elems_in_scene) + elem];
                elem_z[j] = coords_world[(3*num_elems_in_scene*2) + (j * num_elems_in_scene) + elem];
            }
            

            // debugging
            // printf("dir: %e %e %e\n", cam_pos_world[0] ,cam_pos_world[1] ,cam_pos_world[2]);
            // printf("dir: %e %e %e\n", direction[0] ,direction[1] ,direction[2]);
            // printf("elem_x: %e %e %e\n", elem_x[0] ,elem_x[1] ,elem_x[2]);
            // printf("elem_y: %e %e %e\n", elem_y[0] ,elem_y[1] ,elem_y[2]);
            // printf("elem_z: %e %e %e\n", elem_z[0] ,elem_z[1] ,elem_z[2]);
            
            
            // check intersection
            cam_to_intersect = 1.0e6;
            if ((intersect(cam_pos_world, direction, elem_x, elem_y, elem_z, cam_to_intersect)) && (cam_to_intersect < nearest_intersect)){
                nearest_intersect = cam_to_intersect;
            } 
        }

        depth_buffer[idx] = nearest_intersect;


        // debugging
        // printf("%d %d %e %e\n", y, x,nearest_intersect);
    }
}




//////////////////////////////////////////////
// Setup raytracer. Init and copy vars to GPU
//////////////////////////////////////////////    


void raytrace_gpu_setup(
    double focal_length,
    double pixel_size,
    int buffer_width,
    int buffer_height,
    std::vector<double> cam_pos_world,
    std::vector<double> cam_rot,
    std::vector<double> coords_world,
    std::vector<double> &depth_buffer,
    std::vector<double> &image_buffer){

    // total number of pixels in image
    int num_pixels = buffer_width * buffer_height;
    std::cout << "num pixels: " << num_pixels << std::endl;

    // Access the data pointer for each array
    // double *cam_pos_world_ptr = (double *)PyArray_DATA(cam_pos_world);
    // double *cam_rot_ptr = (double *)PyArray_DATA(cam_rot);
    // double *coords_world_ptr = (double *)PyArray_DATA(coords_world);

    // // get the correction dimensions of the resizable c++ arrays for memory allocation on GPU
    // npy_intp *cam_pos_world_dims = PyArray_DIMS(cam_pos_world);
    // npy_intp *cam_rot_dims = PyArray_DIMS(cam_rot);
    // npy_intp *coords_world_dims = PyArray_DIMS(coords_world);

    // std::cout << "dims" << std::endl;
    // std::cout << cam_pos_world_dims[0] << std::endl;
    // std::cout << cam_rot_dims[0] << " " << cam_rot_dims[1] << std::endl;
    // std::cout << coords_world_dims[0] << std::endl;

    // get the number of elements and the corresponding array size.
    int num_elems_in_scene = coords_world.size() / 3 / 3;
    // std::cout << "coords_world_dims[0]: " << coords_world_dims[0] << std::endl;

    // pointers for GPU memory allocation
    double *depth_buffer_gpu;
    double *image_buffer_gpu;
    double *cam_pos_world_gpu;
    double *cam_rot_gpu;
    double *coords_world_gpu;

    // create depth_buffer
    MEASURE_TIME("C++ Buffer resize:", {
        depth_buffer.resize(num_pixels, 0.0);
        image_buffer.resize(num_pixels, 0.0);
    });

    MEASURE_TIME("Cuda Memory Allcation:", {
        CUDA_CALL(cudaMalloc((void**)&cam_pos_world_gpu,    sizeof(double) * 3));
        CUDA_CALL(cudaMalloc((void**)&cam_rot_gpu,          sizeof(double) * 3 * 3));
        CUDA_CALL(cudaMalloc((void**)&coords_world_gpu,     sizeof(double) * coords_world.size()));
        CUDA_CALL(cudaMalloc((void**)&depth_buffer_gpu,     sizeof(double) * depth_buffer.size()));
        CUDA_CALL(cudaMalloc((void**)&image_buffer_gpu,     sizeof(double) * image_buffer.size()));
    });

    MEASURE_TIME("Cuda Memory copy from CPU to GPU:", {
        CUDA_CALL(cudaMemcpy(cam_pos_world_gpu,     &cam_pos_world[0],  sizeof(double) * 3,                           cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(cam_rot_gpu,           &cam_rot[0],        sizeof(double) * 3 * 3,                       cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(coords_world_gpu,      &coords_world[0],   sizeof(double) * coords_world.size(),        cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(depth_buffer_gpu,      &depth_buffer[0],   sizeof(double) * depth_buffer.size(),         cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(image_buffer_gpu,      &image_buffer[0],   sizeof(double) * image_buffer.size(),         cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(*(&focal_length_gpu),      &focal_length,   sizeof(double)));
        CUDA_CALL(cudaMemcpyToSymbol(*(&pixel_size_gpu),        &pixel_size,     sizeof(double)));
        CUDA_CALL(cudaMemcpyToSymbol(*(&buffer_height_gpu),     &buffer_height,  sizeof(int)));
        CUDA_CALL(cudaMemcpyToSymbol(*(&buffer_width_gpu),      &buffer_width,   sizeof(int)));
        CUDA_CALL(cudaMemcpyToSymbol(*(&num_pixels_gpu),        &num_pixels,     sizeof(int)));
    });
    
    // GPU config
    int threads_per_block = 1024;
    int n_blocks = (num_pixels + threads_per_block - 1) / threads_per_block;

    MEASURE_TIME("Cuda kernel run time:", ({ // added an extra bracket because of the comma in the kernel call messing with the macro.
        raytrace_gpu<<<n_blocks,threads_per_block>>>(num_pixels, num_elems_in_scene, cam_pos_world_gpu, cam_rot_gpu, coords_world_gpu, depth_buffer_gpu);
        CUDA_CALL(cudaDeviceSynchronize()); 

    }););

    MEASURE_TIME("Cuda Memcpy back to CPU:", {
        CUDA_CALL(cudaMemcpy(&depth_buffer[0], depth_buffer_gpu,  sizeof(double) * depth_buffer.size(), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(&image_buffer[0], image_buffer_gpu,  sizeof(double) * image_buffer.size(), cudaMemcpyDeviceToHost));
    });
}

int main(){

    double focal_length = 1.5;
    double pixel_size = 3.45e-3;
    int buffer_height = 640;
    int buffer_width = 640;
    std::vector<double> cam_pos_world = {0.0, -97.66, 93.2};
    std::vector<double> cam_rot = {0.9703, 0.0984, -0.2210, -0.2419, 0.3947, -0.8864, -0.0000, 0.9135,  0.4067};
    std::vector<double> coords_world(3*3*10000);

    std::vector<double> depth_buffer, image_buffer;

    for (int i = 0; i <3*3*10000; i++){
        coords_world[i] =  rand() % 101 / 100.0;
    }
    
    std::cout << "finished init" << std::endl;


    raytrace_gpu_setup(focal_length, pixel_size, buffer_height, buffer_width, cam_pos_world, cam_rot, coords_world, depth_buffer, image_buffer);


    return 0;
}
