#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <utility>
#include <numpy/arrayobject.h>

double sub_samp = 1.0;
double focal_length = 25.0;
double spacing = 1.0/sub_samp;
double start_val = 1.0/(2.0 * sub_samp);

int buffer_width = 640;
int buffer_height = 640;
int num_pixels = buffer_width * buffer_height;

        
// double frame_buffer.resize(sub_samp * sub_samp * num_pixels, 0.0);        

std::vector<double> origin;
std::vector<double> direction;
std::vector<double> elem;
double u;
double v;


// need world coordinates of vertices


void raytrace(PyArrayObject *cam_pos_world,
            PyArrayObject *cam_rot,
            PyArrayObject *coords_world,
            std::vector<double> &image_buffer,
            std::vector<double> &depth_buffer){


  depth_buffer.resize(sub_samp * sub_samp * num_pixels, 1e6);
  image_buffer.resize(sub_samp * sub_samp * num_pixels, 0.0);
              
  // timer
  auto start = std::chrono::high_resolution_clock::now();

  // Access the data pointer for each array
  double *cam_pos_world_ptr = (double *)PyArray_DATA(cam_pos_world);
  double *cam_rot_ptr = (double *)PyArray_DATA(cam_rot);
  double *coords_world_ptr = (double *)PyArray_DATA(coords_world);

  int num_elems_in_scene = 13846;

  // std::vector<double> cam_pos_world = {7.97931958e-03, 1.18577702e+02, 1.06066017e+0};
  // std::vector<double> rot_world = {1, 0, 0, 0, 0.70710678,  0.7071067, 0, -0.70710678,  0.7071067};


  // std::cout << cam_rot_ptr[0] << " " << cam_rot_ptr[1] << " " << cam_rot_ptr[2] << std::endl;



  float scale = tan((50.039 * M_PI / 180.0 / 2.0));
  float imageAspectRatio = buffer_width / buffer_height;

  // loop over all pixels in the image 
  for (int y = 0; y < buffer_height; y++){
    for (int x = 0; x < buffer_width; x++){      

      std::vector<double> vec(3);
      vec[0] = (2 * (x + 0.5) / buffer_width - 1) * imageAspectRatio * scale;
      vec[1] = (1 - 2 * (y + 0.5) / buffer_height) * scale;
      vec[2] = -1.0;

      // std::cout << scale << " " << imageAspectRatio << " " << vec[0] << " " << vec[1] << " " << vec[2] << std::endl;

      std::vector<double> dir(3);

      // Matrix multiplication (3x3 * 3x1) + cam_position
      for (int i = 0; i < 3; ++i) {
        dir[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            // Accessing elements of the matrix in row-major order
            dir[i] += cam_rot_ptr[i * 3 + j] * vec[j];
        }
      }

      bool inter;

      
      // loop over number of elements
      for (int elem = 0; elem < num_elems_in_scene; elem++){
        inter = intersect(&cam_pos_world[0], &dir[0], elem_x[elem], elem_y[elem], elem_z[elem])
      }


    }
  }
}


        // assign direction of primary ray and normalise
        // z = -1;
        // direction[0] = x / sqrt(x*x + y*y + z*z);
        // direction[1] = y / sqrt(x*x + y*y + z*z);
        // direction[2] = z / sqrt(x*x + y*y + z*z);





bool intersect(double *origin, 
    double *direction, 
    double *elem_x,
    double *elem_y,
    double *elem_z,
    double t
    ){

    // distance from camera to intersection with object
    double cam_to_intersect;

    // barycentric coords 
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates.html 
    double u, v;


    //////////////////////////////////
    // muller trumbore algorithm
    //////////////////////////////////
  

    // first need to check whether ray is parallel to triangle element
    std::vector<double> v0v1(3);
    std::vector<double> v0v2(3);

    v0v1[0] = elem_y[0] - elem_x[0];
    v0v1[1] = elem_y[1] - elem_x[1];
    v0v1[2] = elem_y[2] - elem_x[2];
    v0v2[0] = elem_z[0] - elem_x[0];
    v0v2[1] = elem_z[1] - elem_x[1];
    v0v2[2] = elem_z[2] - elem_x[2];

    std::vector<double> det;

    // cross product of direction vector with distance fr
    std::vector<double> pvec;
    pvec[0] = direction[1] * v0v2[2] - direction[2] * v0v2[1]; 
    pvec[1] = direction[2] * v0v2[0] - direction[0] * v0v2[2];
    pvec[2] = direction[0] * v0v2[1] - direction[1] * v0v2[0]; 

    std::vector<double> qvec;
    qvec[0] = direction[1] * v0v1[2] - direction[2] * v0v1[1]; 
    qvec[1] = direction[2] * v0v1[0] - direction[0] * v0v1[2];
    qvec[2] = direction[0] * v0v1[1] - direction[1] * v0v1[0]; 

    // dot product
    det[0] = v0v1[0] * pvec[0] + v0v1[1] * pvec[1] + v0v1[2] * pvec[2]; 

    // if ray is parallel to the element (within tolerance) then ray and element don't intersect
    tolerance = 1.0e-8;
    if (abs(det < tolerance) return false;


    double invdet = 1.0 / det;
    
    std::vector<double> tvec;
    tvec[0] = origin[0] - elem_x[0];
    tvec[1] = origin[1] - elem_x[1];
    tvec[2] = origin[2] - elem_x[2];

    std::vector<double> qvec;
    qvec[0] = direction[1] * v0v1[2] - direction[2] * v0v1[1]; 
    qvec[1] = direction[2] * v0v1[0] - direction[0] * v0v1[2];
    qvec[2] = direction[0] * v0v1[1] - direction[1] * v0v1[0]; 

    // barycentric coordinates along the triangle edges
    double u = tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2];
    double v = direction[0] * qvec[0] + direction[1] * qvec[1] + direction[2] * qvec[2];

    // check if the intersection is outside the triange. If it is, return false.
    if (u < 0 || u > 1) return false;
    if (v < 0 || u + v > 1) return false;
    

    cam_to_intersect = invdet * (v0v2[0] * qvec[0] + v0v2[0] * qvec[0] + v0v2[0] * qvec[0]);

    return true;
  }
}


