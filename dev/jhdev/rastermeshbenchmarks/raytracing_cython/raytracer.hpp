#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <utility>
#include <numpy/arrayobject.h>

double sub_samp = 1.0;
double focal_length = 25.0;
double spacing = 1.0/sub_samp;
double start_val = 1.0/(2.0 * sub_samp);

int buffer_width = 2464;
int buffer_height = 2056;
int num_pixels = buffer_width * buffer_height;

        
// double frame_buffer.resize(sub_samp * sub_samp * num_pixels, 0.0);        

std::vector<double> origin(3);
std::vector<double> direction(3);
std::vector<double> elem_x(3), elem_y(3), elem_z(3);
double u;
double v;


std::vector<double> dir(3);
std::vector<double> v0v1(3);
std::vector<double> v0v2(3);
    std::vector<double> pvec(3);
    std::vector<double> tvec(3);
    std::vector<double> qvec(3);
    double tolerance = 1.0e-8;
double det, invdet;
std::vector<double> vec(3);

double cam_to_intersect;
double t, tnear;

bool intersect(double *origin, 
    double *direction, 
    double *elem_x,
    double *elem_y,
    double *elem_z){
    
    //////////////////////////////////
    // muller trumbore algorithm
    //////////////////////////////////
  

    v0v1[0] = elem_y[0] - elem_x[0];
    v0v1[1] = elem_y[1] - elem_x[1];
    v0v1[2] = elem_y[2] - elem_x[2];
    v0v2[0] = elem_z[0] - elem_x[0];
    v0v2[1] = elem_z[1] - elem_x[1];
    v0v2[2] = elem_z[2] - elem_x[2];

    // std::cout << "v0v1: " << v0v1[0] << " " << v0v1[1] << " " << v0v1[2] << std::endl;

    // cross product of direction vector with distance fr
    pvec[0] = direction[1] * v0v2[2] - direction[2] * v0v2[1]; 
    pvec[1] = direction[2] * v0v2[0] - direction[0] * v0v2[2];
    pvec[2] = direction[0] * v0v2[1] - direction[1] * v0v2[0]; 
    // std::cout << "pvec: " << pvec[0] << " " << pvec[1] << " " << pvec[2] << std::endl;


    tvec[0] = origin[0] - elem_x[0];
    tvec[1] = origin[1] - elem_x[1];
    tvec[2] = origin[2] - elem_x[2];
    // std::cout << "tvec: " << tvec[0] << " " << tvec[1] << " " << tvec[2] << std::endl;

    qvec[0] = tvec[1] * v0v1[2] - tvec[2] * v0v1[1]; 
    qvec[1] = tvec[2] * v0v1[0] - tvec[0] * v0v1[2];
    qvec[2] = tvec[0] * v0v1[1] - tvec[1] * v0v1[0]; 

    // std::cout << "qvec: " << qvec[0] << " " << qvec[1] << " " << qvec[2] << std::endl;


    // dot product
    det = v0v1[0] * pvec[0] + v0v1[1] * pvec[1] + v0v1[2] * pvec[2]; 

    // std::cout << "det:  " << det << std::endl;

    // if ray is parallel to the element (within tolerance) then ray and element don't intersect
    if (abs(det < tolerance)) return false;


  //std::cout << __FILE__ << " " << __LINE__ << std::endl;

    invdet = 1.0 / det;
    // std::cout << "invdet:  " << invdet << std::endl;


   


  //std::cout << __FILE__ << " " << __LINE__ << std::endl;


    // barycentric coordinates along the triangle edges
    u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invdet;
    v = (direction[0] * qvec[0] + direction[1] * qvec[1] + direction[2] * qvec[2]) * invdet;

    // std::cout << "dir: " << direction[0] << " " << direction[1] << " " << direction[2] << std::endl;
    // std::cout << "u v :" << u << " " << v << std::endl;

//   //std::cout << __FILE__ << " " << __LINE__ << std::endl;


    // check if the intersection is outside the triange. If it is, return false.
    if (u < 0 || u > 1) return false;
    if (v < 0 || u + v > 1) return false;
    

    cam_to_intersect = invdet * (v0v2[0] * qvec[0] + v0v2[1] * qvec[1] + v0v2[2] * qvec[2]);
    // std::cout << "cam_to_intersect: " << cam_to_intersect << std::endl;

    t = cam_to_intersect;

    return true;
}





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

    int num_elems_in_scene = 13804;

    double scale = 0.4667252302; //tan((50.039 * M_PI / 180.0 / 2.0));
    double imageAspectRatio = buffer_width / buffer_height;

    // Image plane dimensions
    double pixel_size = 3.45e-3;
    double image_plane_width = buffer_width * pixel_size;
    double image_plane_height = buffer_height * pixel_size;

    // loop over all pixels in the image 
    for (int y = 0; y < buffer_height; y++){
        for (int x = 0; x < buffer_width; x++){      

            // vec[0] = (2.0 * (x + 0.5) / buffer_width - 1.0) * imageAspectRatio * scale;
            // vec[1] = (1.0 - 2.0 * (y + 0.5) / buffer_height) * scale;
            // vec[2] = -1.0;


            vec[0] = (x + 0.5) * pixel_size - image_plane_width / 2.0; 
            vec[1] = (image_plane_height / 2.0) - (y + 0.5) * pixel_size;  
            vec[2] = -25.0;


            // Matrix multiplication (3x3 * 3x1) + cam_position
            for (int i = 0; i < 3; ++i) {
                dir[i] = 0.0;
                for (int j = 0; j < 3; ++j) {
                    // Accessing elements of the matrix in row-major order
                    dir[i] += cam_rot_ptr[i * 3 + j] * vec[j];
                }
            }

            tnear = 1.0e6;


            for (int elem = 0; elem < num_elems_in_scene; elem++){
                for (int j = 0; j < 3; j++){  

                    elem_x[j] = coords_world_ptr[(3*num_elems_in_scene*0) + (j * num_elems_in_scene) + elem];
                    elem_y[j] = coords_world_ptr[(3*num_elems_in_scene*1) + (j * num_elems_in_scene) + elem];
                    elem_z[j] = coords_world_ptr[(3*num_elems_in_scene*2) + (j * num_elems_in_scene) + elem];
                    
                }


                // debugging
                // std::cout << "orig: " << cam_pos_world_ptr[0] << " " << cam_pos_world_ptr[1] << " " << cam_pos_world_ptr[2] << std::endl;
                // std::cout << "dir: " << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
                // std::cout << elem << " ";
                // std::cout << "elem_x: " << elem_x[0] << " " << elem_x[1] << " " << elem_x[2] << " ";
                // std::cout << "elem_y: " << elem_y[0] << " " << elem_y[1] << " " << elem_y[2] << " ";
                // std::cout << "elem_z: " << elem_z[0] << " " << elem_z[1] << " " << elem_z[2] << std::endl;
                // exit(0);

                t = 1.0e6;

                if ((intersect(cam_pos_world_ptr, &dir[0], &elem_x[0], &elem_y[0], &elem_z[0])) && (t < tnear)){

                    tnear = t;
                } 
            }
            std::cout << tnear << " " << " ";
            // exit(0);
        }
        std::cout << "\n";
    }
}


