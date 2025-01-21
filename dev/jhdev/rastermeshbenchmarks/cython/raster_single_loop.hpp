#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <utility>
#include <numpy/arrayobject.h>



// global stuff
double focal_length, spacing, start_val;

int buffer_width, buffer_height, num_pixels;

std::vector<double> elem_raster_coords_x(3);
std::vector<double> elem_raster_coords_y(3);
std::vector<double> elem_raster_coords_z(3);
std::vector<int> elem_bound_box_inds(4);
std::vector<double> field_frame_divide_z(3);
float elem_areas;






//////////////////////////////////
// raster_one_element
//////////////////////////////////    

void raster_one_element(int sub_samp, std::vector<double> &image_buffer, std::vector<double> &depth_buffer){

    const double x_st = elem_bound_box_inds[0] + start_val;
    const double x_en = elem_bound_box_inds[1] + start_val;
    const double y_st = elem_bound_box_inds[2] + start_val;
    const double y_en = elem_bound_box_inds[3] + start_val;
    const int num_subpx_x = (x_en - x_st) / spacing;
    const int num_subpx_y = (y_en - y_st) / spacing;


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
        double subpx_x = elem_bound_box_inds[0] + start_val;

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

                int index = subpx_y_ind * (buffer_height * sub_samp) + subpx_x_ind;

                if (px_coord_z < depth_buffer[index]) {

                    // Update the depth buffer
                    depth_buffer[index] = px_coord_z;
                    image_buffer[index] = field_interp;
                }
            }

            subpx_x+=spacing;
            subpx_x_ind++;
        }   

        subpx_y+=spacing;
        subpx_y_ind++;
    }
}



////////////////////////////////////
/// raster image
////////////////////////////////////
void raster(int sub_samp,
            PyArrayObject *raster_coords,
            PyArrayObject *bound_box_inds,
            PyArrayObject *areas,
            PyArrayObject *frame_divide_z,
            std::vector<double> &image_buffer,
            std::vector<double> &depth_buffer){


    focal_length = 50.0;
    spacing = 1.0/sub_samp;
    start_val = 1.0/(2.0 * sub_samp);

    buffer_width = 2056;
    buffer_height = 2464;
    num_pixels = buffer_width * buffer_height;

    depth_buffer.resize(sub_samp * sub_samp * num_pixels, 1e6);
    image_buffer.resize(sub_samp * sub_samp * num_pixels, 0.0);        


    // timer
    auto start = std::chrono::high_resolution_clock::now();

    // Access the data pointer for each array
    double *elem_raster_coords_ptr = (double *)PyArray_DATA(raster_coords);
    int *elem_bound_box_inds_ptr = (int *)PyArray_DATA(bound_box_inds);
    double *elem_areas_ptr = (double *)PyArray_DATA(areas);
    double *field_frame_divide_z_ptr = (double *)PyArray_DATA(frame_divide_z);

    // Get the shape of each array
    npy_intp *ar_shape = PyArray_DIMS(areas);
    int num_elems_in_scene = ar_shape[0];


    // loop over the number of elements
    for (int i = 0; i < num_elems_in_scene; i++){
        for (int j = 0; j < 3; j++){
            elem_raster_coords_x[j] = elem_raster_coords_ptr[(i * 3 * 3) + (0 * 3) + j];
            elem_raster_coords_y[j] = elem_raster_coords_ptr[(i * 3 * 3) + (1 * 3) + j];
            elem_raster_coords_z[j] = elem_raster_coords_ptr[(i * 3 * 3) + (2 * 3) + j];
            elem_bound_box_inds[j] = elem_bound_box_inds_ptr[j * num_elems_in_scene + i];
            field_frame_divide_z[j] = field_frame_divide_z_ptr[j * num_elems_in_scene + i];

        }
        
        elem_bound_box_inds[3] = elem_bound_box_inds_ptr[3 * num_elems_in_scene + i];
        elem_areas = elem_areas_ptr[i];


        raster_one_element(sub_samp, image_buffer, depth_buffer);


    }


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "C++ raster Time: " << duration.count() / 1.0e6 << " [s]" << std::endl;
}
