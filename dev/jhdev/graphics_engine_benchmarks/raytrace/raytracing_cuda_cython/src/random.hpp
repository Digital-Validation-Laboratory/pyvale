
namespace curandom {

    extern double *pixel_samples;

    void setup_curand_generator();
    void destroy_generator();
    void generate_pixel_samples(int num_samples);
}
