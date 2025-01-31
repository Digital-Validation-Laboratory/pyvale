// system headers
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <curand_kernel.h>

// custom headers
#include "define.hpp"


namespace curandom {

    double *pixel_samples;
    curandGenerator_t rand_generator;
    int seed = 1;

    void destroy_generator(){
        curandDestroyGenerator(rand_generator);
        INFO_OUT("Destroying Cuda random number generator: ", "success");
    }
    
    
    void setup_curand_generator(){
        CURAND_CALL(curandCreateGenerator(&rand_generator, CURAND_RNG_PSEUDO_MTGP32));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rand_generator, seed));
        INFO_OUT("Curand Seed: ",seed);
        atexit(destroy_generator);
    }

    
    void generate_pixel_samples(int num_samples){
        // generate uniformly distributed floats
        CURAND_CALL(curandGenerateUniformDouble(rand_generator, pixel_samples, num_samples));
    }

}
