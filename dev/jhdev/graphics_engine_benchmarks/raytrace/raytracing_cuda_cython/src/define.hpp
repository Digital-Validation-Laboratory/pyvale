#define CUDA_CALL(x) do { if((x) != cudaSuccess) { printf("Error at %s:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {printf("Error at %s:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE);}} while(0)
#define INFO_OUT(a,b) std::cout.width(75); std::cout << std::left << a; std::cout << b << std::endl;
#define MEASURE_TIME(taskName, block)                              \
    do {                                                           \
        auto start = std::chrono::high_resolution_clock::now();    \
        { block }                                                  \
        auto end = std::chrono::high_resolution_clock::now();      \
        std::chrono::duration<double> duration = end - start;      \
        INFO_OUT(taskName, std::setprecision(8) << duration.count() << " [s]");         \
    } while (0)
