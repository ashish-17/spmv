#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(/*Arguments*/){
    /* This is just an example empty kernel, you don't have to use the same kernel
 * name*/
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/

		/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
    /*Deallocate.*/
}
