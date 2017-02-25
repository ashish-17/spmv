#include "genresult.cuh"
#include <sys/time.h>

extern void sort_matrix(MatrixInfo *mat);

/* Put your own kernel(s) here*/
__global__ void design_kernel(const int nz, const int *rIndex, const int *cIndex, const float *val, const float *vec, float *res) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int iter = nz % threadCount ? nz / threadCount + 1 : nz / threadCount;

	for (int i = 0; i < iter; ++i) {
		int dataId = threadId + i*threadCount;
		if (dataId < nz) {
			float data = val[dataId];
			int r = rIndex[dataId];
			int c = cIndex[dataId];
			float tmp = data * vec[c];
			atomicAdd(&res[r], tmp);
		}
	}
}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	/*Allocate here...*/
	
	sort_matrix(mat);

	int *d_cIndex, *d_rIndex;
	float *d_val, *d_vec, *d_res;

	memset(res->val, 0, sizeof(float)*res->nz);
	cudaMalloc((void **)&d_cIndex, sizeof(int)*mat->nz);
	cudaMalloc((void **)&d_rIndex, sizeof(int)*mat->nz);
	cudaMalloc((void **)&d_val, sizeof(float)*mat->nz);
	cudaMalloc((void **)&d_vec, sizeof(float)*vec->nz);
	cudaMalloc((void **)&d_res, sizeof(float)*mat->M);

	cudaMemcpy(d_cIndex, mat->cIndex, mat->nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rIndex, mat->rIndex, mat->nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, mat->val, mat->nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, vec->val, vec->nz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_res, res->val, res->nz*sizeof(float), cudaMemcpyHostToDevice);

	/* Sample timing code */
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	/*Invoke kernels...*/
	getMulAtomic_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_res);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	/*Deallocate.*/
	cudaMemcpy(res->val, d_res, res->nz*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_cIndex);
	cudaFree(d_rIndex);
	cudaFree(d_val);
	cudaFree(d_vec);
	cudaFree(d_res);
}
