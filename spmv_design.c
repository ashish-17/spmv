#include "genresult.cuh"
#include <sys/time.h>

extern void sort_matrix(MatrixInfo *mat);

/* Put your own kernel(s) here*/
__global__ void design_kernel(const int nz, const int *rIndex, const int *cIndex, const float *val, const float *vec, float *res) {

	__shared__ float shared_val[1024];

	const int nzPerWarp = 64;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nzPerIter = countWarps * nzPerWarp;
	int iter = nz % nzPerIter ? nz / nzPerIter + 1 : nz / nzPerIter;
	int warpIdx = threadId / 32;
	for (int i = 0; i < iter; ++i) {
		int warpId = warpIdx + i*countWarps;
		int nzStart = warpId*nzPerWarp;
		int nzEnd = nzStart + nzPerWarp - 1;

		for (int j = nzStart + lane; j <= nzEnd && j < nz; j += 32) {
			shared_val[threadIdx.x] = val[j] * vec[cIndex[j]];

			if (lane >= 1 && rIndex[j] == rIndex[j - 1]) {
				shared_val[threadIdx.x] += shared_val[threadIdx.x-1];
			}
			if (lane >= 2 && rIndex[j] == rIndex[j - 2]) {
				shared_val[threadIdx.x] += shared_val[threadIdx.x-2];
			}
			if (lane >= 4 && rIndex[j] == rIndex[j - 4]) {
				shared_val[threadIdx.x] += shared_val[threadIdx.x-4];
			}
			if (lane >= 8 && rIndex[j] == rIndex[j - 8]) {
				shared_val[threadIdx.x] += shared_val[threadIdx.x-8];
			}
			if (lane >= 16 && rIndex[j] == rIndex[j - 16]) {
				shared_val[threadIdx.x] += shared_val[threadIdx.x-16];
			}

			__syncthreads();

			if (((j < nz - 1) && rIndex[j] != rIndex[j+1]) || (j % 32 == 31 || j == nz - 1)) {
				atomicAdd(&res[rIndex[j]],  shared_val[threadIdx.x]);
			}
		}
	}
}


typedef struct matrixDataF {
	int rIndex;
	int cIndex;
	float val;
	int* freq;
} matf_t;


static inline int matfComparator(void* a, void*b) {
	matf_t* a1 = (matf_t*) a;
	matf_t* b1 = (matf_t*) b;
	if (a1->rIndex != b1->rIndex) 
		return (a1->rIndex - b1->rIndex);
	else 
		return (a1->cIndex - b1->cIndex);
}

void reorder_data(MatrixInfo* mat) {
	printf("Reordering the matrix by frequency..\n");
	matf_t *mem = (matf_t*)malloc(sizeof(matf_t)*mat->nz);
	matf_t *mem1 = (matf_t*)malloc(sizeof(matf_t)*mat->nz);
	int* freq = (int*)malloc(sizeof(int) * mat->M);
	memset(freq, 0 ,sizeof(int) * mat->M);
	int* done = (int*)malloc(sizeof(int) * mat->nz);
	memset(done, 0 ,sizeof(int) * mat->nz);
	for (int i = 0; i < mat->nz; ++i) {
		mem[i].rIndex = mat->rIndex[i];
		mem[i].cIndex = mat->cIndex[i];
		mem[i].val = mat->val[i];
		freq[mat->rIndex[i]]++;
		mem[i].freq = freq;
	}

	mergeSortSeq(mem, sizeof(matf_t), mat->nz, matfComparator);

	int count = 0;
	for (int j = 32; j >= 2; j = j >> 1) {
		for (int i = 0; i < mat->nz; ++i) {
			if (done[i] == 0 && freq[mem[i].rIndex] / j >= 1) {
				for (int z = i; z < i + j; ++z) {
					freq[mem[z].rIndex]--;
					done[z] = 1;
					mem1[count].rIndex = mem[z].rIndex;
					mem1[count].cIndex = mem[z].cIndex;
					mem1[count].val = mem[z].val;

					count++;
				}

				i += j -1;
			}
		}
	}

	for (int i = 0; i < mat->nz; ++i) {
		if (done[i] == 0) {
			done[i] = 1;
			mem1[count].rIndex = mem[i].rIndex;
			mem1[count].cIndex = mem[i].cIndex;
			mem1[count].val = mem[i].val;

			count++;
		}
	}

	for (int i = 0; i < mat->nz; ++i) {
		mat->rIndex[i] = mem1[i].rIndex;
		mat->cIndex[i] = mem1[i].cIndex;
		mat->val[i] = mem1[i].val;
	}

	free(mem);
	free(mem1);
	free(freq);
	free(done);
}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
	/*Allocate here...*/

	reorder_data(mat);

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
	design_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_res);
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
