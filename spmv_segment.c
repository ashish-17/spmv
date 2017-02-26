#include "genresult.cuh"
#include <sys/time.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static void merge(void* data, int item_size, int l, int m, int r, int (*comparator)(void*, void*), void* aux_memory);
static void mergeSortHelper(void* data, int item_size, int l, int r, int (*comparator)(void*, void*), void* aux_memory);

void mergeSortSeq(void* data, int item_size, int n, int (*comparator)(void*, void*)) {
	void* aux_memory = malloc(item_size*n);
	mergeSortHelper(data, item_size, 0, n-1, comparator, aux_memory);
	free(aux_memory);
	aux_memory = NULL;
}

static void mergeSortHelper(void* data, int item_size, int l, int r, int (*comparator)(void*, void*), void* aux_memory) {
	if (l < r) {
		int m = l + (r-l) / 2;
		mergeSortHelper(data, item_size, l, m, comparator, aux_memory);
		mergeSortHelper(data, item_size, m+1, r, comparator, aux_memory);

		merge(data, item_size, l, m, r, comparator, aux_memory);
	}
}

static void merge(void* data, int item_size, int l, int m, int r, int (*comparator)(void*, void*), void* aux_memory) {
	int idxLeftArray = 0, idxRightArray = 0, idxMainArray = l;
	int nLeftArray = (m -l + 1), nRightArray = (r - m);

	memcpy((char*)aux_memory + l*item_size, (char*)data + l*item_size, item_size*nLeftArray);
	memcpy((char*)aux_memory + (m+1)*item_size, (char*)data + (m+1)*item_size, item_size*nRightArray);

	char* left = (char*)aux_memory + l*item_size;
	char* right = (char*)aux_memory + (m+1)*item_size;
	while (idxLeftArray < nLeftArray && idxRightArray < nRightArray) {
		if (comparator((void*)(left + idxLeftArray*item_size), ((void*)(right + idxRightArray*item_size)) ) < 0) {
			memcpy(((void*)((char*)data + idxMainArray*item_size)), ((void*)(left + idxLeftArray*item_size)), item_size);
			idxLeftArray++;
		} else {
			memcpy(((void*)((char*)data + idxMainArray*item_size)), ((void*)(right + idxRightArray*item_size)), item_size);
			idxRightArray++;
		}
		idxMainArray++;
	}

	while (idxLeftArray < nLeftArray) {
		memcpy(((void*)((char*)data + idxMainArray*item_size)), ((void*)(left + idxLeftArray*item_size)), item_size);
		idxLeftArray++;
		idxMainArray++;
	}

	while (idxRightArray < nRightArray) {
		memcpy(((void*)((char*)data + idxMainArray*item_size)), ((void*)(right + idxRightArray*item_size)), item_size);
		idxRightArray++;
		idxMainArray++;
	}
}

__global__ void putProduct_kernel(const int nz, const int *rIndex, const int *cIndex, const float *val, const float *vec, float *res) {

	__shared__ float shared_val[1024];

	const int nzPerWarp = 64;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nzPerIter = countWarps * nzPerWarp;
	int iter = nz % nzPerIter ? nz / nzPerIter + 1 : nz / nzPerIter;

	for (int i = 0; i < iter; ++i) {
		int warpId = threadId / 32 + i*countWarps;
		int nzStart = warpId*nzPerWarp;
		int nzEnd = nzStart + nzPerWarp - 1;
		int inc = blockDim.x < 32 ? blockDim.x : 32;

		shared_val[threadIdx.x] = 0;
		for (int j = nzStart + lane; j <= nzEnd && j < nz; j += inc) {
			shared_val[threadIdx.x] = val[j] * vec[cIndex[j]];
			__syncthreads();

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

typedef struct matrixData {
	int rIndex;
	int cIndex;
	float val;
} mat_t;

static inline int matComparator(void* a, void*b) {
	return (((mat_t*)a)->rIndex - ((mat_t*)b)->rIndex);
}

void sort_matrix(MatrixInfo *mat) {
	printf("Sorting the matrix rowwise..\n");
	mat_t *mem = (mat_t*)malloc(sizeof(mat_t)*mat->nz);
	for (int i = 0; i < mat->nz; ++i) {
		mem[i].rIndex = mat->rIndex[i];
		mem[i].cIndex = mat->cIndex[i];
		mem[i].val = mat->val[i];
	}

	mergeSortSeq(mem, sizeof(mat_t), mat->nz, matComparator);

	// Verify sort
	/*printf("Verifying sort\n");
	for (int i = 1; i < mat->nz; ++i) {
		if (mem[i-1].rIndex > mem[i].rIndex) {
			printf("Sort Error \n");
		}
	}*/

	for (int i = 0; i < mat->nz; ++i) {
		mat->rIndex[i] = mem[i].rIndex;
		mat->cIndex[i] = mem[i].cIndex;
		mat->val[i] = mem[i].val;
	}

	free(mem);
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
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
	putProduct_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_res);
	cudaDeviceSynchronize();

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	printf("Segment scan Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

	/*Deallocate.*/
	cudaMemcpy(res->val, d_res, res->nz*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_cIndex);
	cudaFree(d_rIndex);
	cudaFree(d_val);
	cudaFree(d_vec);
	cudaFree(d_res);
}
