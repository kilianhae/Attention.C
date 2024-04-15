#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <algorithm>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_DIM 32
#define BB 1
#define H 8

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code == cudaSuccess) return;
    fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}

double timeStamp() {
    struct timeval tv; 
    gettimeofday(&tv, NULL);
    return tv.tv_usec / 1000.0 + tv.tv_sec;
}

void displayResults(float *A, float *T, int M, int N, int fromIdx, int toIdx){
    // display results
	printf("Matrix A: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
        if (i >= fromIdx && i < toIdx) {
            for (int j = 0; j < N; ++j) {
                if (j >= fromIdx && j < toIdx) {
                    printf("A: %.2f ", A[i * N + j]);
                } else {
                    continue;
                }
            }
        } else {
            continue;
        }
		printf("\n");
	}

	printf("----------\n");
	printf("Transpose: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
        if (i >= fromIdx && i < toIdx) {
            for (int j = 0; j < M; ++j) {
                if (j >= fromIdx && j < toIdx) {
                    printf("%.2f ", T[i * M + j]);
                } else {
                    continue;
                }
            }
        } else {
            continue;
        }
		printf("\n");
	}
}

void transposeCPU(float *A, float *T, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T[j * M + i] = A[i * N + j];
        }
    }
}

void validateResults(float *h_A, float *h_T, int M, int N){
    // Allocate memory for the transpose matrix on CPU
    float *h_T_CPU = (float *)malloc(M * N * sizeof(float));
    // Transpose matrix A on CPU
    transposeCPU(h_A, h_T_CPU, M, N);

    // Validate the results
    int incorrectCount = 0;
    for (int i = 0; i < M * N; ++i) {
        if (abs(h_T_CPU[i] - h_T[i]) > 1e-5) {
            incorrectCount++;
        }
    }
    
    if (incorrectCount == 0) {
        printf("Validation Passed!\n");
    } else {
        printf("Validation Failed: %d elements incorrect.\n", incorrectCount);
    }

    // Clean up CPU transpose matrix
    free(h_T_CPU);
}

__global__ void transposeNaive(float *d_A, float *d_T, int M, int N) {
	int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
	int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

	if (row < M && col < N) {
		d_T[col * M + row] = d_A[row * N + col];
	}
}

__global__ void transposeCoalesced(float *d_A, float *d_T, int M, int N) {
    // avoid bank conflicts so offsett by 1 element
	__shared__ float tile[TILE_DIM][TILE_DIM+1];
	
	unsigned int row = blockIdx.y * TILE_DIM + threadIdx.y;
	unsigned int col = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int index_in = row * N + col;
	
    if((row < M) && (col < N) && (index_in < M*N)) {
        tile[threadIdx.y][threadIdx.x] = d_A[index_in];
	}
    
	__syncthreads();
    
	row = blockIdx.y * TILE_DIM + threadIdx.x;
	col = blockIdx.x * TILE_DIM + threadIdx.y;
	if((row < M) && (col < N)) {
        unsigned int index_out = col * M + row;
		d_T[index_out] = tile[threadIdx.x][threadIdx.y];
	}
}


__global__ void copySharedMem(float *d_A, float *d_T, int M, int N)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
    // block is 8 x 32 so x is 8 and y is 32
  int x = blockIdx.x * TILE_DIM + threadIdx.x*4; // 0,4,8,12,16,20,24,28
  int y = blockIdx.y * TILE_DIM + threadIdx.y; // 0,1,2,3,...,31
    //   printf("%d",blockIdx.y);
    //   printf("%d",blockIdx.x);
  int width = gridDim.x * TILE_DIM;

    if (x>=N || y>=M){return;}
    // load all your elements into shared memory
    for (int j=0; j<4;j+=1){
        tile[threadIdx.y*TILE_DIM+threadIdx.x*4+j]=d_A[y*N+x+j]; //thread 0: 0,1,2,3, thread 1: 4,5,6,7 ... ,28,29,30,31, loading is done with offset
    }
    
    __syncthreads();
  // shared memory now contain an exact copy of the tile. We need to load this back coalesced now

    // calculate the elelements that this thread will load back and to where it will load back
    //idx=(threadIdx.x*BLOCK_ROWS+j)*TILE_DIM
    int idy = threadIdx.y;
    int idx;

    for (int j = 0; j < 4; j += 1){
        idx=(threadIdx.x*4+j); // 
        d_T[blockIdx.x*TILE_DIM*N + blockIdx.y*TILE_DIM + idy*M + idx]=tile[idx*TILE_DIM+idy];
    }}


__global__ void copySharedMem_coalesced(float *d_A, float *d_T, int M, int N)
{
  __shared__ float tile[TILE_DIM * TILE_DIM];
    // block is 8 x 32 so x is 8 and y is 32
    int x = blockIdx.x * TILE_DIM + threadIdx.x; // 0,4,8,12,16,20,24,28
    int y = blockIdx.y * TILE_DIM; // 0,1,2,3,...,31
    //   printf("%d",blockIdx.y);
    //   printf("%d",blockIdx.x);
    int width = gridDim.x * TILE_DIM;

    if (x>=N || y>=M){return;}
    // load all your elements into shared memory
    for (int j=threadIdx.y; j<TILE_DIM+threadIdx.y;j+=BLOCK_ROWS){
        tile[j*TILE_DIM+threadIdx.x]=d_A[(y+j)*N+x]; //thread 0: 0,1,2,3, thread 1: 4,5,6,7 ... ,28,29,30,31, loading is done with offset
    }
    
    __syncthreads();
    // shared memory now contain an exact copy of the tile. We need to load this back coalesced now

    // calculate the elelements that this thread will load back and to where it will load back
    //idx=(threadIdx.x*BLOCK_ROWS+j)*TILE_DIM
    int idy;
    int ix=threadIdx.x*TILE_DIM;
    int xx = blockIdx.x*TILE_DIM*N;
    int yy = blockIdx.y*TILE_DIM;
    for (int j = threadIdx.y; j < TILE_DIM+threadIdx.y; j += BLOCK_ROWS){
        d_T[xx + yy + j*M + threadIdx.x]=tile[ix+j];
    }

}


int main(int argc, char *argv[]) {
    // Set matrix size
    // int M = atoi(argv[1]);
    // int N = atoi(argv[2]);
    int M = 6294;
    int N = 64;
    if (M <= 0 || N <= 0) return 0;
    size_t bytes = M * N * sizeof(float);

	float *h_A, *h_T;
	float *d_A, *d_T;

	// allocate host memory
    gpuErrchk(cudaHostAlloc((void **)&h_A, bytes, cudaHostAllocMapped));
    gpuErrchk(cudaHostAlloc((void **)&h_T, bytes, cudaHostAllocMapped));
    
	// initialize data
	for (int i = 0; i < M * N; ++i) {
        h_A[i] = (float)(rand() % 10 + 1);
	}

    // allocate device memory
    gpuErrchk(cudaMalloc(&d_A, bytes));
    gpuErrchk(cudaMalloc(&d_T, bytes));
    

	// copy host data to device
	gpuErrchk(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

	dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
    double start_total_GPU = timeStamp();
    
	// transposeNaive<<<gridDim, blockDim>>>(d_A, d_T, M, N);
	transposeCoalesced<<<gridDim, blockDim>>>(d_A, d_T, M, N);
    // run_transpose_cublas(d_A, d_T, M, N);
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    double end_total_GPU = timeStamp();
    float total_GPU_time = end_total_GPU - start_total_GPU;

	// copy result back to host
	gpuErrchk(cudaMemcpy(h_T, d_T, bytes, cudaMemcpyDeviceToHost));

    printf("GPU execution time: %.4f milliseconds\n", total_GPU_time);

    validateResults(h_A, h_T, M, N);

	// clean up data
    gpuErrchk(cudaFreeHost(h_A));
    gpuErrchk(cudaFreeHost(h_T));
    gpuErrchk(cudaFree(d_A)); 
    gpuErrchk(cudaFree(d_T));
    gpuErrchk(cudaDeviceReset());

	return 0;
}
