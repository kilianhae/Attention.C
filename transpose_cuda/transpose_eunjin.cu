#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cublas_v2.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_DIM 32

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
            // Uncomment the next line to print each incorrect element
            // printf("Mismatch at index %d, CPU: %f, GPU: %f\n", i, h_T_CPU[i], h_T[i]);
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

__global__ void transposeSharedMem(float *d_A, float *d_T, int M, int N) {
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




void run_transpose_cublas(float *A, float *C, int M,int N) {
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;  // cuBLAS functions status
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    // loop over batchsize and head
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            // get the i-th batch and j-th head
            float* Aij = A;
            float* Cij = C;

            // perform matrix transposition using cublasSgeam
            stat = cublasSgeam(handle,
                               CUBLAS_OP_T,  // transpose A
                               CUBLAS_OP_N,  // do not transpose B (NULL)
                               N,  // number of rows of A^T
                               M,  // number of columns of A^T
                               &alpha,
                               Aij,  // pointer to A
                               N,  // leading dimension of A
                               &beta,
                               NULL,  // B is NULL
                               N,  // set ldb to a valid value
                               Cij,  // pointer to C
                               N);  // leading dimension of C
        }
    }
    cublasDestroy(handle);
}




int main(int argc, char *argv[]) {
    // Set matrix size
    // int M = atoi(argv[1]);
    // int N = atoi(argv[2]);
    int M = 4096;
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

    
	// launch kernel instance
	dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);

    double start_total_GPU = timeStamp();
	//transposeNaive<<<gridDim, blockDim>>>(d_A, d_T, M, N);
	//transposeSharedMem<<<gridDim, blockDim>>>(d_A, d_T, M, N);
    run_transpose_cublas(d_A, d_T, M, N);
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    double end_total_GPU = timeStamp();
    float total_GPU_time = end_total_GPU - start_total_GPU;

	// copy result back to host
	gpuErrchk(cudaMemcpy(h_T, d_T, bytes, cudaMemcpyDeviceToHost));


    printf("GPU execution time: %.4f milliseconds\n", total_GPU_time);

  	// display results
    // displayResults(h_A, h_T, M, N, 30, 40);
    validateResults(h_A, h_T, M, N);

	// clean up data
    gpuErrchk(cudaFreeHost(h_A));
    gpuErrchk(cudaFreeHost(h_T));
    gpuErrchk(cudaFree(d_A)); 
    gpuErrchk(cudaFree(d_T));
    gpuErrchk(cudaDeviceReset());

	return 0;
}

// $ nvcc -arch sm_89 transpose_eunjin.cu -o transpose_eunjin
// $ ./transpose_eunjin 5 4

// $ nvcc -arch sm_89 transpose_cuda/transpose_eunjin.cu -o transpose_cuda/transpose_eunjin
// $ transpose_cuda/transpose_eunjin 5 4