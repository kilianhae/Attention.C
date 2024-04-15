#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cuda.h>
#include <chrono>


// Define TILE_DIM_X and TILE_DIM_Y if they're not already defined
#ifndef TILE_DIM_X
#define TILE_DIM_X 32
#endif

#ifndef TILE_DIM_Y
#define TILE_DIM_Y 32
#endif
// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}



__global__ void softmax_forward_kernel4(float* out, float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}



template <typename T>
__global__ void softmaxKernel2D_rows(const T* input, T* exp_sums, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    T val = 0;
    // Copy data from global memory to shared memory
    if (row < N && col < M) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = max(max_val, input[row * M + i]);
        }
        if (sizeof(T) == 8)
            val = exp(input[row * M + col] - max_val);
        else
            val = expf(input[row * M + col] - max_val);
    }
    // warp shuffle reduction
    // Use XOR mode to perform butterfly reduction
    for (int i = 16; i >= 1; i >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    // update global value for row
    if ((threadIdx.x == 0) && (row < N)) atomicAdd(exp_sums + row, val);
}

template <typename T>
__global__ void softmaxKernel2D_elementwise(const T* input, const T* exp_sums, T* output, int N, int M) {
    int row = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int col = blockIdx.x * TILE_DIM_X + threadIdx.x;
    // Compute the softmax values
    if (row < N && col < M) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = max(max_val, input[row * M + i]);
        }
        T exp_sum_row = exp_sums[row];
        if (sizeof(T) == 8)
            output[row * M + col] = exp(input[row * M + col] - max_val) / exp_sum_row;
        else
            output[row * M + col] = expf(input[row * M + col] - max_val) / exp_sum_row;
    }
}
template <typename T>
void softmax2D_rows_cpu(const T* input, T* exp_sums, int N, int M) {
    for (int row = 0; row < N; ++row) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = std::max(max_val, input[row * M + i]);
        }
        T sum = 0;
        for (int col = 0; col < M; ++col) {
            T val = exp(input[row * M + col] - max_val);
            exp_sums[row] += val;
            sum += val;
        }
        exp_sums[row] = sum;
    }
}

template <typename T>
void softmax2D_elementwise_cpu(const T* input, const T* exp_sums, T* output, int N, int M) {
    for (int row = 0; row < N; ++row) {
        T max_val = input[row * M];  // Initialize max_val with the first element of the row
        for (int i = 1; i < M; ++i) {
            max_val = std::max(max_val, input[row * M + i]);
        }
        T exp_sum_row = exp_sums[row];
        for (int col = 0; col < M; ++col) {
            output[row * M + col] = exp(input[row * M + col] - max_val) / exp_sum_row;
        }
    }
}
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

int main(int argc, char *argv[]) {
    // Parse command-line arguments
//    if (argc != 3) {
  //      std::cerr << "Usage: " << argv[0] << " <num_rows> <num_cols>" << std::endl;
    //    return 1;
   // }
    const int N = 8192;//std::stoi(argv[1]);
    const int M = 8192;//std::stoi(argv[2]);

    // Generate random input
    std::vector<float> input(N * M);
    for (int i = 0; i < N * M; ++i) {
        input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
    }

    // Allocate memory for output on CPU
    std::vector<float> output_cpu(N * M);
    std::vector<float> output_gpu(N * M);

    // Allocate device memory
    float *d_input, *d_exp_sums, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_exp_sums, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * M * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * M * sizeof(float), cudaMemcpyHostToDevice));

    // Set grid and block dimensions
    //dim3 blockDim(TILE_DIM_X, TILE_DIM_Y);
    //dim3 gridDim((M + TILE_DIM_X - 1) / TILE_DIM_X, (N + TILE_DIM_Y - 1) / TILE_DIM_Y);

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));

    // Record start event for GPU
    CUDA_CHECK(cudaEventRecord(start_gpu));

    // Launch softmax kernel for rows on GPU
    int block_size = 256;
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(d_exp_sums, d_input, N, N);
    //softmax_forward_kernel4<<<gridDim, blockDim>>>(d_input, d_exp_sums, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy exp_sums back to host
    std::vector<float> exp_sums(N);
    CUDA_CHECK(cudaMemcpy(exp_sums.data(), d_exp_sums, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Record stop event for GPU
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));

    // Calculate GPU execution time
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu));

    // Create chrono objects for timing CPU execution
    auto start_cpu = std::chrono::high_resolution_clock::now();

    // Compute softmax rows CPU for verification
    softmax2D_rows_cpu<float>(input.data(), exp_sums.data(), N, M);
    cudaDeviceSynchronize();

    // Compute softmax elementwise CPU for verification
    //softmax2D_elementwise_cpu<float>(input.data(), exp_sums.data(), output_cpu.data(), N, M);
    cudaDeviceSynchronize();

    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);

    // Print CPU execution time
    std::cout << "CPU Execution Time: " << duration_cpu.count() << " ms" << std::endl;

    // Create CUDA events for timing GPU execution
    cudaEvent_t start_gpu_elementwise, stop_gpu_elementwise;
    CUDA_CHECK(cudaEventCreate(&start_gpu_elementwise));
    CUDA_CHECK(cudaEventCreate(&stop_gpu_elementwise));

    // Record start event for GPU elementwise computation
    CUDA_CHECK(cudaEventRecord(start_gpu_elementwise));

    // Launch softmax kernel elementwise on GPU
   // softmaxKernel2D_elementwise<<<gridDim, blockDim>>>(d_input, d_exp_sums, d_output, N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record stop event for GPU elementwise computation
    CUDA_CHECK(cudaEventRecord(stop_gpu_elementwise));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu_elementwise));

    // Calculate GPU elementwise execution time
    float gpu_elementwise_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elementwise_time, start_gpu_elementwise, stop_gpu_elementwise));

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(output_gpu.data(), d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    bool passed = true;
    for (int i = 0; i < N * M; ++i) {
        if (std::abs(output_cpu[i] - output_gpu[i]) > 1e-5) {
            std::cout << "Verification failed at index " << i << ": CPU = " << output_cpu[i] << ", GPU = " << output_gpu[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "Verification passed!" << std::endl;
    }

    // Print GPU execution time
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;
    //std::cout << "GPU Elementwise Execution Time: " << gpu_elementwise_time << " ms" << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_exp_sums));
    CUDA_CHECK(cudaFree(d_output));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));
    CUDA_CHECK(cudaEventDestroy(start_gpu_elementwise));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_elementwise));

    return 0;
}


