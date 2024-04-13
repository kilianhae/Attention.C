// assume row wise layout called from pytorch via pybind
// Eunjin: Transpose kernel call fucntion for pybind
// Aditya: Softmax Kernel and call function
// Kilian: Matrix Multiplication



#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
 #include <cudnn.h>





double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


void run_sgemm_cublas_batched(torch::Tensor A, torch::Tensor B, torch::Tensor C, bool transpose, int batchsize, int head, int M, int K, int N){
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    const float alpha = 1.0;
    const float beta = 0.0;
    // loop over batchsize and head
    // make array of pointers of elements of A with stride of M*K
    // make array of pointers of elements of B with stride of K*N
    // make array of pointers of elelments of C with stride of M*N

    float *Aarray[batchsize*head];
    float *Barray[batchsize*head];
    float *Carray[batchsize*head];

    for (int i = 0; i < batchsize; i++) {
        for (int j = 0; j < head; j++) {
            Aarray[i*head+j] = A[i][j].data_ptr<float>();;
            Barray[i*head+j] = B[i][j].data_ptr<float>();;
            Carray[i*head+j] = C[i][j].data_ptr<float>();;
        }
    }


    float **Aarray_d;
    float **Barray_d;
    float **Carray_d;
    cudaMalloc((void**)&Aarray_d, batchsize*head*sizeof(float*));
    cudaMalloc((void**)&Barray_d, batchsize*head*sizeof(float*));
    cudaMalloc((void**)&Carray_d, batchsize*head*sizeof(float*));

    cudaMemcpy(Aarray_d, Aarray, batchsize*head*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(Barray_d, Barray, batchsize*head*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(Carray_d, Carray, batchsize*head*sizeof(float*), cudaMemcpyHostToDevice);
    if(transpose){
      //stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 3000, 3000, 4000, &alpha, Barray_d, 4000, Aarray_d, 4000, &beta, Carray_d, 3000, B.size(0)*B.size(1));
      double start, end;
      start = getTimeStamp();
      cudaDeviceSynchronize();
      // if its transpose then B will be NxK and A will be MxK
      stat = cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, Barray_d, K , Aarray_d, K, &beta, Carray_d, N, head*batchsize);
      cudaDeviceSynchronize();
      end = getTimeStamp();
      printf("Time taken short: %f\n", end-start);
    }
    else{
      double start, end;
      start = getTimeStamp();
      cudaDeviceSynchronize();
      // if its not transpose then B will be KxN and A will be MxK
      stat = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, Barray_d, N, Aarray_d, K, &beta, Carray_d, N, head*batchsize);
      cudaDeviceSynchronize();
      end = getTimeStamp();
      printf("Time taken shaort: %f\n", end-start);
      }
}

void softmax_cudnn(torch::Tensor S, torch::Tensor A, int s) {
    // Set up cuDNN
    
    for (int i = 0; i < S.size(0); i++) {
        for (int j = 0; j < S.size(1); j++) {
            // get the i-th batch and j-th head
            torch::Tensor Sij = S[i][j];
            torch::Tensor Aij = A[i][j];


            cudnnHandle_t cudnn;
            cudnnCreate(&cudnn);
            
            cudnnTensorDescriptor_t input_desc, output_desc;
            cudnnCreateTensorDescriptor(&input_desc);
            cudnnCreateTensorDescriptor(&output_desc);
            
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s, s, 1, 1);
            cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s, s, 1, 1);
            
            // Perform softmax operation
            float alpha = 1.0f, beta = 0.0f;
            cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, input_desc, Sij.data_ptr<float>(), &beta, output_desc, Aij.data_ptr<float>());
            
            // Clean up
            cudnnDestroyTensorDescriptor(input_desc);
            cudnnDestroyTensorDescriptor(output_desc);
            cudnnDestroy(cudnn);
        }
        }
    }



__global__ void transposeSharedMem(float *d_A, float *d_T, int M, int N) {
	__shared__ float tile[32][32+1];
	
	unsigned int row = blockIdx.y * 32 + threadIdx.y;
	unsigned int col = blockIdx.x * 32 + threadIdx.x;
    unsigned int index_in = row * N + col;
	
    if((row < M) && (col < N) && (index_in < M*N)) {
        tile[threadIdx.y][threadIdx.x] = d_A[index_in];
	}
    
	__syncthreads();
    
	row = blockIdx.y * 32 + threadIdx.x;
	col = blockIdx.x * 32 + threadIdx.y;
	if((row < M) && (col < N)) {
        unsigned int index_out = col * M + row;
		d_T[index_out] = tile[threadIdx.x][threadIdx.y];
	}
}


torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool transpose) {
    // double start, end;
    // start = getTimeStamp();
    cudaDeviceSynchronize();
    int b = Q.size(0);
    int h = Q.size(1);
    int s = Q.size(2);
    int d = Q.size(3);


    // make tensor S of size {b, h, s, d}, the value doesnt matter
    torch::Tensor S = torch::empty({b, h, s, s}, torch::kCUDA);

    

    // run matmul with transpose
    if (transpose){
        run_sgemm_cublas_batched(Q, K, S, true, b, h, s, d, s);

        printf("Transposed\n");
    }
    else{
        dim3 blockDim(32, 32); // each thread will process 4 cosnecutive 
	    dim3 gridDim((d + 32 - 1)/32, (s + 32 - 1)/32);
        torch::Tensor Kt = torch::empty({b, h, d, s}, torch::kCUDA);
        transposeSharedMem<<<gridDim, blockDim>>>(K.data_ptr<float>(),Kt.data_ptr<float>(),s,d);
        run_sgemm_cublas_batched(Q, Kt, S, false, b, h, s, d, s);
        printf("Not Transposed\n");
    }
    

    // Now S will hold the unnormalized scores and we apply softmax on it
    
    torch::Tensor A = torch::empty({b, h, s, s}, torch::kCUDA);
    
    softmax_cudnn(S,A,s);
    //S.reset();

    // remove S from gpu memory as its no longer needed

    // Now we multiply the softmaxed scores with V

    torch::Tensor O = torch::empty({b, h, s, d}, torch::kCUDA);

    run_sgemm_cublas_batched(A, V, O, false, b, h, s, s, d);
    //A.reset();
    cudaDeviceSynchronize();
    // end = getTimeStamp();
    // printf("Time taken: %lf\n", (end-start));
    return O;
}