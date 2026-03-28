// GPU Matrix Multiplication using cuBLAS
// Author: Mehul Kapoor
// Description: Implements high-performance matrix multiplication
// using NVIDIA cuBLAS library for optimized GPU computation.


#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <ctime>

// Error handling macros
#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

#define CHECK_CUBLAS(call)                                                 \
    {                                                                      \
        cublasStatus_t status = call;                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                  \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << ", status: " << status << std::endl;              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

// Matrix dimensions (change these values as needed)
constexpr int M = 1024; // Rows in A and C
constexpr int N = 1024; // Columns in B and C
constexpr int K = 1024; // Columns in A, Rows in B

// Function to generate a matrix with random values between 1 and 20
void generateRandomMatrix(std::vector<float> &matrix, int size)
{
    for (int i = 0; i < size; ++i)
        matrix[i] = static_cast<float>(rand() % 20 + 1);
}

// Function to measure GPU execution time
float gpuMatrixMultiply(cublasHandle_t handle, float *d_A, float *d_B, float *d_C)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start time
    CHECK_CUDA(cudaEventRecord(start));

    // Perform matrix multiplication using cuBLAS
    CHECK_CUBLAS(cublasSgemm(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N, // No transposition
                              M, N, K,                 // Matrix dimensions
                              &alpha,
                              d_A, M, // Leading dimension of A
                              d_B, K, // Leading dimension of B
                              &beta,
                              d_C, M)); // Leading dimension of C

    // Record stop time
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds; // Return execution time in milliseconds
}

int main()
{
    srand(time(0)); // Seed random number generator

    // Allocate and initialize host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f); // Initialize result matrix to zero

    generateRandomMatrix(h_A, M * K);
    generateRandomMatrix(h_B, K * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    // Copy host matrices to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform matrix multiplication and measure execution time
    float gpuTime = gpuMatrixMultiply(handle, d_A, d_B, d_C);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print execution time
    std::cout << "Matrix multiplication completed in " << gpuTime << " ms." << std::endl;

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
