// CUDA Matrix Multiplication using Thrust
// Author: Mehul Kapoor
// Implements GPU-accelerated matrix multiplication


#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <chrono>

// Functor for matrix multiplication
struct MatMulFunctor {
    thrust::device_ptr<const float> A;
    thrust::device_ptr<const float> B;
    int M, K, N;

    MatMulFunctor(thrust::device_ptr<const float> A_, 
                  thrust::device_ptr<const float> B_, 
                  int M_, int K_, int N_)
        : A(A_), B(B_), M(M_), K(K_), N(N_) {}

    __host__ __device__
    float operator()(int index) const {
        int row = index / N;
        int col = index % N;
        float sum = 0.0f;
        
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        return sum;
    }
};

// Function to perform matrix multiplication using Thrust (GPU)
double matmul_thrust(const thrust::host_vector<float>& h_A, 
                     const thrust::host_vector<float>& h_B, 
                     thrust::host_vector<float>& h_C, 
                     int M, int K, int N) {
    auto start = std::chrono::high_resolution_clock::now();

    // Transfer to GPU
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(M * N);

    // Apply matrix multiplication in parallel
    thrust::counting_iterator<int> indices(0);
    thrust::transform(indices, indices + (M * N), d_C.begin(), 
                      MatMulFunctor(d_A.data(), d_B.data(), M, K, N));

    
    auto end = std::chrono::high_resolution_clock::now();

    // Copy back result
    h_C = d_C;
    
    return std::chrono::duration<double>(end - start).count(); // Return execution time
}

// Function to perform CPU matrix multiplication
double matmul_cpu(const thrust::host_vector<float>& h_A, 
                  const thrust::host_vector<float>& h_B, 
                  thrust::host_vector<float>& h_C, 
                  int M, int K, int N) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += h_A[row * K + i] * h_B[i * N + col];
            }
            h_C[row * N + col] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count(); // Return execution time
}

// Function to randomly initialize a matrix
void initialize_matrix(thrust::host_vector<float>& matrix, int size) {
    for (int i = 0; i < size; ++i)
        matrix[i] = static_cast<float>(rand() % 20 + 1);
}

int main() {
    int M = 2000, K = 2000, N = 2000;
    
    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(K * N);
    thrust::host_vector<float> h_C_gpu(M * N);
    thrust::host_vector<float> h_C_cpu(M * N);

    // Initialize matrices with random values
    initialize_matrix(h_A, M * K);
    initialize_matrix(h_B, K * N);

    // Compute on GPU
    double gpu_time = matmul_thrust(h_A, h_B, h_C_gpu, M, K, N);
    std::cout << "GPU Time: " << gpu_time << " seconds\n";

    // Compute on CPU
    double cpu_time = matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    std::cout << "CPU Time: " << cpu_time << " seconds\n";

    return 0;
}
