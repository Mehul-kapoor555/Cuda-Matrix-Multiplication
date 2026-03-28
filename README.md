# CUDA Matrix Multiplication (cuBLAS vs Thrust)

## Overview
This project implements matrix multiplication on GPU using two different approaches:

- **cuBLAS**: NVIDIA's highly optimized linear algebra library  
- **Thrust**: High-level parallel programming library  

The goal is to compare performance and implementation complexity between optimized libraries and custom parallel approaches.

---

## Features
- GPU-based matrix multiplication using CUDA  
- Implementation using cuBLAS and Thrust  
- Performance comparison between CPU and GPU  
- Demonstrates parallel computation techniques  

---

## Technologies Used
- CUDA  
- C++  
- cuBLAS  
- Thrust  

---

## Project Structure

src/

├── cublas_matmul.cu

└── thrust_matmul.cu


---

## How to Compile

Make sure CUDA Toolkit is installed.

### Compile cuBLAS version
```bash
nvcc src/cublas_matmul.cu -lcublas -o cublas_matmul
```

### Compile Thrust version
```bash
nvcc src/thrust_matmul.cu -o thrust_matmul
```

## How to Run
```bash
./cublas_matmul
./thrust_matmul
```

## Results
- cuBLAS provides highly optimized performance for matrix multiplication
- Thrust implementation demonstrates parallel abstraction but is less efficient
- GPU implementation significantly outperforms CPU for large matrix sizes

## Performance Comparison

This project compares matrix multiplication performance across:

- CPU (serial computation)
- GPU using Thrust
- GPU using cuBLAS

### Key Observations

- GPU implementation significantly outperforms CPU due to massive parallelism
- Thrust provides a simple abstraction for parallel programming but is not fully optimized
- cuBLAS achieves the best performance by leveraging highly optimized GPU kernels

## CPU vs GPU Architecture

### CPU (Central Processing Unit)
- Designed for serial processing
- Contains a small number of powerful cores
- Each core has large cache and high clock speed
- Optimized for sequential tasks

### GPU (Graphics Processing Unit)
- Designed for parallel computing
- Contains thousands of smaller, efficient cores
- Executes the same operation across large datasets simultaneously
- Ideal for matrix operations and linear algebra

## Thrust vs cuBLAS

### Thrust
- High-level parallel programming library
- Easy to use and flexible
- Suitable for general-purpose parallel algorithms
- Lower performance for compute-intensive operations

### cuBLAS
- NVIDIA's optimized linear algebra library
- Designed specifically for matrix and vector operations
- Uses advanced GPU optimizations
- Significantly faster than Thrust for matrix multiplication

## Performance Summary

- GPU implementations outperform CPU for large matrix sizes
- cuBLAS provides the fastest execution among all methods
- Thrust offers ease of implementation but at the cost of performance
- Parallel computation provides substantial speedup over serial execution

## Key Takeaways

- Parallel computing is essential for handling large-scale numerical computations
- GPU acceleration drastically reduces computation time for matrix operations
- Choosing the right library (cuBLAS vs Thrust) impacts performance significantly
- High-performance computing requires both algorithmic understanding and hardware awareness


## Learning Outcomes
- Understanding GPU parallelism
- Using cuBLAS for optimized computations
- Comparing different GPU programming approaches
- Performance benchmarking in CUDA

## Author
Mehul Kapoor
