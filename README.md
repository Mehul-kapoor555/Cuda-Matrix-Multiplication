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

### How to Run
```bash
./cublas_matmul
./thrust_matmul
```

## Results
cuBLAS provides highly optimized performance for matrix multiplication
Thrust implementation demonstrates parallel abstraction but is less efficient
GPU implementation significantly outperforms CPU for large matrix sizes
Learning Outcomes
Understanding GPU parallelism
Using cuBLAS for optimized computations
Comparing different GPU programming approaches
Performance benchmarking in CUDA
Author

Mehul Kapoor
