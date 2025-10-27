# Strassen Algorithm with OpenMP and MPI

This repository contains an implementation of matrix multiplication in C++, currently featuring a naive matrix multiplication algorithm with sequential and parallel (OpenMP) versions. The project is intended to evolve into an optimized implementation of Strassen's matrix multiplication algorithm, enhanced with OpenMP for parallelization and MPI for distributed computing. And comparision performance between them

## Current Implementation

The current codebase in `naively_matrix_multiplication.cpp` implements:
- **Sequential Matrix Multiplication**: A standard O(n³) algorithm for multiplying two matrices.
- **Parallel Matrix Multiplication**: An OpenMP-parallelized version of the naive algorithm, distributing the outer loop across threads.
- **Validation**: Results are validated against the Eigen library's matrix multiplication to ensure correctness.
- **Timing**: Execution times are measured using `omp_get_wtime()` for both sequential and parallel implementations.

Future updates will include:
- Strassen's matrix multiplication algorithm (O(n²·⁸⁰⁷)).
- Parallelization enhancements using OpenMP.
- Distributed computing with MPI for large-scale matrices.

## Prerequisites

To build and run the project, you need:
- **C++ Compiler**: GCC or any C++11-compatible compiler.
- **CMake**: Version 3.10 or higher.
- **Eigen**: A C++ library for linear algebra.
- **OpenMP**: For parallel execution.

### Installation (Ubuntu/Debian)
1. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential cmake libeigen3-dev libopenmpi-dev
2. Build the project at the current directory:
   ```bash
   cmake .
   make
- Or you can build the project at the build directory
    ```bash
    mkdir build
    cd build
    cmake ..
    make
3. Run the project
   ```bash
   ./naively_matrix_multiplication
