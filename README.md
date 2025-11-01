# Strassen Algorithm with OpenMP and OpenMP-CUDA (Naïve & Strassen Comparison)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/LELOCQUOCTHINH/Strassen-algorithm-with-OpenMP-MPI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements and **compares matrix multiplication algorithms** across CPU and GPU:
- **Sequential Naïve** (O(n³))
- **Parallel Naïve** (OpenMP)
- **Sequential Strassen** (O(n²·⁸⁰⁷))
- **Parallel Strassen** (OpenMP task parallelism)
- **Parallel Naïve GPU** (OpenMP offload)
- **Parallel Strassen GPU** (OpenMP offload)

To avoid OpenMP runtime conflicts between CPU tasks and GPU offload, the project is split into two executables:
- `comparisor_openmp`: CPU-only tests (Seq/Par Naïve + Seq/Par Strassen)
- `comparisor_cuda`: GPU-offload tests (Seq Naïve + Seq Strassen + GPU Naïve + GPU Strassen)

**Key Features**:
- **Handles rectangular matrices** (rows ≠ cols)
- **Supports large sizes** (up to 1M×1M, memory permitting)
- **Correctness validated** against Eigen library
- **Execution time comparison** with `omp_get_wtime()`
- **Speed-up calculation** (baseline / optimized, e.g., slower / faster for >1.0x)
- **Memory usage reporting** (in some versions)
- **Separate CPU and GPU builds** to prevent runtime deadlocks

## 🚀 Quick Start

```bash
git clone git@github.com:LELOCQUOCTHINH/Strassen-algorithm-with-OpenMP-MPI.git
cd Strassen-algorithm-with-OpenMP-MPI
mkdir build && cd build
cmake ..
make -j
export OMP_NUM_THREADS=8  # Adjust for your CPU
export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH  # For GPU runtime

./comparisor_openmp   # CPU-only tests
./comparisor_cuda     # GPU-offload tests
```

## 📋 Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev libomp-dev
```

### For GPU Offload
- **Clang-18** or newer with OpenMP offload support
- **CUDA Toolkit** (for nvptx target)
- **NVIDIA GPU** (sm_86 or compatible)

### Dependencies
- **C++14 Compiler** (Clang-18 recommended)
- **CMake** ≥ 3.18
- **Eigen3** (linear algebra library)
- **OpenMP** (parallelization, with GPU offload for CUDA version)
- **libgomp** / **libomp** (OpenMP runtime)

## 🏗️ Project Structure

```
Strassen-algorithm-with-OpenMP-MPI/
│
├── comparisor_openmp.cpp             # CPU-only test program
├── comparisor_cuda.cpp               # GPU-offload test program
│
├── naively_matrix_multiplication.cpp # Naive matrix multiplication (Seq/Par/GPU)
│   └── naively_matrix_multiplication.h
│
├── strassen_algorithm_sequential.cpp # Sequential Strassen
│   └── strassen_algorithm_sequential.h
│
├── strassen_algorithm_openmp.cpp     # Parallel Strassen (OpenMP CPU)
│   └── strassen_algorithm_openmp.h
│
├── strassen_algorithm_openmp_cuda.cpp # Parallel Strassen (OpenMP GPU offload)
│   └── strassen_algorithm_openmp_cuda.h
│
├── CMakeLists.txt                    # CMake build configuration
└── README.md                          # This file
```

## 📊 Sample Output

### CPU-Only (`comparisor_openmp`)

```
OpenMP threads: 20

=== 100x50  *  50x80 ===
Seq Naïve:        220.06 µs  [PASS]
Par Naïve:        200.99 µs  [PASS]
Seq Strassen: 122505.90 µs  [PASS]
Par Strassen:   30476.09 µs  [PASS]

--- Speed-ups (baseline / optimized) ---
  Strassen Seq / Naive Seq       : 556.59x
  Naive Par / Naive Seq          : 1.09x
  Strassen Par / Strassen Seq    : 4.02x
  Strassen Par / Naive Par       : 151.52x
```

### GPU-Offload (`comparisor_cuda`)

```
OpenMP threads: 20

=== 100x50  *  50x80 ===
Seq Naïve:        319.00 µs  [PASS]
Seq Strassen:  114183.90 µs  [PASS]
GPU Naïve:       2927.06 µs  [PASS]
GPU Strassen:    2691.03 µs  [PASS]

--- Speed-ups (baseline / optimized) ---
  Strassen Seq / Naive Seq       : 357.94x
  Naive GPU / Naive Seq          : 0.11x
  Strassen GPU / Strassen Seq    : 42.43x
  Strassen GPU / Naive GPU       : 1.09x
```

## 🛡️ Troubleshooting

### Clock Skew Warning
```
warning: Clock skew detected. Your build may be incomplete.
```
**Fix**: `find . -exec touch {} \;` or `make clean && make`

### OpenMP Not Found
```
CMake Error: Could not find OpenMP
```
**Fix**: `sudo apt-get install libomp-dev`

### Eigen Not Found
```
CMake Error: Could not find Eigen3
```
**Fix**: `sudo apt-get install libeigen3-dev`

### libomptarget.so Not Found (GPU Runtime)
```
error while loading shared libraries: libomptarget.so.18.1
```
**Fix**: `export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH`

### Segmentation Fault
```
=== 2000x500  *  500x800 ===
Seq Naïve  : PASS
Par Naïve  : PASS
Segmentation fault
```
**Cause**: Stack overflow in Strassen recursion  
**Fix**: `ulimit -s unlimited`

### Memory Error
```
malloc(): cannot allocate memory
```
**Cause**: Matrix too large for available RAM  
**Fix**: Use smaller test sizes or run on cluster

### PASS/FAIL Checks Fail
```
Seq Strassen:  [FAIL]
```
**Cause**: Numerical instability or incorrect padding/unpadding  
**Fix**: Verify matrix dimensions and use double precision

## 🔗 Related Projects

- [Eigen](http://eigen.tuxfamily.org) – High-performance linear algebra
- [OpenMP](https://www.openmp.org) – Shared memory parallelism
- [BLAS/LAPACK](https://www.netlib.org/blas/) – Standard linear algebra

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

**Author**: LELOCQUOCTHINH  
**GitHub**: [LELOCQUOCTHINH](https://github.com/LELOCQUOCTHINH)  
**Repository**: [Strassen-algorithm-with-OpenMP-MPI](https://github.com/LELOCQUOCTHINH/Strassen-algorithm-with-OpenMP-MPI)

---

*Made with ❤️ for parallel computing enthusiasts*
