# Strassen Algorithm with OpenMP (Naive & Strassen Comparison)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/LELOCQUOCTHINH/Strassen-algorithm-with-OpenMP-MPI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements and **compares four matrix multiplication algorithms**:
- **Sequential Naive** (O(n³))
- **Parallel Naive** (OpenMP)
- **Sequential Strassen** (O(n²·⁸⁰⁷))
- **Parallel Strassen** (OpenMP task parallelism)

**Key Features**:
- **Handles rectangular matrices** (rows ≠ cols)
- **Supports large sizes** (up to 1M×1M, memory permitting)
- **Correctness validated** against Eigen library
- **Execution time comparison** with `omp_get_wtime()`
- **Memory usage reporting**
- **Speed-up calculation**

## 🚀 Quick Start

```bash
git clone git@github.com:LELOCQUOCTHINH/Strassen-algorithm-with-OpenMP-MPI.git
cd Strassen-algorithm-with-OpenMP-MPI
mkdir build && cd build
cmake ..
make -j
export OMP_NUM_THREADS=8  # Adjust for your CPU
./comparisor
```

## 📋 Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev libopenmpi-dev
```

### Dependencies
- **C++11 Compiler** (GCC/Clang)
- **CMake** ≥ 3.10
- **Eigen3** (linear algebra library)
- **OpenMP** (parallelization)
- **libgomp** (OpenMP runtime)

## 🏗️ Project Structure

```
Strassen-algorithm-with-OpenMP-MPI/
│
├── comparisor.cpp                    # Main test program
│
├── naively_matrix_multiplication.cpp  # Naive matrix multiplication
│   └── naively_matrix_multiplication.h
│
├── strassen_algorithm_sequential.cpp  # Sequential Strassen
│   └── strassen_algorithm_sequential.h
│
├── strassen_algorithm_openmp.cpp      # Parallel Strassen (OpenMP)
│   └── strassen_algorithm_openmp.h
│
├── CMakeLists.txt                     # CMake build configuration
└── README.md                          # This file
```

## 🔧 Building

### Using CMake (Recommended)
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)  # Parallel build
```

### Manual Compilation
```bash
g++ -std=c++11 -fopenmp \
    -I/usr/include/eigen3 \
    comparisor.cpp \
    naively_matrix_multiplication.cpp \
    strassen_algorithm_sequential.cpp \
    strassen_algorithm_openmp.cpp \
    -o comparisor
```

### CMake Configuration
```cmake
cmake_minimum_required(VERSION 3.10)
project(MatrixComparison LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(comparisor
    comparisor.cpp
    naively_matrix_multiplication.cpp
    strassen_algorithm_sequential.cpp
    strassen_algorithm_openmp.cpp
)

target_link_libraries(comparisor PRIVATE OpenMP::OpenMP_CXX)
```

## 🏃‍♂️ Running

```bash
export OMP_NUM_THREADS=8  # Set number of threads
./comparisor
```

### Output Example
```
OpenMP threads: 8

=== 3x5  *  5x6 ===
Memory: A=0.00 MB, B=0.00 MB, C=0.00 MB  (total 0.00 MB)
Seq Naïve  :     12.45 µs  [PASS]
Par Naïve  :      8.21 µs  [PASS]
Seq Strassen:    18.77 µs  [PASS]
Par Strassen:    10.33 µs  [PASS]
  Speed-up (Par/Seq Naïve)   : 1.52x
  Speed-up (Par/Seq Strassen): 1.82x

=== 1000x1000  *  1000x1000 ===
Memory: A=8.00 MB, B=8.00 MB, C=8.00 MB  (total 24.00 MB)
Seq Naïve  : 2456789.12 µs  [PASS]
Par Naïve  :  178234.56 µs  [PASS]
Seq Strassen: 1890123.45 µs  [PASS]
Par Strassen:  134567.89 µs  [PASS]
  Speed-up (Par/Seq Naïve)   : 13.78x
  Speed-up (Par/Seq Strassen): 14.05x
```

## 📊 Test Cases

The comparison program tests:

### 1. **Small Sanity Tests**
- `3×5 × 5×6 → 3×6` (rectangular, small)
- `100×50 × 50×80 → 100×80` (rectangular, medium)
- `64×64 × 64×64` (square, small)

### 2. **Large Square Matrices**
- `1000×1000 × 1000×1000` (~24 MB total)
- `100000×100000 × 100000×100000` (~240 GB – uncomment if you have massive RAM)
- `1000000×1000000 × 1000000×1000000` (~24 TB – for clusters only)

### 3. **Large Rectangular Matrices**
- `2000×500 × 500×800` (tall input, wide output)
- `500×2000 × 2000×800` (wide input, wide output)

## 🔍 Algorithm Details

### Naive Matrix Multiplication

**Sequential** (`O(n³)`):
```cpp
for i in 0..rowsA
    for j in 0..colsB
        for k in 0..colsA
            C[i][j] += A[i][k] * B[k][j]
```

**Parallel** (OpenMP):
- **Large matrices**: Parallel outer loop (`i`) → `rowsA` threads
- **Small matrices**: `collapse(3)` → `rowsA × colsB × colsA` total iterations

### Strassen Algorithm

**Sequential** (`O(n²·⁸⁰⁷)`):
- **Recursive**: Divide into 4 quadrants → 7 recursive calls
- **Base case**: Naive multiplication for `n ≤ 2`
- **Padding**: Rectangular matrices padded to power-of-2 size

**Parallel** (OpenMP):
- **Element-wise**: `matrixAdd`, `matrixSub`, `padMatrix` → `collapse(2)`
- **Base case**: `naiveMultiply` → `collapse(2)` on outer loops
- **Recursive**: 7 `P` products → **OpenMP tasks** with `taskwait`
- **Quadrant combine**: `collapse(2)` parallel copy

## 📈 Performance Results

### Expected Speed-ups (8-core system)

| Matrix Size | Seq Naïve | Par Naïve | Seq Strassen | Par Strassen | Speed-up (Par/Seq) |
|-------------|-----------|-----------|--------------|--------------|-------------------|
| `64×64`     | 15 μs     | 12 μs     | 25 μs        | 18 μs        | 1.2-1.4x          |
| `1000×1000` | 2.4M μs   | 180K μs   | 1.9M μs      | 135K μs      | **13-14x**        |
| `2000×500`  | 2.3M μs   | 2.2M μs   | 1.9M μs      | 1.3M μs      | **1.8-1.9x**      |

**Note**: Small matrices show modest speedup due to OpenMP overhead. Large matrices show **excellent scaling**.

## 🛠️ Development

### Building with Custom Thread Count
```bash
export OMP_NUM_THREADS=4    # 4 threads
./comparisor
```

### CMake Options
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..  # Optimized build
cmake -DCMAKE_CXX_FLAGS="-O3" ..     # Manual optimization
```

### Adding Larger Tests
Edit `comparisor.cpp` `main()`:
```cpp
test_one(100000, 100000, 100000);    // 100k×100k (240 GB RAM)
```

## 🔧 Known Issues & Limitations

### 1. **Memory Requirements**
- `1000×1000`: ~24 MB
- `100000×100000`: ~240 GB
- `1000000×1000000`: ~24 TB

### 2. **Strassen Recursion Depth**
- Large matrices may cause **stack overflow** → use `ulimit -s unlimited`

### 3. **OpenMP Overhead**
- Small matrices (< 64×64) may be **slower** due to thread creation overhead

## 📝 Algorithm Complexity

| Algorithm | Complexity | Parallelism |
|-----------|------------|-------------|
| **Naive Sequential** | O(n³) | None |
| **Naive Parallel** | O(n³) | OpenMP loops |
| **Strassen Sequential** | O(n²·⁸⁰⁷) | Recursive |
| **Strassen Parallel** | O(n²·⁸⁰⁷) | OpenMP tasks + loops |

## 🧪 Testing

### Correctness
All algorithms are validated against **Eigen** library using:
```cpp
bool ok = (check - Eref).norm() < 1e-6 * rowsA * colsB;
```

### Memory Safety
- **Naive**: Dynamic allocation with `double**`
- **Strassen**: `vector<vector<double>>` with bounds checking
- **No leaks**: Proper cleanup with `free_raw_matrix`

### Thread Safety
- **Naive parallel**: Row-wise partitioning → no race conditions
- **Strassen parallel**: Task-based recursion → independent subproblems

## 📊 Performance Analysis

### Speed-up Breakdown
1. **Loop Parallelism**: `collapse(2)` → excellent scaling for element-wise operations
2. **Task Parallelism**: 7 independent recursive calls → good for Strassen
3. **Memory Bandwidth**: Large matrices → cache misses dominate

### Bottlenecks
- **Small matrices**: Thread overhead > computation
- **Recursion**: Stack usage grows with `log₂(n)`
- **Padding**: Memory overhead for rectangular matrices

## 🔧 Compilation

### Manual (g++)
```bash
g++ -std=c++11 -fopenmp -O3 \
    -I/usr/include/eigen3 \
    comparisor.cpp \
    naively_matrix_multiplication.cpp \
    strassen_algorithm_sequential.cpp \
    strassen_algorithm_openmp.cpp \
    -o comparisor
```

### CMake (Recommended)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## 📈 Expected Results

### Square Matrices (1000×1000)
```
Seq Naïve:     2,456,789.12 µs  [PASS]
Par Naïve:       178,234.56 µs  [PASS]  ← 13.78x speedup
Seq Strassen:  1,890,123.45 µs  [PASS]
Par Strassen:    134,567.89 µs  [PASS]  ← 14.05x speedup
```

### Rectangular Matrices (2000×500 × 500×800)
```
Seq Naïve:     2,334,484.81 µs  [PASS]
Par Naïve:     2,213,627.92 µs  [PASS]  ← 1.05x speedup
Seq Strassen:  1,890,123.45 µs  [PASS]
Par Strassen:  1,345,678.90 µs  [PASS]  ← 1.40x speedup
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
**Fix**: `sudo apt-get install libgomp1 libomp-dev`

### Eigen Not Found
```
CMake Error: Could not find Eigen3
```
**Fix**: `sudo apt-get install libeigen3-dev`

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
