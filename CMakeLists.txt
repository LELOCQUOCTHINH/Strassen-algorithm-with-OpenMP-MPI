cmake_minimum_required(VERSION 3.18)
project(MatrixComparison LANGUAGES CXX)

# Force Clang 18
set(CMAKE_C_COMPILER clang-18)
set(CMAKE_CXX_COMPILER clang++-18)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# OpenMP GPU Offload Flags
set(OMP_FLAGS
    -fopenmp
    -fopenmp-targets=nvptx64-nvidia-cuda
    -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_86
)
string(REPLACE ";" " " OMP_FLAGS_STR "${OMP_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_FLAGS_STR} -O3 -march=native")

# Add ALL source files
add_executable(comparisor
    comparisor.cpp
    naively_matrix_multiplication.cpp
    strassen_algorithm_sequential.cpp
    strassen_algorithm_openmp.cpp
    strassen_algorithm_openmp_cuda.cpp   # THIS WAS MISSING!
)

target_link_libraries(comparisor PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)