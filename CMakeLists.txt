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