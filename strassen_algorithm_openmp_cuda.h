// strassen_algorithm_openmp_cuda.h
#ifndef STRASSEN_ALGORITHM_OPENMP_CUDA_H
#define STRASSEN_ALGORITHM_OPENMP_CUDA_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// ==============================================================
// Public API – GPU-accelerated Strassen
// ==============================================================
vector<double> strassen_parallel_cuda(
    const vector<double>& A_flat, int rowsA, int colsA,
    const vector<double>& B_flat, int rowsB, int colsB
);

// ==============================================================
// Conversion helpers: vector<vector<double>> ↔ double*
// ==============================================================
double* to_flat(const vector<vector<double>>& src, int rows, int cols);
vector<vector<double>> from_flat(const double* flat, int rows, int cols);
#endif