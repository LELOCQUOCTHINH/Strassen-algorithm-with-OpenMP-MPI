// naively_matrix_multiplication.h
#ifndef NAIVELY_MATRIX_MULTIPLICATION_H
#define NAIVELY_MATRIX_MULTIPLICATION_H

#include <iostream>
#include <omp.h>
#include <iomanip>

#include <eigen3/Eigen/Dense> // Instead of <Eigen/Dense>
using namespace std;
using namespace Eigen;

double** naively_matrix_multiplication_sequential(double** matrixA, int rowsA, int columnsA,
                                                  double** matrixB, int rowsB, int columnsB);

double** naively_matrix_multiplication_openmp(double** matrixA, int rowsA, int columnsA,
                                                double** matrixB, int rowsB, int columnsB);

double* naively_matrix_multiplication_openmp_cuda(
    const double* A_flat, int rowsA, int colsA,
    const double* B_flat, int rowsB, int colsB);

void printMatrix(double** matrix, int rows, int columns);

double* to_flat_double_star(double** matrix, int rows, int cols);

double** from_flat_double_star(const double* flat, int rows, int cols);

bool compareMatrices(double** matrix1, const Eigen::MatrixXd& matrix2, int rows, int columns, double epsilon = 1e-6);

#endif