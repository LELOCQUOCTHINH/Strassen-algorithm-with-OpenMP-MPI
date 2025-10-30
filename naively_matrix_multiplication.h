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

double** naively_matrix_multiplication_parallel(double** matrixA, int rowsA, int columnsA,
                                                double** matrixB, int rowsB, int columnsB);

void printMatrix(double** matrix, int rows, int columns);

bool compareMatrices(double** matrix1, const Eigen::MatrixXd& matrix2, int rows, int columns, double epsilon = 1e-6);

#endif