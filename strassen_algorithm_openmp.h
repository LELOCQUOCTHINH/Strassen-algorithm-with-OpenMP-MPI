// strassen_algorithm_openmp.h
#ifndef STRASSEN_ALGORITHM_OPENMP_H
#define STRASSEN_ALGORITHM_OPENMP_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

vector<vector<double>> strassen_parallel(const vector<vector<double>>& A,
                                         const vector<vector<double>>& B);

#endif