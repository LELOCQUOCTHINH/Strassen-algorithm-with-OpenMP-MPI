// strassen_algorithm_sequential.h
#ifndef STRASSEN_ALGORITHM_SEQUENTIAL_H
#define STRASSEN_ALGORITHM_SEQUENTIAL_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

vector<vector<double>> strassen_sequential(const vector<vector<double>>& A,
                                           const vector<vector<double>>& B);

#endif