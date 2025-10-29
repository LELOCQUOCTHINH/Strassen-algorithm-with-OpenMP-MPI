#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

// Function to print a matrix
void printMatrix(const vector<vector<double>>& matrix) {
    if (matrix.empty()) {
        cout << "Matrix is empty!" << endl;
        return;
    }
    cout << "Matrix (" << matrix.size() << "x" << matrix[0].size() << "):" << endl;
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << setw(8) << fixed << setprecision(2) << val << " ";
        }
        cout << endl;
    }
}

// Function to compare Strassen result with Eigen
bool compareMatrices(const vector<vector<double>>& matrix1, const MatrixXd& matrix2, int rows, int cols, double epsilon = 1e-6) {
    bool isEqual = true;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (abs(matrix1[i][j] - matrix2(i, j)) > epsilon) {
                isEqual = false;
                cout << "Mismatch at [" << i << "][" << j << "]: "
                     << "Strassen = " << fixed << setprecision(2) << matrix1[i][j]
                     << ", Eigen = " << matrix2(i, j) << endl;
            }
        }
    }
    return isEqual;
}

// Function to add two matrices
vector<vector<double>> matrixAdd(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

// Function to subtract two matrices
vector<vector<double>> matrixSub(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

// Naive matrix multiplication for base case
vector<vector<double>> naiveMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Strassen's recursive multiplication
vector<vector<double>> strassenMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    if (n <= 2) { // Base case threshold
        return naiveMultiply(A, B);
    }

    int mid = n / 2;
    vector<vector<double>> A11(mid, vector<double>(mid));
    vector<vector<double>> A12(mid, vector<double>(mid));
    vector<vector<double>> A21(mid, vector<double>(mid));
    vector<vector<double>> A22(mid, vector<double>(mid));
    vector<vector<double>> B11(mid, vector<double>(mid));
    vector<vector<double>> B12(mid, vector<double>(mid));
    vector<vector<double>> B21(mid, vector<double>(mid));
    vector<vector<double>> B22(mid, vector<double>(mid));

    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + mid];
            A21[i][j] = A[i + mid][j];
            A22[i][j] = A[i + mid][j + mid];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + mid];
            B21[i][j] = B[i + mid][j];
            B22[i][j] = B[i + mid][j + mid];
        }
    }

    auto P1 = strassenMultiply(A11, matrixSub(B12, B22));
    auto P2 = strassenMultiply(matrixAdd(A11, A12), B22);
    auto P3 = strassenMultiply(matrixAdd(A21, A22), B11);
    auto P4 = strassenMultiply(A22, matrixSub(B21, B11));
    auto P5 = strassenMultiply(matrixAdd(A11, A22), matrixAdd(B11, B22));
    auto P6 = strassenMultiply(matrixSub(A12, A22), matrixAdd(B21, B22));
    auto P7 = strassenMultiply(matrixSub(A11, A21), matrixAdd(B11, B12));

    auto C11 = matrixSub(matrixAdd(matrixAdd(P5, P4), P6), P2);
    auto C12 = matrixAdd(P1, P2);
    auto C21 = matrixAdd(P3, P4);
    auto C22 = matrixSub(matrixAdd(P5, P1), matrixAdd(P3, P7));

    vector<vector<double>> C(n, vector<double>(n));
    for (int i = 0; i < mid; ++i) {
        for (int j = 0; j < mid; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + mid] = C12[i][j];
            C[i + mid][j] = C21[i][j];
            C[i + mid][j + mid] = C22[i][j];
        }
    }
    return C;
}

// Pad matrix to next power of 2
vector<vector<double>> padMatrix(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int size = 1;
    while (size < max(rows, cols)) {
        size *= 2;
    }
    vector<vector<double>> padded(size, vector<double>(size, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            padded[i][j] = matrix[i][j];
        }
    }
    return padded;
}

// Unpad matrix to original size
vector<vector<double>> unpadMatrix(const vector<vector<double>>& padded, int rows, int cols) {
    vector<vector<double>> original(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            original[i][j] = padded[i][j];
        }
    }
    return original;
}

// Wrapper for Strassen multiplication
vector<vector<double>> strassen(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    if (colsA != rowsB) {
        cout << "Matrix dimensions do not match for multiplication!" << endl;
        return {};
    }

    auto paddedA = padMatrix(A);
    auto paddedB = padMatrix(B);
    auto paddedC = strassenMultiply(paddedA, paddedB);
    return unpadMatrix(paddedC, rowsA, colsB);
}

int main() {
    // Test 1: Small matrices (3x5 * 5x6)
    cout << "=== Test 1: Small Matrices (3x5 * 5x6) ===\n";
    vector<vector<double>> matrixA = {
        {0, 8, 6, 9, 2},
        {9, 5, 4, 1, 6},
        {8, 3, 2, 6, 1}
    };
    vector<vector<double>> matrixB = {
        {8, 8, 6, 4, 2, 7},
        {9, 5, 1, 6, 4, 9},
        {6, 6, 2, 2, 5, 4},
        {8, 8, 1, 7, 3, 2},
        {9, 0, 6, 8, 4, 2}
    };
    int rowsA = matrixA.size(), colsA = matrixA[0].size();
    int rowsB = matrixB.size(), colsB = matrixB[0].size();

    // Eigen reference result
    MatrixXd eigenA(rowsA, colsA);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            eigenA(i, j) = matrixA[i][j];
        }
    }
    MatrixXd eigenB(rowsB, colsB);
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            eigenB(i, j) = matrixB[i][j];
        }
    }
    MatrixXd eigenResult = eigenA * eigenB;

    // Time and test Strassen
    double start = omp_get_wtime();
    auto strassenResult = strassen(matrixA, matrixB);
    double end = omp_get_wtime();
    double duration = (end - start) * 1e6; // Convert to microseconds
    cout << "Strassen Time: " << fixed << setprecision(2) << duration << " microseconds\n";
    cout << "Strassen Correctness: " << (compareMatrices(strassenResult, eigenResult, rowsA, colsB) ? "Correct" : "Incorrect") << "\n";

    // Print results
    cout << "\nStrassen Result:\n";
    printMatrix(strassenResult);
    cout << "\nEigen Result:\n";
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            cout << setw(8) << fixed << setprecision(2) << eigenResult(i, j) << " ";
        }
        cout << endl;
    }

    // Test 2: Large matrices (100x100 * 100x100)
    cout << "\n=== Test 2: Large Matrices (100x100 * 100x100) ===\n";
    int largeRowsA = 100, largeColsA = 100, largeRowsB = 100, largeColsB = 100;
    vector<vector<double>> largeA(largeRowsA, vector<double>(largeColsA));
    vector<vector<double>> largeB(largeRowsB, vector<double>(largeColsB));
    for (int i = 0; i < largeRowsA; ++i) {
        for (int j = 0; j < largeColsA; ++j) {
            largeA[i][j] = i * largeColsA + j + 1.0; // Unique values
        }
    }
    for (int i = 0; i < largeRowsB; ++i) {
        for (int j = 0; j < largeColsB; ++j) {
            largeB[i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix
        }
    }

    // Eigen reference result
    MatrixXd eigenLargeA(largeRowsA, largeColsA);
    for (int i = 0; i < largeRowsA; ++i) {
        for (int j = 0; j < largeColsA; ++j) {
            eigenLargeA(i, j) = largeA[i][j];
        }
    }
    MatrixXd eigenLargeB = MatrixXd::Identity(largeRowsB, largeColsB);
    MatrixXd eigenLargeResult = eigenLargeA * eigenLargeB;

    // Time and test Strassen
    start = omp_get_wtime();
    auto largeStrassenResult = strassen(largeA, largeB);
    end = omp_get_wtime();
    duration = (end - start) * 1e6;
    cout << "Strassen Time: " << fixed << setprecision(2) << duration << " microseconds\n";
    cout << "Strassen Correctness: " << (compareMatrices(largeStrassenResult, eigenLargeResult, largeRowsA, largeColsB) ? "Correct" : "Incorrect") << "\n";

    return 0;
}