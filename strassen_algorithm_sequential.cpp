#include "strassen_algorithm_sequential.h"

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
vector<vector<double>> padMatrix(const vector<vector<double>>& matrix, int size) {
    int rows = matrix.size();
    int cols = matrix[0].size();

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
vector<vector<double>> strassen_sequential(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    if (colsA != rowsB) {
        cout << "Matrix dimensions do not match for multiplication!" << endl;
        return {};
    }

    int size = 1;
    int padded_size = max(rowsA, max(colsA, max(rowsB, colsB)));
    while (size < padded_size) {
        size *= 2;
    }

    auto paddedA = padMatrix(A, size);
    auto paddedB = padMatrix(B, size);
    auto paddedC = strassenMultiply(paddedA, paddedB);
    return unpadMatrix(paddedC, rowsA, colsB);
}
