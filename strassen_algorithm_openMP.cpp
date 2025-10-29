// strassen.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// ---------------------------------------------------------------
// Helper functions (add / subtract / naive base case)
// ---------------------------------------------------------------
vector<vector<double>> matrixAdd(const vector<vector<double>>& A,
                                const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    if(n >= omp_get_num_threads())
    {
        #pragma openmp parallel if(n > 100)
        #pragma openmp for
        {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = A[i][j] + B[i][j];
        }
    }

    else
    {
        #pragma openmp parallel if(n > 100)
        #pragma openmp for collapse (2)
        {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

vector<vector<double>> matrixSub(const vector<vector<double>>& A,
                                const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    if(n >= omp_get_num_threads())
    {
        #pragma openmp parallel if(n > 100)
        #pragma openmp for
        {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = A[i][j] - B[i][j];
        }
    }

    else
    {
        #pragma openmp parallel if(n > 100)
        #pragma openmp for collapse (2)
        {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

vector<vector<double>> naiveMultiply(const vector<vector<double>>& A,
                                    const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

// ---------------------------------------------------------------
// Core recursive Strassen (explicit return type)
// ---------------------------------------------------------------
vector<vector<double>> strassenMultiply(const vector<vector<double>>& A,
                                       const vector<vector<double>>& B) {
    int n = A.size();

    // ---- base case (n <= 2) ----
    if (n <= 2) {
        return naiveMultiply(A, B);
    }

    // ---- split into quadrants ----
    int mid = n / 2;
    vector<vector<double>> A11(mid, vector<double>(mid));
    vector<vector<double>> A12(mid, vector<double>(mid));
    vector<vector<double>> A21(mid, vector<double>(mid));
    vector<vector<double>> A22(mid, vector<double>(mid));
    vector<vector<double>> B11(mid, vector<double>(mid));
    vector<vector<double>> B12(mid, vector<double>(mid));
    vector<vector<double>> B21(mid, vector<double>(mid));
    vector<vector<double>> B22(mid, vector<double>(mid));

    #pragma openmp parallel if (mid > 100)
    #pragma openmp for
    {
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
    }

    // ---- seven recursive products ----
    vector<vector<double>> P1, P2, P3, P4, P5, P6, P7;
    vector<vector<double>> C11, C12, C21, C22;

    vector<vector<double>> C(n, vector<double>(n, 0.0));
    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp task
        P1 = strassenMultiply(A11, matrixSub(B12, B22));
        #pragma omp task
        P2 = strassenMultiply(matrixAdd(A11, A12), B22);
        #pragma omp task
         P3 = strassenMultiply(matrixAdd(A21, A22), B11);
        #pragma omp task
        P4 = strassenMultiply(A22, matrixSub(B21, B11));
        #pragma omp task
        P5 = strassenMultiply(matrixAdd(A11, A22), matrixAdd(B11, B22));
        #pragma omp task
        P6 = strassenMultiply(matrixSub(A12, A22), matrixAdd(B21, B22));
        #pragma omp task
        P7 = strassenMultiply(matrixSub(A11, A21), matrixAdd(B11, B12));
    
        #pragma omp taskwait

        // ---- combine quadrants ----
        #pragma omp task
        C11 = matrixSub(matrixAdd(matrixAdd(P5, P4), P6), P2);
        #pragma omp task
        C12 = matrixAdd(P1, P2);
        #pragma omp task
        C21 = matrixAdd(P3, P4);
        #pragma omp task
        C22 = matrixSub(matrixAdd(P5, P1), matrixAdd(P3, P7));
    }

    // ---- assemble final matrix ----
    #pragma openmp parallel if (mid > 100)
    #pragma openmp for
        {
        for (int i = 0; i < mid; ++i) {
            for (int j = 0; j < mid; ++j) {
                C[i][j]            = C11[i][j];
                C[i][j + mid]      = C12[i][j];
                C[i + mid][j]      = C21[i][j];
                C[i + mid][j + mid]= C22[i][j];
            }
        }
    }
    return C;
}

// ---------------------------------------------------------------
// Padding / unpadding (for non-square matrices)
// ---------------------------------------------------------------
vector<vector<double>> padMatrix(const vector<vector<double>>& M) {
    int rows = M.size();
    int cols = M[0].size();
    int sz = 1;
    while (sz < max(rows, cols)) sz *= 2;

    vector<vector<double>> P(sz, vector<double>(sz, 0.0));

    if(rows >= omp_get_num_threads())
    {
        #pragma openmp parallel if(rows > 100 && cols > 100)
        #pragma openmp for
        {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    P[i][j] = M[i][j];
        }
    }

    else
    {
        #pragma openmp parallel if(rows > 100 && cols > 100)
        #pragma openmp for collapse (2)
        {
        for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    P[i][j] = M[i][j];
        }
    }

    return P;
}

vector<vector<double>> unpadMatrix(const vector<vector<double>>& P,
                                   int rows, int cols) {
    vector<vector<double>> M(rows, vector<double>(cols, 0.0));

    if(rows >= omp_get_num_threads())
    {
        #pragma openmp parallel if(rows > 100 && cols > 100)
        #pragma openmp for
        {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    M[i][j] = P[i][j];
        }
    }

    else
    {
        #pragma openmp parallel if(rows > 100 && cols > 100)
        #pragma openmp for collapse (2)
        {
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    M[i][j] = P[i][j];
        }
    }

    return M;
}

// ---------------------------------------------------------------
// Public wrapper (handles dimension check + pad/unpad)
// ---------------------------------------------------------------
vector<vector<double>> strassen(const vector<vector<double>>& A,
                                const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        cout << "ERROR: Incompatible dimensions for multiplication!\n";
        return vector<vector<double>>();
    }

    vector<vector<double>> Apad;
    vector<vector<double>> Bpad;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task
            Apad = padMatrix(A);

            #pragma omp task
            Bpad = padMatrix(B);
        }
    }

    vector<vector<double>> Cpad = strassenMultiply(Apad, Bpad);

    return unpadMatrix(Cpad, rowsA, colsB);
}

// ---------------------------------------------------------------
// Correctness checking (same as before)
// ---------------------------------------------------------------
bool compareMatrices(const vector<vector<double>>& S,
                     const MatrixXd& E, int rows, int cols,
                     double eps = 1e-6) {
    bool ok = true;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (abs(S[i][j] - E(i, j)) > eps) {
                ok = false;
                cout << "Mismatch [" << i << "][" << j << "]: "
                     << "Strassen=" << S[i][j]
                     << "  Eigen=" << E(i, j) << "\n";
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------
// main – two test cases + timing + validation
// ---------------------------------------------------------------
int main() {
    // ---------- Test 1: 3x5 * 5x6 ----------
    cout << "=== Test 1: 3x5 * 5x6 ===\n";
    vector<vector<double>> A = {
        {0, 8, 6, 9, 2},
        {9, 5, 4, 1, 6},
        {8, 3, 2, 6, 1}
    };
    vector<vector<double>> B = {
        {8, 8, 6, 4, 2, 7},
        {9, 5, 1, 6, 4, 9},
        {6, 6, 2, 2, 5, 4},
        {8, 8, 1, 7, 3, 2},
        {9, 0, 6, 8, 4, 2}
    };
    int ra = A.size(), ca = A[0].size();
    int rb = B.size(), cb = B[0].size();

    // Eigen reference
    MatrixXd EA(ra, ca);  for (int i=0;i<ra;++i) for (int j=0;j<ca;++j) EA(i,j)=A[i][j];
    MatrixXd EB(rb, cb);  for (int i=0;i<rb;++i) for (int j=0;j<cb;++j) EB(i,j)=B[i][j];
    MatrixXd Eref = EA * EB;

    // Strassen + timing
    double t0 = omp_get_wtime();
    vector<vector<double>> Sres = strassen(A, B);
    double t1 = omp_get_wtime();
    double us = (t1 - t0) * 1e6;

    cout << "Strassen time: " << fixed << setprecision(2) << us << " us\n";
    cout << "Correctness: " << (compareMatrices(Sres, Eref, ra, cb) ? "PASS" : "FAIL") << "\n";

    // ---------- Test 2: 100x100 * 100x100 (identity) ----------
    cout << "\n=== Test 2: 100x100 * 100x100 (identity) ===\n";
    const int N = 100;
    vector<vector<double>> LA(N, vector<double>(N));
    vector<vector<double>> LB(N, vector<double>(N, 0.0));
    for (int i=0;i<N;++i) {
        for (int j=0;j<N;++j) LA[i][j] = i*N + j + 1.0;
        LB[i][i] = 1.0;                     // identity
    }

    MatrixXd ELA(N,N); for (int i=0;i<N;++i) for (int j=0;j<N;++j) ELA(i,j)=LA[i][j];
    MatrixXd ELB = MatrixXd::Identity(N,N);
    MatrixXd ELref = ELA * ELB;            // should equal ELA

    t0 = omp_get_wtime();
    vector<vector<double>> LSres = strassen(LA, LB);
    t1 = omp_get_wtime();
    us = (t1 - t0) * 1e6;

    cout << "Strassen time: " << fixed << setprecision(2) << us << " us\n";
    cout << "Correctness: " << (compareMatrices(LSres, ELref, N, N) ? "PASS" : "FAIL") << "\n";

    return 0;
}