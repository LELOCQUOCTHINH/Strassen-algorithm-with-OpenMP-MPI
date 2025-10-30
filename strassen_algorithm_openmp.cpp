// strassen.cpp
#include "strassen_algorithm_openmp.h"

// ---------------------------------------------------------------
// Helper functions (add / subtract / naive base case)
// ---------------------------------------------------------------
vector<vector<double>> matrixAdd_parallel(const vector<vector<double>>& A,
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

vector<vector<double>> matrixSub_parallel(const vector<vector<double>>& A,
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

vector<vector<double>> naiveMultiply_parallel(const vector<vector<double>>& A,
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
vector<vector<double>> strassenMultiply_parallel(const vector<vector<double>>& A,
                                       const vector<vector<double>>& B) {
    int n = A.size();

    // ---- base case (n <= 2) ----
    if (n <= 2) {
        return naiveMultiply_parallel(A, B);
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
        P1 = strassenMultiply_parallel(A11, matrixSub_parallel(B12, B22));
        #pragma omp task
        P2 = strassenMultiply_parallel(matrixAdd_parallel(A11, A12), B22);
        #pragma omp task
         P3 = strassenMultiply_parallel(matrixAdd_parallel(A21, A22), B11);
        #pragma omp task
        P4 = strassenMultiply_parallel(A22, matrixSub_parallel(B21, B11));
        #pragma omp task
        P5 = strassenMultiply_parallel(matrixAdd_parallel(A11, A22), matrixAdd_parallel(B11, B22));
        #pragma omp task
        P6 = strassenMultiply_parallel(matrixSub_parallel(A12, A22), matrixAdd_parallel(B21, B22));
        #pragma omp task
        P7 = strassenMultiply_parallel(matrixSub_parallel(A11, A21), matrixAdd_parallel(B11, B12));
    
        #pragma omp taskwait

        // ---- combine quadrants ----
        #pragma omp task
        C11 = matrixSub_parallel(matrixAdd_parallel(matrixAdd_parallel(P5, P4), P6), P2);
        #pragma omp task
        C12 = matrixAdd_parallel(P1, P2);
        #pragma omp task
        C21 = matrixAdd_parallel(P3, P4);
        #pragma omp task
        C22 = matrixSub_parallel(matrixAdd_parallel(P5, P1), matrixAdd_parallel(P3, P7));
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
vector<vector<double>> padMatrix_parallel(const vector<vector<double>>& M, int sz) {
    int rows = M.size();
    int cols = M[0].size();

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

vector<vector<double>> unpadMatrix_parallel(const vector<vector<double>>& P,
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
vector<vector<double>> strassen_parallel(const vector<vector<double>>& A,
                                const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        cout << "ERROR: Incompatible dimensions for multiplication!\n";
        return vector<vector<double>>();
    }

    int size = 1;
    int padded_size = max(rowsA, max(colsA, max(rowsB, colsB)));
    while (size < padded_size) {
        size *= 2;
    }

    vector<vector<double>> Apad;
    vector<vector<double>> Bpad;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task
            Apad = padMatrix_parallel(A, size);

            #pragma omp task
            Bpad = padMatrix_parallel(B, size);
        }
    }

    vector<vector<double>> Cpad = strassenMultiply_parallel(Apad, Bpad);

    return unpadMatrix_parallel(Cpad, rowsA, colsB);
}


