// strassen.cpp
#include "strassen_algorithm_openmp_cuda.h"

/* ==============================================================
   Helper: flatten / un-flatten (used only by the wrappers)
   ============================================================== */
 double* to_flat(const vector<vector<double>>& src, int rows, int cols)
{
    double* flat = new double[rows * cols];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = src[i][j];
    return flat;
}
 vector<vector<double>> from_flat(const double* flat, int rows, int cols)
{
    vector<vector<double>> dst(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[i][j] = flat[i * cols + j];
    return dst;
}

/* ==============================================================
   GPU kernels – **flat buffers only**
   ============================================================== */
static void matrixAdd_parallel_cuda(const double* A, const double* B,
                                    double* C, int n)
{
    #pragma omp target teams if(n > 100) \
        map(to:A[0:n*n],B[0:n*n]) map(from:C[0:n*n])
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i*n + j] = A[i*n + j] + B[i*n + j];
       
}

static void matrixSub_parallel_cuda(const double* A, const double* B,
                                    double* C, int n)
{
    #pragma omp target teams if(n > 100) \
        map(to:A[0:n*n],B[0:n*n]) map(from:C[0:n*n])
    #pragma omp distribute parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i*n + j] = A[i*n + j] - B[i*n + j];
}

static void naiveMultiply_parallel_cuda(const double* A, const double* B,
                                        double* C, int n)
{ //base case so we don't need to parallel it
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k)
                sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
}

/* ==============================================================
   Recursive Strassen – flat buffers
   ============================================================== */
static void strassenMultiply_parallel_cuda(const double* A, const double* B,
                                           double* C, int n)
{
    if (n <= 2) {
        naiveMultiply_parallel_cuda(A, B, C, n);
        return;
    }

    int mid = n / 2;
    int sub = mid * mid;

    double *A11 = new double[sub], *A12 = new double[sub],
           *A21 = new double[sub], *A22 = new double[sub];
    double *B11 = new double[sub], *B12 = new double[sub],
           *B21 = new double[sub], *B22 = new double[sub];

    // split quadrants (host)
    #pragma openmp parallel if (mid > 100)
    #pragma openmp for
    for (int i = 0; i < mid; ++i)
        for (int j = 0; j < mid; ++j) {
            A11[i*mid+j] = A[i*n+j];          A12[i*mid+j] = A[i*n+(j+mid)];
            A21[i*mid+j] = A[(i+mid)*n+j];    A22[i*mid+j] = A[(i+mid)*n+(j+mid)];
            B11[i*mid+j] = B[i*n+j];          B12[i*mid+j] = B[i*n+(j+mid)];
            B21[i*mid+j] = B[(i+mid)*n+j];    B22[i*mid+j] = B[(i+mid)*n+(j+mid)];
        }

    double *P1 = new double[sub], *P2 = new double[sub], *P3 = new double[sub],
           *P4 = new double[sub], *P5 = new double[sub], *P6 = new double[sub],
           *P7 = new double[sub];


    double *C11 = new double[sub], *C12 = new double[sub],
           *C21 = new double[sub], *C22 = new double[sub];

    double *tmp1 = new double[sub], *tmp2 = new double[sub],
           *tmp3 = new double[sub], *tmp4 = new double[sub],
           *tmp5a = new double[sub], *tmp5b = new double[sub],
           *tmp6a = new double[sub], *tmp6b = new double[sub],
           *tmp7a = new double[sub], *tmp7b = new double[sub];

    // --- 7 FULLY INDEPENDENT TASKS (no shared tmp!) ---
    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp task
        {
            matrixSub_parallel_cuda(B12, B22, tmp1, mid);
            strassenMultiply_parallel_cuda(A11, tmp1, P1, mid);
        }
        #pragma omp task
        {
            matrixAdd_parallel_cuda(A11, A12, tmp2, mid);
            strassenMultiply_parallel_cuda(tmp2, B22, P2, mid);
        }
        #pragma omp task
        {
            matrixAdd_parallel_cuda(A21, A22, tmp3, mid);
            strassenMultiply_parallel_cuda(tmp3, B11, P3, mid);
        }
        #pragma omp task
        {
            matrixSub_parallel_cuda(B21, B11, tmp4, mid);
            strassenMultiply_parallel_cuda(A22, tmp4, P4, mid);
        }
        #pragma omp task
        {
            matrixAdd_parallel_cuda(A11, A22, tmp5a, mid);
            matrixAdd_parallel_cuda(B11, B22, tmp5b, mid);
            strassenMultiply_parallel_cuda(tmp5a, tmp5b, P5, mid);
        }
        #pragma omp task
        {
            matrixSub_parallel_cuda(A12, A22, tmp6a, mid);
            matrixAdd_parallel_cuda(B21, B22, tmp6b, mid);
            strassenMultiply_parallel_cuda(tmp6a, tmp6b, P6, mid);
        }
        #pragma omp task
        {
            matrixSub_parallel_cuda(A11, A21, tmp7a, mid);
            matrixAdd_parallel_cuda(B11, B12, tmp7b, mid);
            strassenMultiply_parallel_cuda(tmp7a, tmp7b, P7, mid);
        }
    } // end parallel single — all 7 tasks now run concurrently!
    
    #pragma omp parallel sections
    {
        #pragma omp section
        { matrixAdd_parallel_cuda(P5, P4, tmp1, mid);
          matrixAdd_parallel_cuda(tmp1, P6, tmp2, mid);
          matrixSub_parallel_cuda(tmp2, P2, C11, mid); }
        #pragma omp section
        { matrixAdd_parallel_cuda(P1, P2, C12, mid); }
        #pragma omp section
        { matrixAdd_parallel_cuda(P3, P4, C21, mid); }
        #pragma omp section
        { matrixAdd_parallel_cuda(P5, P1, tmp3, mid);
          matrixAdd_parallel_cuda(P3, P7, tmp4, mid);
          matrixSub_parallel_cuda(tmp3, tmp4, C22, mid); }
    }

    #pragma openmp parallel if (mid > 100)
    #pragma openmp for
    for (int i = 0; i < mid; ++i)
        for (int j = 0; j < mid; ++j) {
            C[i*n + j]           = C11[i*mid+j];
            C[i*n + (j+mid)]     = C12[i*mid+j];
            C[(i+mid)*n + j]     = C21[i*mid+j];
            C[(i+mid)*n + (j+mid)] = C22[i*mid+j];
        }

    // cleanup
    delete[] A11; delete[] A12; delete[] A21; delete[] A22;
    delete[] B11; delete[] B12; delete[] B21; delete[] B22;
    delete[] P1; delete[] P2; delete[] P3; delete[] P4; delete[] P5; delete[] P6; delete[] P7;
    delete[] tmp1; delete[] tmp2; delete[] tmp3; delete[] tmp4;
    delete[] tmp5a; delete[] tmp5b; delete[] tmp6a; delete[] tmp6b;
    delete[] tmp7a; delete[] tmp7b;
    delete[] C11; delete[] C12; delete[] C21; delete[] C22;
}

/* ==============================================================
   padMatrix_parallel_cuda
   Input : M_flat – flat row-major vector<double>
           rows, cols – original size of M
           sz      – padded power-of-2 size (sz >= max(rows,cols))
   Output: padded flat vector<double> of size sz*sz
   ============================================================== */
vector<double> padMatrix_parallel_cuda(const vector<double>& M_flat,
                                       int rows, int cols, int sz)
{
    vector<double> P(sz * sz, 0.0);

    if (rows >= omp_get_num_threads()) {
        #pragma omp parallel if(rows > 100 || cols > 100)
        #pragma omp for
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                P[i*sz + j] = M_flat[i*cols + j];
    } else {
        #pragma omp parallel if(rows > 100 || cols > 100)
        #pragma omp for collapse(2)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                P[i*sz + j] = M_flat[i*cols + j];
    }

    return P;
}

/* ==============================================================
   unpadMatrix_parallel_cuda
   Input : P_flat – padded flat vector<double> (sz*sz)
           rows, cols – original (unpadded) dimensions
           sz      – padded size
   Output: flat vector<double> of size rows*cols
   ============================================================== */
vector<double> unpadMatrix_parallel_cuda(const vector<double>& P_flat,
                                         int rows, int cols, int sz)
{
    vector<double> M(rows * cols, 0.0);

    if (rows >= omp_get_num_threads()) {
        #pragma omp parallel if(rows > 100 || cols > 100)
        #pragma omp for
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                M[i*cols + j] = P_flat[i*sz + j];
    } else {
        #pragma omp parallel if(rows > 100 || cols > 100)
        #pragma omp for collapse(2)
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                M[i*cols + j] = P_flat[i*sz + j];
    }

    return M;
}

/* ==============================================================
   strassen_parallel_cuda – public wrapper
   Input : A_flat, B_flat – flat row-major vector<double>
           rowsA, colsA, rowsB, colsB – matrix dimensions
   Output: C_flat – flat result vector<double>
   ============================================================== */
vector<double> strassen_parallel_cuda(const vector<double>& A_flat,
                                      int rowsA, int colsA,
                                      const vector<double>& B_flat,
                                      int rowsB, int colsB)
{
    if (colsA != rowsB) {
        cerr << "ERROR: Incompatible dimensions for multiplication!\n";
        return {};
    }

    // ---- compute padded size (power of 2) ----
    int max_dim = max({rowsA, colsA, rowsB, colsB});
    int sz = 1;
    while (sz < max_dim) sz *= 2;

    // ---- pad A and B (flat) ----
    vector<double> Apad = padMatrix_parallel_cuda(A_flat, rowsA, colsA, sz);
    vector<double> Bpad = padMatrix_parallel_cuda(B_flat, rowsB, colsB, sz);

    // ---- allocate padded result ----
    vector<double> Cpad(sz * sz, 0.0);

    // ---- call recursive Strassen (flat version) ----
    strassenMultiply_parallel_cuda(Apad.data(), Bpad.data(), Cpad.data(), sz);

    // ---- unpad result ----
    return unpadMatrix_parallel_cuda(Cpad, rowsA, colsB, sz);
}