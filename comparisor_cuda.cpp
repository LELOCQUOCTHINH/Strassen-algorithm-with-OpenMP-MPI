// comparisor_cuda.cpp - GPU versions: Naive Seq, Strassen Seq, Naive GPU, Strassen GPU
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <omp.h>
#include <eigen3/Eigen/Dense>

#include "naively_matrix_multiplication.h"
#include "strassen_algorithm_sequential.h"
#include "strassen_algorithm_openmp_cuda.h"  // For GPU Strassen and helpers

using namespace std;
using namespace Eigen;

// ------------------------------------------------------------------
// 1. Helpers – conversion raw <-> vector, memory, fill
// ------------------------------------------------------------------
double** to_raw(const vector<vector<double>>& v) {
    int r = v.size(), c = v[0].size();
    double** p = new double*[r];
    for (int i = 0; i < r; ++i) {
        p[i] = new double[c];
        for (int j = 0; j < c; ++j) p[i][j] = v[i][j];
    }
    return p;
}
void free_raw(double** p, int r) { for (int i = 0; i < r; ++i) delete[] p[i]; delete[] p; }

size_t bytes(const vector<vector<double>>& m) {
    return m.size() * m[0].size() * sizeof(double);
}

void fill_random(vector<vector<double>>& m, unsigned seed = 1) {
    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(-10.0, 10.0);
    for (auto& row : m) for (auto& v : row) v = dist(rng);
}

// ------------------------------------------------------------------
// 2. Test driver – one size (rowsA × colsA) * (colsA × colsB)
// ------------------------------------------------------------------
void test_one(int rowsA, int colsA, int colsB) {
    cout << "\n=== " << rowsA << "x" << colsA << "  *  " << colsA << "x" << colsB << " ===\n";

    // ---- create matrices ------------------------------------------------
    vector<vector<double>> A_vec(rowsA, vector<double>(colsA));
    vector<vector<double>> B_vec(colsA, vector<double>(colsB));
    fill_random(A_vec, 1);
    fill_random(B_vec, 2);

    // ---- Eigen reference ------------------------------------------------
    MatrixXd A_eig(rowsA, colsA), B_eig(colsA, colsB);
    for (int i = 0; i < rowsA; ++i) for (int j = 0; j < colsA; ++j) A_eig(i,j) = A_vec[i][j];
    for (int i = 0; i < colsA; ++i) for (int j = 0; j < colsB; ++j) B_eig(i,j) = B_vec[i][j];
    MatrixXd C_eig = A_eig * B_eig;

    // ---- Convert to raw double** for legacy functions -------------------
    double** A_raw = to_raw(A_vec);
    double** B_raw = to_raw(B_vec);

    // -----------------------------------------------------------------
    // 1. Sequential Naïve
    // -----------------------------------------------------------------
    double t_seq_naive = 0.0;
    double** C_seq_naive = nullptr;
    {
        double t0 = omp_get_wtime();
        C_seq_naive = naively_matrix_multiplication_sequential(A_raw, rowsA, colsA, B_raw, colsA, colsB);
        double t1 = omp_get_wtime();
        t_seq_naive = (t1 - t0) * 1e6;
    }
    bool ok_seq_naive = compareMatrices(C_seq_naive, C_eig, rowsA, colsB);
    cout << "Seq Naïve:     " << fixed << setprecision(2) << setw(9) << t_seq_naive << " µs  [" << (ok_seq_naive?"PASS":"FAIL") << "]\n";

    // -----------------------------------------------------------------
    // 2. Sequential Strassen
    // -----------------------------------------------------------------
    double t_seq_strass = 0.0;
    vector<vector<double>> C_seq_strass;
    {
        double t0 = omp_get_wtime();
        C_seq_strass = strassen_sequential(A_vec, B_vec);
        double t1 = omp_get_wtime();
        t_seq_strass = (t1 - t0) * 1e6;
    }
    double** C_seq_strass_raw = to_raw(C_seq_strass);
    bool ok_seq_strass = compareMatrices(C_seq_strass_raw, C_eig, rowsA, colsB);
    free_raw(C_seq_strass_raw, rowsA);
    cout << "Seq Strassen:  " << fixed << setprecision(2) << setw(9) << t_seq_strass << " µs  [" << (ok_seq_strass?"PASS":"FAIL") << "]\n";

    // -----------------------------------------------------------------
    // 3. GPU Naïve (OpenMP Offload)
    // -----------------------------------------------------------------
    double* A_flat = to_flat(A_vec, rowsA, colsA);
    double* B_flat = to_flat(B_vec, colsA, colsB);

    double t_gpu_naive = 0.0;
    double* C_gpu_naive_flat = nullptr;
    {
        double t0 = omp_get_wtime();
        C_gpu_naive_flat = naively_matrix_multiplication_openmp_cuda(
            A_flat, rowsA, colsA, B_flat, colsA, colsB
        );
        double t1 = omp_get_wtime();
        t_gpu_naive = (t1 - t0) * 1e6;
    }
    double** C_gpu_naive = from_flat_double_star(C_gpu_naive_flat, rowsA, colsB);
    bool ok_gpu_naive = compareMatrices(C_gpu_naive, C_eig, rowsA, colsB);
    cout << "GPU Naïve:     " << fixed << setprecision(2) << setw(9) << t_gpu_naive << " µs  [" << (ok_gpu_naive?"PASS":"FAIL") << "]\n";

    // -----------------------------------------------------------------
    // 4. GPU Strassen (OpenMP Offload)
    // -----------------------------------------------------------------
    double t_gpu_strass = 0.0;
    vector<double> C_gpu_strass_flat;
    {
        vector<double> A_flat_vec(A_flat, A_flat + rowsA*colsA);
        vector<double> B_flat_vec(B_flat, B_flat + colsA*colsB);

        double t0 = omp_get_wtime();
        C_gpu_strass_flat = strassen_parallel_cuda(
            A_flat_vec, rowsA, colsA,
            B_flat_vec, colsA, colsB
        );
        double t1 = omp_get_wtime();
        t_gpu_strass = (t1 - t0) * 1e6;
    }
    vector<vector<double>> C_gpu_strass_vec = from_flat(C_gpu_strass_flat.data(), rowsA, colsB);
    double** C_gpu_strass_raw = to_raw(C_gpu_strass_vec);
    bool ok_gpu_strass = compareMatrices(C_gpu_strass_raw, C_eig, rowsA, colsB);
    free_raw(C_gpu_strass_raw, rowsA);
    cout << "GPU Strassen:  " << fixed << setprecision(2) << setw(9) << t_gpu_strass << " µs  [" << (ok_gpu_strass?"PASS":"FAIL") << "]\n";

    // -----------------------------------------------------------------
    // Speed-ups (sequential / other)
    // -----------------------------------------------------------------
    cout << "\n--- Speed-ups (baseline / optimized) ---\n";
    if (t_seq_naive > 0 && t_seq_strass > 0) {
        cout << "  Strassen Seq / Naive Seq       : " << fixed << setprecision(2) 
            << t_seq_naive / t_seq_strass << "x\n";
    }
    if (t_seq_naive > 0 && t_gpu_naive > 0) {
        cout << "  Naive GPU / Naive Seq          : " << fixed << setprecision(2) 
            << t_seq_naive / t_gpu_naive << "x\n";
    }
    if (t_seq_strass > 0 && t_gpu_strass > 0) {
        cout << "  Strassen GPU / Strassen Seq    : " << fixed << setprecision(2) 
            << t_seq_strass / t_gpu_strass << "x\n";
    }
    if (t_gpu_naive > 0 && t_gpu_strass > 0) {
        cout << "  Strassen GPU / Naive GPU       : " << fixed << setprecision(2) 
            << t_gpu_naive / t_gpu_strass << "x\n";
    }

    // -----------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------
    free_raw(A_raw, rowsA);
    free_raw(B_raw, colsA);
    free_raw(C_seq_naive, rowsA);
    free_raw(C_gpu_naive, rowsA);
    delete[] A_flat;
    delete[] B_flat;
    delete[] C_gpu_naive_flat;
}

// ------------------------------------------------------------------
// main – test cases
// ------------------------------------------------------------------
int main() {
    cout << fixed << setprecision(2);
    cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";

    test_one(3, 5, 6);

    test_one(64, 64, 64);

    test_one(100, 50, 80);
    
    test_one(1000, 1000, 1000);
    test_one(2000, 500, 800);
    test_one(500, 2000, 800);

    return 0;
}