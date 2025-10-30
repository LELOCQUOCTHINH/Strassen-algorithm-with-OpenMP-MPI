// comparisor.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <omp.h>
#include <eigen3/Eigen/Dense>

#include "naively_matrix_multiplication.h"
#include "strassen_algorithm_sequential.h"
#include "strassen_algorithm_openmp.h"

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
size_t bytes(double** p, int r, int c) { return r * c * sizeof(double); }

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
    vector<vector<double>> A(rowsA, vector<double>(colsA));
    vector<vector<double>> B(colsA, vector<double>(colsB));
    fill_random(A, 1);
    fill_random(B, 2);

    // ---- Eigen reference ------------------------------------------------
    MatrixXd EA(rowsA, colsA), EB(colsA, colsB), Eref(rowsA, colsB);
    for (int i = 0; i < rowsA; ++i) for (int j = 0; j < colsA; ++j) EA(i,j) = A[i][j];
    for (int i = 0; i < colsA; ++i) for (int j = 0; j < colsB; ++j) EB(i,j) = B[i][j];
    Eref = EA * EB;

    // ---- memory ---------------------------------------------------------
    size_t memA = bytes(A), memB = bytes(B), memC = rowsA * colsB * sizeof(double);
    cout << "Memory: A=" << memA/1e6 << " MB, B=" << memB/1e6
         << " MB, C=" << memC/1e6 << " MB  (total " << (memA+memB+memC)/1e6 << " MB)\n";

    double t_seq_naive = 0, t_par_naive = 0;
    double t_seq_strass = 0, t_par_strass = 0;

    // -----------------------------------------------------------------
    // 1. Sequential naïve
    // -----------------------------------------------------------------
    {
        double** rawA = to_raw(A);
        double** rawB = to_raw(B);
        double t0 = omp_get_wtime();
        double** C = naively_matrix_multiplication_sequential(rawA, rowsA, colsA,
                                                             rawB, colsA, colsB);
        double t1 = omp_get_wtime();
        t_seq_naive = (t1-t0)*1e6;

        MatrixXd check(rowsA, colsB);
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsB; ++j) check(i,j) = C[i][j];
        bool ok = (check - Eref).norm() < 1e-6 * rowsA * colsB;
        cout << "Seq Naïve  : " << fixed << setprecision(2) << setw(9) << t_seq_naive << " µs  [" << (ok?"PASS":"FAIL") << "]\n";

        free_raw(rawA, rowsA); free_raw(rawB, colsA); free_raw(C, rowsA);
    }

    // -----------------------------------------------------------------
    // 2. Parallel naïve
    // -----------------------------------------------------------------
    {
        double** rawA = to_raw(A);
        double** rawB = to_raw(B);
        double t0 = omp_get_wtime();
        double** C = naively_matrix_multiplication_parallel(rawA, rowsA, colsA,
                                                           rawB, colsA, colsB);
        double t1 = omp_get_wtime();
        t_par_naive = (t1-t0)*1e6;

        MatrixXd check(rowsA, colsB);
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsB; ++j) check(i,j) = C[i][j];
        bool ok = (check - Eref).norm() < 1e-6 * rowsA * colsB;
        cout << "Par Naïve  : " << fixed << setprecision(2) << setw(9) << t_par_naive << " µs  [" << (ok?"PASS":"FAIL") << "]\n";

        free_raw(rawA, rowsA); free_raw(rawB, colsA); free_raw(C, rowsA);
    }

    // -----------------------------------------------------------------
    // 3. Sequential Strassen
    // -----------------------------------------------------------------
    {
        double t0 = omp_get_wtime();
        auto C = strassen_sequential(A, B);
        double t1 = omp_get_wtime();
        t_seq_strass = (t1-t0)*1e6;

        MatrixXd check(rowsA, colsB);
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsB; ++j) check(i,j) = C[i][j];
        bool ok = (check - Eref).norm() < 1e-6 * rowsA * colsB;
        cout << "Seq Strassen: " << fixed << setprecision(2) << setw(9) << t_seq_strass << " µs  [" << (ok?"PASS":"FAIL") << "]\n";
    }

    // -----------------------------------------------------------------
    // 4. Parallel Strassen (OpenMP)
    // -----------------------------------------------------------------
    {
        double t0 = omp_get_wtime();
        auto C = strassen_parallel(A, B);
        double t1 = omp_get_wtime();
        t_par_strass = (t1-t0)*1e6;

        MatrixXd check(rowsA, colsB);
        for (int i = 0; i < rowsA; ++i)
            for (int j = 0; j < colsB; ++j) check(i,j) = C[i][j];
        bool ok = (check - Eref).norm() < 1e-6 * rowsA * colsB;
        cout << "Par Strassen: " << fixed << setprecision(2) << setw(9) << t_par_strass << " µs  [" << (ok?"PASS":"FAIL") << "]\n";
    }

    // -----------------------------------------------------------------
    // Speed-up
    // -----------------------------------------------------------------
    if (t_seq_naive > 0)
        cout << "  Speed-up (Par/Seq Naïve)   : " << fixed << setprecision(2) << t_seq_naive/t_par_naive << "x\n";
    if (t_seq_strass > 0)
        cout << "  Speed-up (Par/Seq Strassen): " << fixed << setprecision(2) << t_seq_strass/t_par_strass << "x\n";
}

// ------------------------------------------------------------------
// 5. main – list of test cases
// ------------------------------------------------------------------
int main() {
    cout << fixed << setprecision(2);
    cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";

    // -----------------------------------------------------------------
    // Small sanity checks (square & rectangular)
    // -----------------------------------------------------------------
    test_one(3, 5, 6);      // 3×5  * 5×6  → 3×6
    test_one(100, 50, 80);  // 100×50 * 50×80 → 100×80
    test_one(64, 64, 64);   // classic square

    // -----------------------------------------------------------------
    // Large square matrices
    // -----------------------------------------------------------------
    test_one(1000, 1000, 1000);
    // test_one(100000, 100000, 100000);   // uncomment when you have ~80 GB RAM
    // test_one(1000000, 1000000, 1000000); // ~8 TB – only for clusters

    // -----------------------------------------------------------------
    // Large rectangular (rows >> cols  and  cols >> rows)
    // -----------------------------------------------------------------
    test_one(2000, 500, 800);   // tall
    test_one(500, 2000, 800);   // wide

    return 0;
}