#include <iostream>
#include <omp.h>
#include <iomanip>

#include <eigen3/Eigen/Dense> // Instead of <Eigen/Dense>
using namespace std;
using namespace Eigen;

double ** naively_matrix_multiplication_sequential(double ** matrixA, int rowsA, int columnsA, double ** matrixB, int rowsB, int columnsB)
{
    if(rowsA < 0 || columnsA < 0 || rowsB < 0 || columnsB < 0)
    {
        cout << "The matrix size isn't suitable! \n";
        return NULL;
    }

    if (columnsA != rowsB)
    {
        cout << "The matrix can not multiplicate! \n";
        return NULL;
    }

    double ** matrix_out = new double * [rowsA];

    for (int i = 0 ; i < rowsA ; ++i)
    {
        matrix_out[i] = new double [columnsB];
    }

    for(int i = 0 ; i < rowsA ; ++i)
    {
        for(int j = 0 ; j < columnsB ; ++j)
        {
            double sum  = 0;
            for(int z = 0 ; z < columnsA ; ++z) //columns A == rows B so we consider either of them
            {
                sum += matrixA[i][z] * matrixB[z][j];
            }
            matrix_out[i][j] = sum;
        }
    }

    return matrix_out;
}

double ** naively_matrix_multiplication_parallel(double ** matrixA, int rowsA, int columnsA, double ** matrixB, int rowsB, int columnsB)
{
    if(rowsA < 0 || columnsA < 0 || rowsB < 0 || columnsB < 0)
    {
        cout << "The matrix size isn't suitable! \n";
        return NULL;
    }

    if (columnsA != rowsB)
    {
        cout << "The matrix can not multiplicate! \n";
        return NULL;
    }

    double ** matrix_out = new double * [rowsA];

    for (int i = 0 ; i < rowsA ; ++i)
    {
        matrix_out[i] = new double [columnsB];
    }

    if(rowsA >= omp_get_max_threads())
    {
        #pragma openmp parallel default(none) shared(rowsA, columnsB, columnsA, matrix_out, matrixA, matrixB) if(columnsB > 100)
        {
            #pragma omp for
            for(int i = 0 ; i < rowsA ; ++i)
            {
                for(int j = 0 ; j < columnsB ; ++j)
                {
                    double sum  = 0;
                    for(int z = 0 ; z < columnsA ; ++z) //columns A == rows B so we consider either of them
                    {
                        sum += matrixA[i][z] * matrixB[z][j];
                    }
                    matrix_out[i][j] = sum;
                }
            }
        }
    }
    
    else
    {
        #pragma openmp parallel default(none) shared(rowsA, columnsB, columnsA, matrix_out, matrixA, matrixB) if(columnsB > 100)
        {
            double sum = 0; //omp for collapse need a perfect loop so i adjust the declaration position of the sum
            #pragma omp for collapse(3)
            for(int i = 0 ; i < rowsA ; ++i)
            {
                for(int j = 0 ; j < columnsB ; ++j)
                {
                    for(int z = 0 ; z < columnsA ; ++z) //columns A == rows B so we consider either of them
                    {
                        sum += matrixA[i][z] * matrixB[z][j];
                        if(z + 1 == columnsA) //for perfect loop
                        {
                            matrix_out[i][j] = sum;
                            sum = 0;
                        }
                    }
                }
            }
        }
    }

    return matrix_out;
}

// Function to print a matrix
void printMatrix(double **matrix, int rows, int columns) {
    if (matrix == nullptr) {
        cout << "Matrix is null!" << endl;
        return;
    }
    cout << "Matrix (" << rows << "x" << columns << "):" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            cout << setw(8) << fixed << setprecision(2) << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// Function to compare two matrices
bool compareMatrices(double **matrix1, const MatrixXd &matrix2, int rows, int columns, double epsilon = 1e-6) {
    if (matrix1 == nullptr) {
        cout << "Comparison failed: Input matrix is null!" << endl;
        return false;
    }
    bool isEqual = true;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            if (abs(matrix1[i][j] - matrix2(i, j)) > epsilon) {
                isEqual = false;
                cout << "Mismatch at [" << i << "][" << j << "]: "
                     << "Your result = " << fixed << setprecision(2) << matrix1[i][j]
                     << ", Eigen = " << matrix2(i, j) << endl;
            }
        }
    }
    return isEqual;
}

int main() {
    // Test 1: Small matrices (3x5 * 5x6)
    cout << "=== Test 1: Small Matrices (3x5 * 5x6) ===\n";
    int rowsA = 3, columnsA = 5;
    int rowsB = 5, columnsB = 6;
    double **matrixA = new double *[rowsA];
    for (int i = 0; i < rowsA; ++i) {
        matrixA[i] = new double[columnsA];
    }
    double tempA[3][5] = {{0, 8, 6, 9, 2}, {9, 5, 4, 1, 6}, {8, 3, 2, 6, 1}};
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < columnsA; ++j) {
            matrixA[i][j] = tempA[i][j];
        }
    }
    double **matrixB = new double *[rowsB];
    for (int i = 0; i < rowsB; ++i) {
        matrixB[i] = new double[columnsB];
    }
    double tempB[5][6] = {
        {8, 8, 6, 4, 2, 7},
        {9, 5, 1, 6, 4, 9},
        {6, 6, 2, 2, 5, 4},
        {8, 8, 1, 7, 3, 2},
        {9, 0, 6, 8, 4, 2}
    };
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < columnsB; ++j) {
            matrixB[i][j] = tempB[i][j];
        }
    }

    // Compute Eigen result for reference
    MatrixXd eigenA(rowsA, columnsA);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < columnsA; ++j) {
            eigenA(i, j) = matrixA[i][j];
        }
    }
    MatrixXd eigenB(rowsB, columnsB);
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < columnsB; ++j) {
            eigenB(i, j) = matrixB[i][j];
        }
    }
    MatrixXd eigenResult = eigenA * eigenB;

    // Time and test sequential function
    double startSeq = omp_get_wtime();
    double **seqResult = naively_matrix_multiplication_sequential(matrixA, rowsA, columnsA, matrixB, rowsB, columnsB);
    double endSeq = omp_get_wtime();
    double seqDuration = (endSeq - startSeq) * 1e6; // Convert to microseconds
    cout << "Sequential Time: " << fixed << setprecision(2) << seqDuration << " microseconds\n";
    cout << "Sequential Correctness: " << (compareMatrices(seqResult, eigenResult, rowsA, columnsB) ? "Correct" : "Incorrect") << "\n";

    // Time and test parallel function
    double startPar = omp_get_wtime();
    double **parResult = naively_matrix_multiplication_parallel(matrixA, rowsA, columnsA, matrixB, rowsB, columnsB);
    double endPar = omp_get_wtime();
    double parDuration = (endPar - startPar) * 1e6; // Convert to microseconds
    cout << "Parallel Time: " << fixed << setprecision(2) << parDuration << " microseconds\n";
    cout << "Parallel Correctness: " << (compareMatrices(parResult, eigenResult, rowsA, columnsB) ? "Correct" : "Incorrect") << "\n";

    // Print results (optional)
    cout << "\nSequential Result:\n";
    printMatrix(seqResult, rowsA, columnsB);
    cout << "\nParallel Result:\n";
    printMatrix(parResult, rowsA, columnsB);
    cout << "\nEigen Result:\n";
    cout << fixed << setprecision(2);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < columnsB; ++j) {
            cout << setw(8) << eigenResult(i, j) << " ";
        }
        cout << endl;
    }

    // Free small matrices
    for (int i = 0; i < rowsA; ++i) {
        delete[] matrixA[i];
        if (seqResult) delete[] seqResult[i];
        if (parResult) delete[] parResult[i];
    }
    for (int i = 0; i < rowsB; ++i) {
        delete[] matrixB[i];
    }
    delete[] matrixA;
    delete[] matrixB;
    delete[] seqResult;
    delete[] parResult;

    // Test 2: Large matrices (100x100 * 100x100)
    cout << "\n=== Test 2: Large Matrices (100x100 * 100x100) ===\n";
    int largeRowsA = 100, largeColsA = 100, largeRowsB = 100, largeColsB = 100;
    double **largeA = new double *[largeRowsA];
    for (int i = 0; i < largeRowsA; ++i) {
        largeA[i] = new double[largeColsA];
        for (int j = 0; j < largeColsA; ++j) {
            largeA[i][j] = i * largeColsA + j + 1.0; // Unique values
        }
    }
    double **largeB = new double *[largeRowsB];
    for (int i = 0; i < largeRowsB; ++i) {
        largeB[i] = new double[largeColsB];
        for (int j = 0; j < largeColsB; ++j) {
            largeB[i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix
        }
    }

    // Compute Eigen result for large matrices
    MatrixXd eigenLargeA(largeRowsA, largeColsA);
    for (int i = 0; i < largeRowsA; ++i) {
        for (int j = 0; j < largeColsA; ++j) {
            eigenLargeA(i, j) = largeA[i][j];
        }
    }
    MatrixXd eigenLargeB = MatrixXd::Identity(largeRowsB, largeColsB);
    MatrixXd eigenLargeResult = eigenLargeA * eigenLargeB;

    // Time and test sequential function
    startSeq = omp_get_wtime();
    seqResult = naively_matrix_multiplication_sequential(largeA, largeRowsA, largeColsA, largeB, largeRowsB, largeColsB);
    endSeq = omp_get_wtime();
    seqDuration = (endSeq - startSeq) * 1e6; // Convert to microseconds
    cout << "Sequential Time: " << fixed << setprecision(2) << seqDuration << " microseconds\n";
    cout << "Sequential Correctness: " << (compareMatrices(seqResult, eigenLargeResult, largeRowsA, largeColsB) ? "Correct" : "Incorrect") << "\n";

    // Time and test parallel function
    startPar = omp_get_wtime();
    parResult = naively_matrix_multiplication_parallel(largeA, largeRowsA, largeColsA, largeB, largeRowsB, largeColsB);
    endPar = omp_get_wtime();
    parDuration = (endPar - startPar) * 1e6; // Convert to microseconds
    cout << "Parallel Time: " << fixed << setprecision(2) << parDuration << " microseconds\n";
    cout << "Parallel Correctness: " << (compareMatrices(parResult, eigenLargeResult, largeRowsA, largeColsB) ? "Correct" : "Incorrect") << "\n";

    // Free large matrices
    for (int i = 0; i < largeRowsA; ++i) {
        delete[] largeA[i];
        if (seqResult) delete[] seqResult[i];
        if (parResult) delete[] parResult[i];
    }
    for (int i = 0; i < largeRowsB; ++i) {
        delete[] largeB[i];
    }
    delete[] largeA;
    delete[] largeB;
    delete[] seqResult;
    delete[] parResult;

    return 0;
}

