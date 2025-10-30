#include "naively_matrix_multiplication.h"

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
bool compareMatrices(double **matrix1, const MatrixXd &matrix2, int rows, int columns, double epsilon) {
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


