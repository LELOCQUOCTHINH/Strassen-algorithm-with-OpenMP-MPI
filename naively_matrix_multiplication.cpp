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

double ** naively_matrix_multiplication_openmp(double ** matrixA, int rowsA, int columnsA, double ** matrixB, int rowsB, int columnsB)
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
        #pragma openmp parallel if(rowsA > 100 || columnsB > 100 || columnsA > 100)
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
        #pragma openmp parallel if(rowsA > 100 || columnsB > 100 || columnsA > 100)
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

// ------------------------------------------------------------------
// 2. NEW: Convert double** ↔ double* (row-major flat)
// ------------------------------------------------------------------
double* to_flat_double_star(double** matrix, int rows, int cols) {
    double* flat = new double[rows * cols];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = matrix[i][j];
    return flat;
}

double** from_flat_double_star(const double* flat, int rows, int cols) {
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[cols];
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = flat[i * cols + j];
    }
    return matrix;
}

// ------------------------------------------------------------------
// 3. NEW: GPU-accelerated naive multiply (flat double*)
// ------------------------------------------------------------------
double* naively_matrix_multiplication_openmp_cuda(
    const double* A_flat, int rowsA, int colsA,
    const double* B_flat, int rowsB, int colsB)
{
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0) {
        cerr << "Invalid matrix dimensions!\n";
        return nullptr;
    }
    if (colsA != rowsB) {
        cerr << "Incompatible dimensions for multiplication!\n";
        return nullptr;
    }

    double* C_flat = new double[rowsA * colsB]();

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A_flat[0:rowsA*colsA], B_flat[0:rowsB*colsB]) \
        map(from: C_flat[0:rowsA*colsB])
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                sum += A_flat[i * colsA + k] * B_flat[k * colsB + j];
            }
            C_flat[i * colsB + j] = sum;
        }
    }

    return C_flat;
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


