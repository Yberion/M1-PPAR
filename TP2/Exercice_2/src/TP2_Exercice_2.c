#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

static int generateRandomNumberRange(int min, int max)
{
    return (rand() % (max - min + 1)) + min;
}

static void freeMatrix(int **matrix, int dimension)
{
    for (int i = 0; i < dimension; ++i)
    {
        free(matrix[i]);
    }

    free(matrix);
}

static int** generateZeroSquareMatrix(int dimension)
{
    int **matrix = (int**)calloc(dimension, sizeof(int*));

    if (!matrix)
    {
        fputs("unable to allocate memory for the matrix", stderr);

        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dimension; ++i)
    {
        matrix[i] = (int*)calloc(dimension, sizeof(int));

        if (!matrix[i])
        {
            freeMatrix(matrix, dimension);

            exit(EXIT_FAILURE);
        }
    }

    return matrix;
}

static int** generateRandomSquareMatrix(int dimension)
{
    srand(time(NULL));

    int **matrix = (int**)calloc(dimension, sizeof(int*));

    if (!matrix)
    {
        fputs("unable to allocate memory for the matrix", stderr);

        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dimension; ++i)
    {
        matrix[i] = (int*)calloc(dimension, sizeof(int));

        if (!matrix[i])
        {
            freeMatrix(matrix, dimension);

            exit(EXIT_FAILURE);
        }
    }

// Declare a new team of thread to work in parallel to initialize the matrix with random numbers
#pragma omp parallel for
    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            matrix[i][j] = generateRandomNumberRange(1, 100);
        }
    }
// pragma omp parallel for

    return matrix;
}

static double sequentialMultiplicationMatrix(int **matrixResult, int **matrixA, int **matrixB, int dimension)
{
    struct timeval start;
    gettimeofday(&start, 0);

    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            for (int k = 0; k < dimension; ++k)
            {
                matrixResult[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    struct timeval end;
    gettimeofday(&end, 0);

    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0;
}

static double parallelMultiplicationMatrix(int **matrixResult, int **matrixA, int **matrixB, int dimension)
{
    struct timeval start;
    gettimeofday(&start, 0);

// Declare a new team of thread to work in parallel to initialize the matrix with random numbers
#pragma omp parallel for
    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            for (int k = 0; k < dimension; ++k)
            {
                matrixResult[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
// pragma omp parallel for

    struct timeval end;
    gettimeofday(&end, 0);

    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0;
}

static void displayMatrix(int **matrix, int dimension)
{
    printf("-------------- Display matrix : \n");
    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; ++j)
        {
            printf("%d\t", matrix[i][j]);
        }

        printf("\n");
    }
    printf("--------------------------------\n");
}

int main(void)
{
    int n = 0;

    puts("Enter n (> 0)");

    scanf("%d", &n);

    if (n <= 0)
    {
        fputs("n must be > 0", stderr);

        return EXIT_FAILURE;
    }

    int **matrixA = generateRandomSquareMatrix(n);
    int **matrixB = generateRandomSquareMatrix(n);

    int **matrixResult = generateZeroSquareMatrix(n);

    //displayMatrix(matrixA, n);
    //displayMatrix(matrixB, n);

    double timeSequential = sequentialMultiplicationMatrix(matrixResult, matrixA, matrixB, n);
    double timeParallel = parallelMultiplicationMatrix(matrixResult, matrixA, matrixB, n);

    //displayMatrix(matrixResult, n);

    printf("Sequential: %f\nParallel: %f\n", timeSequential, timeParallel);

    /*
     * Test using my VM on my computer:
     *
     * Enter n (> 0)
     * 1400
     * Sequential: 33.051908
     * Parallel: 8.781665
     *
     */

    freeMatrix(matrixResult, n);
    freeMatrix(matrixB, n);
    freeMatrix(matrixA, n);

    return EXIT_SUCCESS;
}
