#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

typedef void (*primeFunctionPtr_t)(bool*, size_t);

static size_t countPrimeNumbers(bool *primes, size_t size)
{
    size_t count = 0;

    for (size_t i = 2; i < size; ++i)
    {
        if (primes[i])
        {
            count += 1;
        }
    }

    return count;
}

static void displayPrimeNumbers(bool *primes, size_t size)
{
    for (size_t i = 2; i < size; ++i)
    {
        if (primes[i])
        {
            printf("%zu ", i);
        }
    }

    puts("");
}

static void EratostheneSequential(bool *primes, size_t size)
{
    size_t lastSqrt = (size_t)sqrt((double)size);

    for (size_t i = 3; i < lastSqrt; i += 2)
    {
        if (primes[i / 2] == true)
        {
            for (size_t j = i * i; j < size; j += 2 * i)
            {
                primes[j / 2] = false;
            }
        }
    }
}

static void EratostheneParallel(bool *primes, size_t size)
{
    omp_set_num_threads(omp_get_num_procs());

    size_t lastSqrt = (size_t)sqrt((double)size);

// Declare a new team of thread to work in parallel to initialize the matrix with random numbers
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 3; i < lastSqrt; i += 2)
    {
        if (primes[i / 2] == true)
        {
            for (size_t j = i * i; j < size; j += 2 * i)
            {
                primes[j / 2] = false;
            }
        }
    }
// pragma omp parallel for
}

static bool* generateTrueArray(size_t size)
{
    bool *tmp = (bool*)calloc(size, sizeof(bool));

    if (!tmp)
    {
        fputs("unable to allocate memory for the prime array", stderr);

        return NULL;
    }

    memset(tmp, true, size);

    return tmp;
}

static double primeFunctionExecutionTime(primeFunctionPtr_t primeFunction, bool *primes, size_t size)
{
    struct timeval start;
    gettimeofday(&start, 0);

    primeFunction(primes, size);

    struct timeval end;
    gettimeofday(&end, 0);

    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0;
}

int main(void)
{
    int exitStatus = EXIT_FAILURE;

    size_t n = 0;

    puts("Enter n (> 0)");

    scanf("%zu", &n);

    if (n <= 0)
    {
        fputs("n must be > 0", stderr);

        return EXIT_FAILURE;
    }

    int display = 0;

    puts("Display prime numbers ? (yes = 1, no = 0, default no)");

    scanf("%d", &display);

    if (display != 1)
    {
        display = 0;
    }

    bool *primesSequential = generateTrueArray(n);

    if (!primesSequential)
    {
        goto exit;
    }

    bool *primesParallel = generateTrueArray(n);

    if (!primesParallel)
    {
        goto freeMemorySeqAndExit;
    }

    double timeSequential = primeFunctionExecutionTime(EratostheneSequential, primesSequential, n);
    double timeParallel = primeFunctionExecutionTime(EratostheneParallel, primesParallel, n);

    /*
     * Enter n (> 0)
     300000000
     Display prime numbers ? (yes = 1, no = 0, default no)
     0
     Sequential: 1.951293 (count: 166252323)
     Parallel: 0.927194 (count: 166252323)
     *
     */

    if (display == 1)
    {
        printf("Sequential -------------------\n");
        displayPrimeNumbers(primesSequential, n);

        printf("Parallel -------------------\n");
        displayPrimeNumbers(primesParallel, n);
        printf("------------------------------\n");
    }

    printf("Sequential: %f (count: %zu)\nParallel: %f (count: %zu)\n", timeSequential, countPrimeNumbers(primesSequential, n), timeParallel,
            countPrimeNumbers(primesParallel, n));

    exitStatus = EXIT_SUCCESS;

    free(primesParallel);
freeMemorySeqAndExit:
    free(primesSequential);
exit:

    return exitStatus;
}
