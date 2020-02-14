#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

typedef void (*primeFunctionPtr_t)(bool*, size_t);

static unsigned int countPrimeNumbers(bool *primes, size_t size)
{
    unsigned int count = 0;

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
    for (size_t p = 2; p * p < size; ++p)
    {
        if (primes[p] == true)
        {
            for (size_t i = p * p; i < size; i += p)
            {
                primes[i] = false;
            }
        }
    }
}

static void EratostheneParallel(bool *primes, size_t size)
{
    for (size_t p = 2; p * p < size; ++p)
    {
        if (primes[p] == true)
        {
            for (size_t i = p * p; i < size; i += p)
            {
                primes[i] = false;
            }
        }
    }
}

static bool* generateTrueArray(size_t size)
{
    bool* tmp = (bool*)calloc(size, sizeof(bool));

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

    bool* primesSequential = generateTrueArray(n);

    if (!primesSequential)
    {
        goto exit;
    }

    bool* primesParallel = generateTrueArray(n);

    if (!primesParallel)
    {
        goto freeMemorySeqAndExit;
    }

    double timeSequential = primeFunctionExecutionTime(EratostheneSequential, primesSequential, n);
    double timeParallel = primeFunctionExecutionTime(EratostheneParallel, primesSequential, n);

    if (display == 1)
    {
        printf("Sequential -------------------\n");
        displayPrimeNumbers(primesSequential, n);

        printf("Parallel -------------------\n");
        displayPrimeNumbers(primesParallel, n);
        printf("------------------------------\n");
    }

    printf("Sequential: %f\nParallel: %f\n", timeSequential, timeParallel);

    exitStatus = EXIT_SUCCESS;

    free(primesParallel);
freeMemorySeqAndExit:
    free(primesSequential);
exit:

    return exitStatus;
}
