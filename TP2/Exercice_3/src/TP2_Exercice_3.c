#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void EratostheneSequential(int n)
{
    bool primes[n] = { 0 };

    for (int p = 2; p * p <= n; ++p)
    {
        if (primes[p] == true)
        {
            for (int i = p * p; i <= n; i += p)
            {
                primes[i] = false;
            }
        }
    }
}

int main(void)
{
    return EXIT_SUCCESS;
}
