#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int k = 0;

    if (argc < 2)
    {
        puts("Enter k (>= 0)");

        scanf("%d", &k);

        // TODO check
    }
    else
    {
        char *tmp;

        k = (int)strtol(argv[1], &tmp, 10);

        // TODO check
    }

    if (k < 0)
    {
        fputs("k must be >= 0", stderr);

        return EXIT_FAILURE;
    }

    int result = 1;

    // TODO check

    for (int i = 0; i < k; ++i)
    {
        result *= 2;
    }

    printf("Result: %d\n", result);

    // for n = 8 -> 0 1 2 3 4 5 6 7
    int *array = (int*)calloc(result, sizeof(int));

    for (int i = 0; i < result; ++i)
    {
        array[i] = i;
    }

    // Complexity O(k)

    // Exercice 1 B

    // Non l'algo n'est pas bon pour la parallélisation

    // Exercice 1 C

    for (int i = 0; i < k; ++i)
    {
        //for (int j = 0; )
    }

    free(array);

    return EXIT_SUCCESS;
}
