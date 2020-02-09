#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void)
{
    int numberThreads = 6;
    int threadId = 0;

    omp_set_num_threads(numberThreads);

#pragma omp parallel private(numberThreads, threadId)
    {
        threadId = omp_get_thread_num();

        printf("(thread id: %d) Hello, world!\n", threadId);

        if (threadId == 0)
        {
            printf("(thread id: %d) There's a total of %d threads working in parallel.", threadId, omp_get_num_threads());
        }
    }
// pragma omp parallel private(numberThreads, threadId)

    return EXIT_SUCCESS;
}
