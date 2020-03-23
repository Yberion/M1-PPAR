
/*
 * Conway's Game of Life
 *
 * A. Mucherino
 *
 * PPAR, TP4
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#if defined(_MSC_VER)
    #include <windows.h>
#else
    #include <unistd.h>
    #include <string.h>
#endif

#define N 32
#define itMax 20

#define VAL_CHAR_EMPTY 0
#define VAL_CHAR_O 1
#define VAL_CHAR_X 2

#if defined(_MSC_VER)
static void Thread_Sleep(DWORD seconds)
    {
        Sleep(1000 * seconds);
    }
#else
    static void Thread_Sleep(unsigned int seconds)
    {
        sleep(seconds);
    }
#endif

// Allocation dynamique du tableau
unsigned int* allocate()
{
    return (unsigned int*)calloc(N * N, sizeof(unsigned int));
}

// conversion cell location : 2d --> 1d
// (row by row)
int code(int x, int y, int neighbourX, int neighbourY)
{
    int row = (x + neighbourX) % N;
    int column = (y + neighbourY) % N;

    if (row < 0)
    {
        row = N + row;
    }

    if (column < 0)
    {
        column = N + column;
    }

    return row * N + column;
}

// writing into a cell location 
void write_cell(int x, int y, unsigned int value, unsigned int* world)
{
    int k;

    k = code(x, y, 0, 0);

    world[k] = value;
}

// random generation
unsigned int* initialize_random()
{
    int x;
    int y;
    unsigned int cell;
    unsigned int* world;

    world = allocate();

    for (x = 0; x < N; x++)
    {
        for (y = 0; y < N; y++)
        {
            if (rand() % 5 != 0)
            {
                cell = VAL_CHAR_EMPTY;
            }
            else if (rand() % 2 == 0)
            {
                cell = VAL_CHAR_O;
            }
            else
            {
                cell = VAL_CHAR_X;
            }

            write_cell(x, y, cell, world);
        }
    }

    return world;
}

// dummy generation
unsigned int* initialize_dummy()
{
    int x;
    int y;
    unsigned int* world;

    world = allocate();

    for (x = 0; x < N; x++)
    {
        for (y = 0; y < N; y++)
        {
            write_cell(x, y, x % 3, world);
        }
    }

    return world;
};

// "glider" generation
unsigned int* initialize_glider()
{
    int x;
    int y;
    int mx;
    int my;
    unsigned int* world;

    world = allocate();

    for (x = 0; x < N; x++)
    {
        for (y = 0; y < N; y++)
        {
            write_cell(x, y, VAL_CHAR_EMPTY, world);
        }
    }

    mx = N / 2 - 1;
    my = N / 2 - 1;
    x = mx;
    y = my + 1;

    write_cell(x, y, VAL_CHAR_O, world);

    x = mx + 1;
    y = my + 2;

    write_cell(x, y, VAL_CHAR_O, world);

    x = mx + 2;
    y = my;

    write_cell(x, y, VAL_CHAR_O, world);

    y = my + 1;

    write_cell(x, y, VAL_CHAR_O, world);

    y = my + 2;
    
    write_cell(x, y, VAL_CHAR_O, world);

    return world;
}

// "small exploder" generation
unsigned int* initialize_small_exploder()
{
    int x;
    int y;
    int mx;
    int my;
    unsigned int* world;

    world = allocate();

    for (x = 0; x < N; x++)
    {
        for (y = 0; y < N; y++)
        {
            write_cell(x, y, VAL_CHAR_EMPTY, world);
        }
    }

    mx = N / 2 - 2;
    my = N / 2 - 2;
    x = mx;
    y = my + 1;

    write_cell(x, y, VAL_CHAR_X, world);

    x = mx + 1;
    y = my;

    write_cell(x, y, VAL_CHAR_X, world);

    y = my + 1;

    write_cell(x, y, VAL_CHAR_X, world);

    y = my + 2;

    write_cell(x, y, VAL_CHAR_X, world);

    x = mx + 2;
    y = my;

    write_cell(x, y, VAL_CHAR_X, world);

    y = my + 2;

    write_cell(x, y, VAL_CHAR_X, world);

    x = mx + 3;
    y = my + 1;

    write_cell(x, y, VAL_CHAR_X, world);

    return world;
}

// reading a cell
int read_cell(int x, int y, int neighbourX, int neighbourY, unsigned int* world)
{
    int k = code(x, y, neighbourX, neighbourY);

    return world[k];
}

// updating counters
void update(int x, int y, int neighbourX, int neighbourY, unsigned int* world, int* neighbourNumbers, int* numberNeighboursO, int* numberNeighboursX)
{
    unsigned int cell = read_cell(x, y, neighbourX, neighbourY, world);

    if (cell != VAL_CHAR_EMPTY)
    {
        (*neighbourNumbers)++;

        if (cell == VAL_CHAR_O)
        {
            (*numberNeighboursO)++;
        }
        else
        {
            (*numberNeighboursX)++;
        }
    }
}

// looking around the cell
void neighbors(int x, int y, unsigned int* world, int* neighbourNumbers, int* numberNeighboursO, int* numberNeighboursX)
{
    int neighbourX;
    int neighbourY;

    (*neighbourNumbers) = 0;
    (*numberNeighboursO) = 0;
    (*numberNeighboursX) = 0;

    // same line
    neighbourX = -1;
    neighbourY = 0;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);
    
    neighbourX = +1;
    neighbourY = 0;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);

    // one line down
    neighbourX = -1;
    neighbourY = +1;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);
    
    neighbourX = 0;
    neighbourY = +1;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);
    
    neighbourX = +1;
    neighbourY = +1;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);

    // one line up
    neighbourX = -1;
    neighbourY = -1;

    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);
    
    neighbourX = 0;
    neighbourY = -1;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);
    
    neighbourX = +1;
    neighbourY = -1;
    
    update(x, y, neighbourX, neighbourY, world, neighbourNumbers, numberNeighboursO, numberNeighboursX);
}

// computing a new generation
short generateNewState(unsigned int* world1, unsigned int* world2, int xStart, int xEnd)
{
    int x;
    int y;
    int neighbourNumbers;
    int numberNeighboursO;
    int numberNeighboursX;
    unsigned int currentCellVal;
    short change = 1;

    // cleaning destination world
    /*
    for (x = 0; x < N; x++)
    {
        for (y = 0; y < N; y++)
        {
            write_cell(x, y, 0, world2);
        }
    }
    */

    memset(world2, 0, N * N);

    // generating the new world
    for (x = xStart; x <= xEnd; x++)
    {
        for (y = 0; y < N; y++)
        {
            currentCellVal = read_cell(x, y, 0, 0, world1);

            neighbors(x, y, world1, &neighbourNumbers, &numberNeighboursO, &numberNeighboursX);

            if (neighbourNumbers < 2)
            {
                write_cell(x, y, VAL_CHAR_EMPTY, world2);

                continue;
            }

            if (neighbourNumbers > 3)
            {
                write_cell(x, y, VAL_CHAR_EMPTY, world2);

                continue;
            }

            if (currentCellVal == VAL_CHAR_EMPTY && neighbourNumbers == 3)
            {
                if (numberNeighboursO > numberNeighboursX)
                {
                    write_cell(x, y, VAL_CHAR_O, world2);
                }
                else // (numberNeighboursX > numberNeighboursO)
                {
                    write_cell(x, y, VAL_CHAR_X, world2);
                }

                continue;
            }

            write_cell(x, y, currentCellVal, world2);
        }
    }

    return change;
}

// cleaning the screen
void cls()
{
    int i;

    for (i = 0; i < 3; i++)
    {
        puts("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    }
}

// diplaying the world
void print(unsigned int* world)
{
    int i;

    // cls();

    for (i = 0; i < N; i++)
    {
        fputs("-", stdout);
    }

    for (i = 0; i < N * N; i++)
    {
        if (i % N == 0)
        {
            puts("");
        }

        if (world[i] == VAL_CHAR_EMPTY)
        {
            fputs(" ", stdout);
        }

        if (world[i] == VAL_CHAR_O)
        {
            fputs("o", stdout);
        }

        if (world[i] == VAL_CHAR_X)
        {
            fputs("x", stdout);
        }
    }

    puts("");

    for (i = 0; i < N; i++)
    {
        fputs("-", stdout);
    }

    puts("");

    Thread_Sleep(1);
}


// main
int main(int argc, char **argv)
{
    int it;
    int change;
    unsigned int* world1;
    unsigned int* world2;
    unsigned int* tmpSwapWorldPtr;
    
    int sectionSize;
    int sectionStart;
    int sectionEnd;

    int currentRank;
    int processNumber;
    int aboverank;
    int belowrank;
    int targetIndex;

    MPI_Status status;
    MPI_Request request;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &currentRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processNumber);
    
    if (N % processNumber != 0)
    {
        printf("N is not divisible by process number\n");

        MPI_Finalize();

        exit(EXIT_FAILURE);
    }

    belowrank = (currentRank == processNumber - 1) ? 0 : currentRank + 1;
    aboverank = (currentRank == 0) ? processNumber - 1 : currentRank - 1;
    
    if (currentRank == 0)
    {
        // getting started  
        // world1 = initialize_dummy();
        //world1 = initialize_random();
        //world1 = initialize_glider();
        world1 = initialize_small_exploder();

        print(world1);
    }
    else
    {
        world1 = allocate();
    }

    MPI_Bcast(world1, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    world2 = allocate();

    // Computing first and last row index
    sectionSize = N / processNumber;
    sectionStart = sectionSize * currentRank;
    sectionEnd = sectionStart + sectionSize - 1;

    it = 0;
    change = 1;

    while (change && it < itMax)
    {
        change = generateNewState(world1, world2, sectionStart, sectionEnd);

        tmpSwapWorldPtr = world1;
        world1 = world2;
        world2 = tmpSwapWorldPtr;

        tmpSwapWorldPtr = NULL;

        targetIndex = sectionEnd * N;

        MPI_Isend(&world1[targetIndex], N, MPI_INT, belowrank, 0, MPI_COMM_WORLD, &request);

        targetIndex = (sectionStart == 0) ? N - 1 : sectionStart - 1;
        targetIndex = targetIndex * N;
        
        MPI_Recv(&world1[targetIndex], N, MPI_INT, aboverank, 0, MPI_COMM_WORLD, &status);

        MPI_Wait(&request, &status);

        targetIndex = sectionStart * N;

        MPI_Isend(&world1[targetIndex], N, MPI_INT, aboverank, 0, MPI_COMM_WORLD, &request);

        targetIndex = (sectionEnd == N - 1) ? 0 : sectionEnd + 1;
        targetIndex = targetIndex * N;

        MPI_Recv(&world1[targetIndex], N, MPI_INT, belowrank, 0, MPI_COMM_WORLD, &status);

        MPI_Wait(&request, &status);

        it++;
    }

    memcpy(&world2[sectionStart * N], &world1[sectionStart * N], N);

    MPI_Gather(&world2[sectionStart * N], N * sectionSize, MPI_INT, world1, N * sectionSize, MPI_INT, 0, MPI_COMM_WORLD);

    if (currentRank == 0)
    {
        print(world1);
    }

    // ending
    free(world1);
    free(world2);

    MPI_Finalize();

    return 0;
}
