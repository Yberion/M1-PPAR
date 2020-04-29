// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int* source_domain, int x, int y, int dx, int dy, unsigned int domain_x, unsigned int domain_y)
{
    // Wrap around
    x = (unsigned int)(x + dx) % domain_x;
    y = (unsigned int)(y + dy) % domain_y;

    return source_domain[y * domain_x + x];
}

// Compute kernel
__global__ void life_kernel(int* source_domain, int* dest_domain, int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= domain_x || ty >= domain_y)
    {
        return;
    }

    // Read cell
    int myself = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);

    // TODO: Read the 8 neighbors and count number of blue and red
    int redcells = 0;
    int bluecells = 0;
    int cell;

    for (int line = -1; line < 2; ++line)
    {
        for (int column = -1; column < 2; ++column)
        {
            //Do not read myself
            if (!(line == 0 && column == 0))
            {
                cell = read_cell(source_domain, tx, ty, line, column, domain_x, domain_y);
                
                if (cell == 1)
                {
                    redcells++;
                }
                else if (cell == 2)
                {
                    bluecells++;
                }
            }
        }
    }

    // TODO: Compute new value
    int sum = redcells + bluecells;
    // By default, the cell dies (or stay empty)
    int newvalue = 0;

    if (myself == 0 && sum == 3)
    {
        // New cell
        newvalue = redcells > bluecells ? 1 : 2;
    }
    else if (sum == 2 || sum == 3)
    {
        // Survives
        newvalue = myself;
    }

    // TODO: Write it in dest_domain
    dest_domain[ty * domain_x + tx] = newvalue;
}

// Compute kernel
__global__ void life_kernel_q5(int* source_domain, int* dest_domain, int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= domain_x || ty >= domain_y)
    {
        return;
    }

    extern __shared__ int sharedData[];

    int ligneDessous = ((int)blockIdx.y - 1 < 0) ? gridDim.y - 1 : blockIdx.y - 1;
    int ligneDessus = (blockIdx.y + 1 >= gridDim.y) ? 0 : blockIdx.y + 1;

    // Ligne de dessus
    memcpy(&sharedData[0 * domain_x], &source_domain[blockIdx.y * domain_x], domain_x);
    // Ligne courante
    memcpy(&sharedData[1 * domain_x], &source_domain[ligneDessus * domain_x], domain_x);
    // Ligne de dessous
    memcpy(&sharedData[2 * domain_x], &source_domain[ligneDessous * domain_x], domain_x);

    // Read cell
    int myself = read_cell(sharedData, tx, ty, 0, 0, domain_x, domain_y);

    // TODO: Read the 8 neighbors and count number of blue and red
    int redcells = 0;
    int bluecells = 0;
    int cell;

    for (int line = -1; line < 2; ++line)
    {
        for (int column = -1; column < 2; ++column)
        {
            //Do not read myself
            if (!(line == 0 && column == 0))
            {
                cell = read_cell(sharedData, tx, ty, line, column, domain_x, domain_y);
                
                if (cell == 1)
                {
                    redcells++;
                }
                else if (cell == 2)
                {
                    bluecells++;
                }
            }
        }
    }

    // TODO: Compute new value
    int sum = redcells + bluecells;
    // By default, the cell dies (or stay empty)
    int newvalue = 0;

    if (myself == 0 && sum == 3)
    {
        // New cell
        newvalue = redcells > bluecells ? 1 : 2;
    }
    else if (sum == 2 || sum == 3)
    {
        // Survives
        newvalue = myself;
    }

    // TODO: Write it in dest_domain
    dest_domain[ty * domain_x + tx] = newvalue;
}

