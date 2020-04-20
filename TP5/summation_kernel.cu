
// GPU kernel
// data_size = data_size_per_thread
__global__ void summation_kernel(int data_size, float* data_out)
{
	// Question 8
	extern __shared__ float s_res[];

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	float res = 0.0F;
	int op = -1;

	for(int j = ind * data_size; j < (ind + 1) * data_size; j++)
	{
		res += j == 0 ? 0 : (float) 1 / j * op;
		op *= -1;
	}

	//data_out[ind] = res;

	// Question 8

	s_res[tid] = res;

	__syncthreads();

	if(tid == 0)
	{
		for(int i = 1; i < blockDim.x; i++)
		{
			res += s_res[i];
		}
		
		data_out[blockIdx.x] = res;
	}

	// Question 9

	__syncthreads();

	res = 0.0F;

	if (ind == 0)
	{
		for (int i = 0; i < gridDim.x; ++i)
		{
			res += data_out[i];
		}
		
		// Clean memory of the first "gridDim.x" elements of the global memory "data_out"
		// because this is the only things being modified, the rest are only 0
		memset(data_out, 0, gridDim.x);
		
		// store the final result in the first indice (0)
		data_out[0] = res;
	}
}

// GPU kernel
// data_size = data_size_per_thread
__global__ void summation_kernel_2(int data_size, float* data_out)
{
	int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

	int op;
	float res = 0.0F;

	for (int i = 0; i < data_size; ++i)
	{
		op = (threadNumber % 2 == 0) ? -1 : 1;

		res += (i == 0 && threadNumber == 0) ? 0 : (float) 1 / (threadNumber + (i * num_threads)) * op;
	}

	data_out[threadNumber] = res;
}
