
// GPU kernel
// data_size = data_size_per_thread
__global__ void summation_kernel(int data_size, float* data_out)
{
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
}

// GPU kernel
__global__ void summation_kernel_2(int data_size, float* data_out)
{
	int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;

	int op;
	float res = 0.0F;

	for (int i = 1; i <= data_size: ++i)
	{
		op = (i % 2 == 0) ? 1 : -1;

		res += (float) 1 / (threadNumber + (i * data_size)) * op;
	}

	data_out[threadNumber] = res;
}