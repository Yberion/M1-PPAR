
// GPU kernel
// data_size = data_size_per_thread
__global__ void summation_kernel(int data_size, float* data_out)
{
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	float res = 0.0F;
	int op = -1;
	for(int j = ind*data_size; j < (ind+1)*data_size; j++){
		res += j == 0 ? 0 : (float) 1/j * op;
		op *= -1;
	}
	data_out[ind] = res;
}

// GPU kernel
__global__ void summation_kernel_2(int data_size, float* data_out)
{
	int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;

	int op;
	float res = 0.0F;

	for (int i = 0; i < data_size: ++i)
	{
		op = (i % 2 == 0) ? 1 : -1;

		res = (i == 0) ? 0 : (float) 1 / (i * threadNumber) * op;
	}

	data_out[threadNumber] = res;
}