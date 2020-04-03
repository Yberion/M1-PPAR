#include "utils.h"
#include <stdlib.h>

#include "summation_kernel.cu"

/*
// CPU implementation
float log2_series_brandon(int n)
{
	float result = 0.0F;

	for (int i = 0; i < n; ++i)
	{
		result += (powf(-1, i)) / (i + 1);
	}
	
	return result;
}
*/

/*
// CPU implementation
float log2_series_thomas(int n)
{
	float res = 0.0F;
	
    int op = 1;
	
    for(int i=1; i<=n; i++)
	{
        res += (float) 1/i * op;
        op *= -1;
    }
	
	return res;
}
*/

// CPU implementation
float log2_series(int n)
{
	float res = 0.0F;
	
    int op = 1;
	
    for(int i=1; i<=n; i++)
	{
        res += (float) 1/i * op;
        op *= -1;
    }
	
	return res;
}

int main(int argc, char ** argv)
{
    int data_size = 1024 * 1024 * 128;

    // Run CPU version
    double start_time = getclock();
    float log2 = log2_series(data_size);
    double end_time = getclock();
    
    printf("CPU result: %f\n", log2);
    printf(" log(2)=%f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);
    
    // Parameter definition
    int threads_per_block = 4 * 32;
    int blocks_in_grid = 8;
    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    int results_size = num_threads;
    float * data_out_cpu;
    // Allocating output data on CPU
	// TODO

	// Allocating output data on GPU
    // TODO

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute kernel
    // TODO

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back
    // TODO
    
    // Finish reduction
    // TODO
	float sum = 0.;
    
    // Cleanup
    // TODO
    
    printf("GPU results:\n");
    printf(" Sum: %f\n", sum);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    return 0;
}
