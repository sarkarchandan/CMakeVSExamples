#include "vecadd.h"

__global__ void vectorAdditionKernel(int* a, int* b, int* c, int n);
void vector_init(int* a, int n);
void errorCheck(int* a, int* b, int* c, int n);


void assertVectorAddition() {
	// Vector of size 2^16
	int n = 1 << 16;

	// Host vector pointers
	int* h_a, * h_b, * h_c;

	// Device vector pointers
	int* d_a, * d_b, * d_c;

	// Allocation size for all vectors
	size_t bytes = sizeof(int) * n;

	// Allocate host memory
	h_a = (int*)malloc(bytes);
	h_b = (int*)malloc(bytes);
	h_c = (int*)malloc(bytes);

	// Allocate device memory
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	vector_init(h_a, n);
	vector_init(h_b, n);

	// Copy data from host memory to GPU memory
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);


	// Declare thread block size
	int NUM_THREADS = 256;

	// Declare grid size. Each thread block has NUM_THREADS number of 
	// threads, and here we want to determine, how many such blocks 
	// of threads we need. We do ceiling to ensure the sufficient number of 
	// blocks. Here, in this example, we want a single thread to handle 
	// a single constant time addition, hence we are ensuring, that we have 
	// sufficient threads available.
	int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

	// NOTE: THe number of threads and grid / block sizes are not random. 
	// It depends on the GPU architectures. Here in this example we have 
	// made some safer assumptions. However, we have to maintain the 
	// consistency with respect to the warp size, which is always 32. 
	// Hence, the number of threads should always be a multiple of 32.


	// Launch vector addition kernel
	vectorAdditionKernel << < NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, n);

	// Copy sum vector from device to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Check result for errors
	errorCheck(h_a, h_b, h_c, n);

	printf("Vector addition completed successfully !!!");
}