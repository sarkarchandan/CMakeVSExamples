#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdditionKernel(int* a, int* b, int* c, int n) {
	// Compute global thread id (tid)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// Vector boundary guard
	if (tid < n)
		c[tid] = a[tid] + b[tid];
}

// Initialize vector of size n to int between 0-99
void vector_init(int* a, int n) {
	for (int i = 0; i < n; i++)
		a[i] = rand() % 100;
}

// Check vector add result
void errorCheck(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++)
		assert(c[i] == a[i] + b[i]);
}
