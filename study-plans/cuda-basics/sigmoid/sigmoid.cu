#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sigmoid_kernel<<<blocks, threads>>>(input, output, N);

    cudaDeviceSynchronize();
}