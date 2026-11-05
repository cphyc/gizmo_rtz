#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <bit>
#include <cstdint>
#define CUDA_OK(expr) do { auto code = (expr); if (code != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(code)); std::exit(2); } } while (0)
__global__ void real_power(const double* input, double* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = ::pow(input[i], 3.0);
}
__global__ void integer_power(const double* input, double* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = ::pow(input[i], 3);
}
__global__ void product_power(const double* input, double* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * input[i] * input[i];
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* file = std::fopen(argv[1], "rb"); if (!file) return 2;
    std::fseek(file, 0, SEEK_END); auto bytes = std::ftell(file); std::rewind(file);
    std::vector<double> input(bytes / sizeof(double)), a(input.size()), b(input.size()), c(input.size());
    if (std::fread(input.data(), sizeof(double), input.size(), file) != input.size()) return 2;
    std::fclose(file);
    int n = input.size(); double *x, *y, *z, *w;
    CUDA_OK(cudaMalloc(&x, bytes)); CUDA_OK(cudaMalloc(&y, bytes)); CUDA_OK(cudaMalloc(&z, bytes)); CUDA_OK(cudaMalloc(&w, bytes));
    CUDA_OK(cudaMemcpy(x, input.data(), bytes, cudaMemcpyHostToDevice));
    real_power<<<(n+127)/128,128>>>(x,y,n); integer_power<<<(n+127)/128,128>>>(x,z,n); product_power<<<(n+127)/128,128>>>(x,w,n);
    CUDA_OK(cudaDeviceSynchronize());
    cudaEvent_t start, stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    constexpr int repeats = 100;
    CUDA_OK(cudaEventRecord(start));
    for (int i=0; i<repeats; ++i) real_power<<<(n+127)/128,128>>>(x,y,n);
    CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop)); float real_ms;
    CUDA_OK(cudaEventElapsedTime(&real_ms,start,stop));
    CUDA_OK(cudaEventRecord(start));
    for (int i=0; i<repeats; ++i) integer_power<<<(n+127)/128,128>>>(x,z,n);
    CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop)); float integer_ms;
    CUDA_OK(cudaEventElapsedTime(&integer_ms,start,stop));
    CUDA_OK(cudaEventRecord(start));
    for (int i=0; i<repeats; ++i) product_power<<<(n+127)/128,128>>>(x,w,n);
    CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop)); float product_ms;
    CUDA_OK(cudaEventElapsedTime(&product_ms,start,stop));
    CUDA_OK(cudaMemcpy(a.data(), y, bytes, cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(b.data(), z, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(c.data(), w, bytes, cudaMemcpyDeviceToHost));
    int different=0, product_different=0; double max_relative=0, product_max_relative=0;
    for (int i=0; i<n; ++i) {
        different += std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]);
        max_relative = std::fmax(max_relative,std::fabs(a[i]-b[i])/std::fabs(a[i]));
        product_different += std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(c[i]);
        product_max_relative = std::fmax(product_max_relative,std::fabs(a[i]-c[i])/std::fabs(a[i]));
    }
    std::printf("{\"precision_bytes\":8,\"samples\":%d,\"repeats\":%d,\"real_exponent_ms\":%.9g,\"integer_exponent_ms\":%.9g,\"product_ms\":%.9g,\"different_bit_patterns\":%d,\"max_relative_difference\":%.17g,\"product_different_bit_patterns\":%d,\"product_max_relative_difference\":%.17g}\n",n,repeats,real_ms/repeats,integer_ms/repeats,product_ms/repeats,different,max_relative,product_different,product_max_relative);
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    CUDA_OK(cudaFree(x)); CUDA_OK(cudaFree(y)); CUDA_OK(cudaFree(z)); CUDA_OK(cudaFree(w));
}
