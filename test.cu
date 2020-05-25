#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include "cublas_v2.h"
#include "nvml.h"

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

struct test_context {
  size_t n;
  float* a;
  float* b;
  float* c;
  cublasHandle_t h;
  nvmlDevice_t dev;
};

#define CHECK_CUDA_ERROR(err) \
  if ((err) != cudaSuccess) { \
    fprintf(stderr, "gpu-stress-test: CUDA error %s at %s:%d.\n", cudaGetErrorName(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }
#define CHECK_CUBLAS_ERROR(err) \
  if ((err) != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "gpu-stress-test: CUBLAS error at %s:%d.\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }
#define CHECK_NVML_ERROR(err) \
  if ((err) != NVML_SUCCESS) { \
    fprintf(stderr, "gpu-stress-test: NVML error %s at %s:%d.\n", nvmlErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }

void init_test_context(struct test_context* ctx, unsigned gpu_index = 0) {
  int cuda_gpu_count;
  unsigned nvml_gpu_count;
  const size_t matrix_n = 16384;
  float* matrix = NULL;
  size_t i;
  cudaError_t cuda_err;
  cublasStatus_t cublas_err;
  nvmlReturn_t nvml_err;

  nvml_err = nvmlInit();
  CHECK_NVML_ERROR(nvml_err);

  cuda_err = cudaGetDeviceCount(&cuda_gpu_count);
  CHECK_CUDA_ERROR(cuda_err);
  nvml_err = nvmlDeviceGetCount(&nvml_gpu_count);
  CHECK_NVML_ERROR(nvml_err);
  if (cuda_gpu_count != nvml_gpu_count) {
    fprintf(stderr, "gpu-stress-test: different numbers of GPUs are detected in CUDA and NVML.\n");
    exit(EXIT_FAILURE);
  }

  if (gpu_index >= cuda_gpu_count) {
    fprintf(stderr, "gpu-stress-test: cannot use GPU #%u because only %d GPUs are found.\n",
            gpu_index, cuda_gpu_count);
    exit(EXIT_FAILURE);
  }

  cuda_err = cudaSetDevice(gpu_index);
  CHECK_CUDA_ERROR(cuda_err);
  nvml_err = nvmlDeviceGetHandleByIndex(gpu_index, &ctx->dev);
  CHECK_NVML_ERROR(nvml_err);

  cublas_err = cublasCreate(&ctx->h);
  CHECK_CUBLAS_ERROR(cublas_err);

  matrix = (float*)malloc(matrix_n * matrix_n * sizeof(float));
  if (matrix == NULL) {
    fprintf(stderr, "gpu-stress-test: host memory allocation failed.\n");
    exit(EXIT_FAILURE);
  }
  for (i = 0; i < matrix_n * matrix_n; ++i) {
    matrix[i] = (float)(i % 100) / 10.0f;
  }

  ctx->n = matrix_n;
  cuda_err = cudaMalloc((void**)&ctx->a, matrix_n * matrix_n * sizeof(float));
  CHECK_CUDA_ERROR(cuda_err);
  cuda_err = cudaMalloc((void**)&ctx->b, matrix_n * matrix_n * sizeof(float));
  CHECK_CUDA_ERROR(cuda_err);
  cuda_err = cudaMalloc((void**)&ctx->c, matrix_n * matrix_n * sizeof(float));
  CHECK_CUDA_ERROR(cuda_err);

  cuda_err = cudaMemcpy(ctx->a, matrix, matrix_n * matrix_n * sizeof(float), cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR(cuda_err);
  cuda_err = cudaMemcpy(ctx->b, matrix, matrix_n * matrix_n * sizeof(float), cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR(cuda_err);

  free(matrix);

  // Dummy run
  {
    float alpha = 1.0;
    float beta = 0.0;
    cublas_err = cublasSgemm(ctx->h, CUBLAS_OP_N, CUBLAS_OP_T, ctx->n, ctx->n, ctx->n,
                             &alpha, ctx->a, ctx->n, ctx->b, ctx->n, &beta, ctx->c, ctx->n);
    CHECK_CUBLAS_ERROR(cublas_err);
    cuda_err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cuda_err);
  }

  // Dummy measurement
  {
    unsigned temperature;
    nvml_err = nvmlDeviceGetTemperature(ctx->dev, NVML_TEMPERATURE_GPU, &temperature);
    CHECK_NVML_ERROR(nvml_err);
  }

  printf("Ready to test GPU #%u.\n", gpu_index);
}

void run_stress(const struct test_context* ctx, unsigned run) {
  float alpha = 1.0;
  float beta = 0.0;
  unsigned trial;
  double t_start, t_end;
  unsigned temperature;
  cudaError_t cuda_err;
  cublasStatus_t cublas_err;
  nvmlReturn_t nvml_err;

  printf("Run %u: ... ", run); fflush(stdout);

  t_start = get_time();
  for (trial = 0; trial < 5; ++trial) {
    cublas_err = cublasSgemm(ctx->h, CUBLAS_OP_N, CUBLAS_OP_T, ctx->n, ctx->n, ctx->n,
                             &alpha, ctx->a, ctx->n, ctx->b, ctx->n, &beta, ctx->c, ctx->n);
    CHECK_CUBLAS_ERROR(cublas_err);
  }
  cuda_err = cudaDeviceSynchronize();
  CHECK_CUDA_ERROR(cuda_err);
  t_end = get_time();

  nvml_err = nvmlDeviceGetTemperature(ctx->dev, NVML_TEMPERATURE_GPU, &temperature);
  CHECK_NVML_ERROR(nvml_err);

  printf("%.2lf GFLOPS, %u degress Celsius\n",
         ((double)ctx->n * ctx->n * (ctx->n * 2 - 1) * trial) /
         (t_end - t_start) / 1e9,
         temperature);
}

void destroy_test_context(struct test_context* ctx) {
  cudaFree(ctx->a);
  cudaFree(ctx->b);
  cudaFree(ctx->c);
  cublasDestroy(ctx->h);
  nvmlShutdown();
}

int main(int argc, char* argv[]) {
  struct test_context ctx;
  unsigned run;

  printf("Preparing the stress test...\n");
  if (argc > 1) {
    init_test_context(&ctx, atoi(argv[1]));
  } else {
    init_test_context(&ctx, 0);
  }
  for (run = 1; run <= 1000; ++run) {
    run_stress(&ctx, run);
  }
  printf("Stress test completed.\n");
  destroy_test_context(&ctx);
  return 0;
}
