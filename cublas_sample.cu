#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include <cstdint>

#define M 2
#define N 4
#define K 3

void printMatrix(float(*matrix)[N], int row, int col) {
    for (int i = 0; i < row; i++)
    {
        std::cout << std::endl;
        std::cout << " [ ";
        for (int j = 0; j < col; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << " ] ";
    }
    std::cout << std::endl;
}

int main(void)
{
    float alpha = 1.0;
    float beta = 0.0;
    float h_A[M][K] = { {1,2,3}, {4,5,6} };
    float h_B[K][N] = { {1,2,3,4}, {5,6,7,8}, {9,10,11,12} };
    float h_C[M][N] = { 0 };
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, &h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, M * N * sizeof(float));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    cudaMemcpy(h_C, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);//此处的h_C是按列存储的CT
    printMatrix(h_C, M, N);//按行读取h_C相当于做了CTT=C的结果
    return 0;
}