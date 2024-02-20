#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>

const int layer_dims[] = { 3, 64, 64, 64, 4 };
const int n_layers = 5;

float* weights;
float* bias;
int num_weights, num_bias;
const int batch_size = 2;
int num_activations = 0;
int output_offset = 0;

float as_cpu[] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 };
float* as;
float* zs;

cublasHandle_t cu_handle;
const float alpha = 1.0f;
const float beta = 0.0f;

const char* weights_path = "../weights.bin";
const char* bias_path = "../bias.bin";

void MallocWeights()
{
    num_weights = 0;
    num_bias = 0;

    for (int i = 1; i < n_layers; i++)
    {
        num_weights += layer_dims[i - 1] * layer_dims[i];
        num_bias += layer_dims[i];
    }

    cudaMalloc(&weights, num_weights * sizeof(float));
    cudaMalloc(&bias, num_bias * sizeof(float));
}

void InitData()
{
    for (int i = 0; i < n_layers; i++)
    {
        num_activations += batch_size * layer_dims[i];
        if (i != n_layers - 1)
		{
			output_offset += batch_size * layer_dims[i];
		}
    }

    cudaMalloc(&zs, num_activations * sizeof(float));
    cudaMalloc(&as, num_activations * sizeof(float));

    cudaMemcpy(as, &as_cpu, layer_dims[0] * batch_size * sizeof(float), cudaMemcpyHostToDevice);
}

// __device__ cannot be called with <<< >>>
__device__ float Activation(const float a)
{
    return std::fmaxf(a, 0.f);
}

// __global__ can be called with <<< >>>
__global__ void BiasActivation(float* input, float* output, float* bias)
{
    const int offset = blockDim.x * blockIdx.x + threadIdx.x;
    *(input + offset) += *(bias + threadIdx.x);
    *(output + offset) = Activation(*(input + offset));
}

__global__ void BiasWithoutActivation(float* input, float* output, float* bias)
{
	const int offset = blockDim.x * blockIdx.x + threadIdx.x;
	*(output + offset) = *(input + offset) + *(bias + threadIdx.x);
}

void Forward()
{
    int weight_offset = 0;
    int bias_offset = 0;
    int as_offset = 0;
    int zs_offset = batch_size * layer_dims[0];

    for (int i = 1; i < n_layers; i++)
    {
        cublasSgemm(
            cu_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            layer_dims[i], batch_size, layer_dims[i - 1],
            &alpha,
            weights + weight_offset, layer_dims[i],
            as + as_offset, layer_dims[i - 1],
            &beta,
            zs + zs_offset, layer_dims[i]
        );

        as_offset += batch_size * layer_dims[i - 1];

        dim3 grid(batch_size);
        dim3 block(layer_dims[i]);

        if (i == n_layers - 1)
        {
            BiasWithoutActivation <<< grid, block >>> (zs + zs_offset, as + as_offset, bias + bias_offset);
		}
        else
        {
            BiasActivation <<< grid, block >>> (zs + zs_offset, as + as_offset, bias + bias_offset);
        }

        cudaDeviceSynchronize();

        weight_offset += layer_dims[i - 1] * layer_dims[i];
        bias_offset += layer_dims[i];
        zs_offset += batch_size * layer_dims[i];
    }

    cudaDeviceSynchronize();
}

void ReadWeights(const char* weights_path, const char* bias_path)
{
    std::ifstream wf(weights_path, std::ios::binary);
    float* weight_data = new float[num_weights];
    wf.read((char*)weight_data, num_weights * sizeof(float));
    wf.close();

    std::ifstream bf(bias_path, std::ios::binary);
    float* bias_data = new float[num_bias];
    bf.read((char*)bias_data, num_bias * sizeof(float));
    bf.close();

    CudaSafeCall(cudaMemcpy(weights, weight_data, num_weights * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(bias, bias_data, num_bias * sizeof(float), cudaMemcpyHostToDevice));
}

int main() {
    cublasCreate(&cu_handle);

    MallocWeights();
    InitData();

    ReadWeights(weights_path, bias_path);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    for (int i = 0; i < 1; i++)
    {
        begin = std::chrono::steady_clock::now();
        Forward();
        end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.f << " ms" << std::endl;
    }

    float* out_host = new float[layer_dims[n_layers - 1] * batch_size];
    cudaMemcpy(out_host, as + output_offset, layer_dims[n_layers - 1] * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int j = 0; j < layer_dims[n_layers - 1] * batch_size; j++)
    {
        std::cout << out_host[j] << " ";
    }

    return 0;
}
