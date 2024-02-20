# CUBLAS_MLP

This repository utilizes CUDA cuBLAS to implement a Multilayer Perceptron (MLP) for rapid network inference. The purpose of this repository is to:

1. Train an MLP using Python and save the resulting weights and biases as binary files.
2. Employ CUDA cuBLAS to perform inference using the previously saved weights and biases.

## Usage

1. Run `train.py` and we get the saved `weights.bin` and `bias.bin`.
2. Specify the paths of weights and bias in `cublas_mlp.cu` and run this script.
3. Finally, we can see that the python MLP and cuda MLP get the same output.