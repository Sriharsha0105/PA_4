#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel: Applies a single-qubit gate to a given qubit across the state vector
__global__ void matrix_multiply(const float *input, float *output, const float *Umatrix, int size, int qbit) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int mask = 1 << qbit;
    int index = i ^ mask;  // Flips the qbit-th bit to find pair

    if (i < size && (i / mask) % 2 == 0) {
        // Only compute each (i, index) pair once
        output[i]     = Umatrix[0] * input[i] + Umatrix[1] * input[index];
        output[index] = Umatrix[2] * input[i] + Umatrix[3] * input[index];
    }
}

// Helper function: Read a 2x2 gate matrix from file
void read_matrix(ifstream &file, float* matrix) {
    for (int i = 0; i < 4; i++) {
        file >> matrix[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " input_file.txt" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    if (!file.is_open()) {
        cerr << "Error opening file." << endl;
        return 1;
    }

    // Read six 2x2 gate matrices
    float matrices[6][4];
    for (int i = 0; i < 6; i++) {
        read_matrix(file, matrices[i]);
    }

    // Read state vector and qubit indices
    vector<float> values;
    float val;
    while (file >> val) {
        values.push_back(val);
    }

    // Extract qubit indices from the last six elements
    int q[6];
    for (int i = 5; i >= 0; i--) {
        q[i] = static_cast<int>(values.back());
        values.pop_back();
    }

    int n = values.size();               // 2^num_qubits
    size_t vec_size = n * sizeof(float);
    size_t gate_size = 4 * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(vec_size);
    float *h_B = (float *)malloc(vec_size);
    float *h_U = (float *)malloc(gate_size);

    // Copy initial vector
    for (int i = 0; i < n; ++i) {
        h_A[i] = values[i];
    }

    // Allocate device memory
    float *d_A, *d_B, *d_U;
    cudaMalloc(&d_A, vec_size);
    cudaMalloc(&d_B, vec_size);
    cudaMalloc(&d_U, gate_size);

    cudaMemcpy(d_A, h_A, vec_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Apply six gates sequentially
    for (int step = 0; step < 6; ++step) {
        // Load gate matrix to device
        for (int i = 0; i < 4; i++) h_U[i] = matrices[step][i];
        cudaMemcpy(d_U, h_U, gate_size, cudaMemcpyHostToDevice);

        // Launch kernel
        matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_U, n, q[step]);
        cudaDeviceSynchronize();

        // Prepare input for next step
        cudaMemcpy(h_A, d_B, vec_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_A, h_A, vec_size, cudaMemcpyHostToDevice);
    }

    // Copy result back to host and print
    cudaMemcpy(h_B, d_B, vec_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        printf("%.3f\n", h_B[i]);
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_U);
    free(h_A); free(h_B); free(h_U);
    cudaDeviceReset();

    return 0;
}