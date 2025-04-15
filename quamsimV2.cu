#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void global_matrix_multiply(float *input, float *output, float *Umatrix, int n, int qbitStride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = qbitStride;

    if (i >= n / 2) return;

    int base = (i / step) * (2 * step) + (i % step);
    int pair = base + step;

    float a = Umatrix[0];
    float b = Umatrix[1];
    float c = Umatrix[2];
    float d = Umatrix[3];

    float x = input[base];
    float y = input[pair];

    output[base] = a * x + b * y;
    output[pair] = c * x + d * y;
}

__global__ void shared_matrix_multiply(float *d_A, float *d_B, float *d_U, int *d_indices, int *d_TB_indices, int *d_qbit_indices) {
    __shared__ float s[64];
    int localA = d_indices[2 * threadIdx.x];
    int localB = d_indices[2 * threadIdx.x + 1];
    int offset = d_TB_indices[blockIdx.x];

    s[2 * threadIdx.x]     = d_A[localA + offset];
    s[2 * threadIdx.x + 1] = d_A[localB + offset];
    __syncthreads();

    for (int k = 0; k < 6; ++k) {
        int stride = d_qbit_indices[k];
        int idx_a = 2 * threadIdx.x - (threadIdx.x % stride);
        int idx_b = idx_a + stride;

        float a = d_U[4 * k + 0];
        float b = d_U[4 * k + 1];
        float c = d_U[4 * k + 2];
        float d = d_U[4 * k + 3];

        float val_a = s[idx_a];
        float val_b = s[idx_b];

        s[idx_a] = a * val_a + b * val_b;
        s[idx_b] = c * val_a + d * val_b;
        __syncthreads();
    }

    d_B[localA + offset] = s[2 * threadIdx.x];
    d_B[localB + offset] = s[2 * threadIdx.x + 1];
}

void apply_global(float *h_A, float *h_B, float *matrices[6], int qubits[6], int n) {
    float *d_A, *d_B, *d_U;
    size_t vec_size = n * sizeof(float), mat_size = 4 * sizeof(float);

    cudaMalloc(&d_A, vec_size);
    cudaMalloc(&d_B, vec_size);
    cudaMalloc(&d_U, mat_size);

    cudaMemcpy(d_A, h_A, vec_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;

    for (int i = 0; i < 6; ++i) {
        float *U = matrices[i];
        cudaMemcpy(d_U, U, mat_size, cudaMemcpyHostToDevice);
        int stride = 1 << qubits[i];

        global_matrix_multiply<<<blocks, threads>>>(d_A, d_B, d_U, n, stride);
        cudaMemcpy(d_A, d_B, vec_size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_B, d_B, vec_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_U);
}

bool all_qubits_under_6(int qubits[6]) {
    for (int i = 0; i < 6; i++) {
        if (qubits[i] > 5) return false;
    }
    return true;
}

void setBit(int &num, int pos, int val) {
    if (val)
        num |= (1 << pos);
    else
        num &= ~(1 << pos);
}

void run_shared(float *h_A, float *h_B, float *matrices[6], int qubits[6], int n) {
    int h_indices[64];
    for (int i = 0; i < 64; ++i) {
        h_indices[i] = 0;
        for (int q = 0; q < 6; ++q) {
            setBit(h_indices[i], qubits[q], (i >> q) & 1);
        }
    }

    int blocks = n / 64;
    int h_TB_indices[blocks];
    for (int i = 0; i < blocks; ++i)
        h_TB_indices[i] = i << 6;

    int h_qbit_indices[6];
    for (int i = 0; i < 6; ++i)
        h_qbit_indices[i] = 1 << qubits[i];

    float *d_A, *d_B, *d_U;
    int *d_indices, *d_TB_indices, *d_qbit_indices;

    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_U, 24 * sizeof(float));
    cudaMalloc(&d_indices, sizeof(h_indices));
    cudaMalloc(&d_TB_indices, sizeof(h_TB_indices));
    cudaMalloc(&d_qbit_indices, sizeof(h_qbit_indices));

    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 6; ++i)
        cudaMemcpy(d_U + i * 4, matrices[i], 4 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_indices, h_indices, sizeof(h_indices), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TB_indices, h_TB_indices, sizeof(h_TB_indices), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qbit_indices, h_qbit_indices, sizeof(h_qbit_indices), cudaMemcpyHostToDevice);

    shared_matrix_multiply<<<blocks, 32>>>(d_A, d_B, d_U, d_indices, d_TB_indices, d_qbit_indices);
    cudaMemcpy(h_B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_U);
    cudaFree(d_indices); cudaFree(d_TB_indices); cudaFree(d_qbit_indices);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: ./quamsim input.txt" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    if (!file) {
        cerr << "File error." << endl;
        return 1;
    }

    // Read 6 matrices
    float *matrices[6];
    for (int i = 0; i < 6; ++i) {
        matrices[i] = new float[4];
        for (int j = 0; j < 4; ++j) {
            file >> matrices[i][j];
        }
    }

    // Read vector and qubit indices
    vector<float> raw;
    float x;
    while (file >> x) raw.push_back(x);

    int qubits[6];
    for (int i = 5; i >= 0; --i) {
        qubits[i] = (int)raw.back();
        raw.pop_back();
    }

    int n = raw.size();
    float *h_A = new float[n];
    float *h_B = new float[n];
    for (int i = 0; i < n; i++) h_A[i] = raw[i];

    // Decide which version to run
    if (all_qubits_under_6(qubits)) {
        run_shared(h_A, h_B, matrices, qubits, n);
    } else {
        apply_global(h_A, h_B, matrices, qubits, n);
    }

    for (int i = 0; i < n; i++) {
        printf("%.3f\n", h_B[i]);
    }

    delete[] h_A; delete[] h_B;
    for (int i = 0; i < 6; i++) delete[] matrices[i];

    return 0;
}