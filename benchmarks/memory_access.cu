#include <cuda.h>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(e) do { if ((e) != cudaSuccess) { printf("CUDA_CHECK failed at %s:%d %s\n", __FILE__, __LINE__, #e); exit(1); } } while(0)

__host__ __device__ void* my_calloc(size_t nmemb, size_t size) {
    void* result = malloc(nmemb * size);
    memset(result, 0, nmemb * size);
    return result;
}

__host__ __device__ float*** allocate_tensor(int nx, int ny, int nz) {
    float* a1 = (float*) my_calloc(nx * ny * nz, sizeof(float));
    float** a2 = (float**) my_calloc(nx * ny, sizeof(float*));
    float*** a3 = (float***) my_calloc(nx, sizeof(float**));

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            a2[i * ny + j] = &a1[(i * ny + j) * nz];
        }
        a3[i] = &a2[i * ny];
    }

    return a3;
}

__host__ __device__ void deallocate_tensor(float*** a) {
    free(**a);
    free(*a);
    free(a);
}

__host__ __device__ int idx(int i, int j, int k, int nx, int ny, int nz) {
    return i * ny * nz + j * nz + k;
}

void stencil_cpu_1(int nx, int ny, int nz, int nt) {
    std::cout << "+++++ stencil_cpu_1 +++++\n";

    auto t1 = std::chrono::steady_clock::now();
    float* a = (float*)calloc(nx * ny * nz, sizeof(float));
    
    auto t2 = std::chrono::steady_clock::now();

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Allocation " << milliseconds.count() << " ms\n";

    for (int t = 0; t < nt; t++) {
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                for (int k = 1; k < nz - 1; k++) {
                    a[idx(i, j, k, nx, ny, nz)] = (1./7.) * (
                        a[idx(i    , j    , k    , nx, ny, nz)] +
                        a[idx(i + 1, j    , k    , nx, ny, nz)] +
                        a[idx(i - 1, j    , k    , nx, ny, nz)] +
                        a[idx(i    , j + 1, k    , nx, ny, nz)] +
                        a[idx(i    , j - 1, k    , nx, ny, nz)] +
                        a[idx(i    , j    , k + 1, nx, ny, nz)] +
                        a[idx(i    , j    , k - 1, nx, ny, nz)]);
                }
            }
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Computation " << milliseconds.count() << " ms\n";

    free(a);   
}

void stencil_cpu_2(int nx, int ny, int nz, int nt) {
    std::cout << "+++++ stencil_cpu_2 +++++\n";

    auto t1 = std::chrono::steady_clock::now();
    float*** a = allocate_tensor(nx, ny, nz);

    auto t2 = std::chrono::steady_clock::now();

    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Allocation " << milliseconds.count() << " ms\n";


    for (int t = 0; t < nt; t++) {
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                for (int k = 1; k < nz - 1; k++) {
                    a[i][j][k] = (1./7.) * (
                        a[i    ][j    ][k    ] +
                        a[i + 1][j    ][k    ] +
                        a[i - 1][j    ][k    ] +
                        a[i    ][j + 1][k    ] +
                        a[i    ][j - 1][k    ] +
                        a[i    ][j    ][k + 1] +
                        a[i    ][j    ][k - 1]);
                }
            }
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cout << "Computation " << milliseconds.count() << " ms\n";

    deallocate_tensor(a);
}

__global__ void stencil_gpu_1(int nx, int ny, int nz, int nt, int gpu_freq) {
    printf("+++++ stencil_gpu_1 +++++\n");

    long long t1 = clock64();
    float* a = (float*)my_calloc(nx * ny * nz, sizeof(float));
    
    long long t2 = clock64();

    long long milliseconds = (t2 - t1) / gpu_freq;
    printf("Allocation %lld ms\n", milliseconds);

    for (int t = 0; t < nt; t++) {
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                for (int k = 1; k < nz - 1; k++) {
                    a[idx(i, j, k, nx, ny, nz)] = (1./7.) * (
                        a[idx(i    , j    , k    , nx, ny, nz)] +
                        a[idx(i + 1, j    , k    , nx, ny, nz)] +
                        a[idx(i - 1, j    , k    , nx, ny, nz)] +
                        a[idx(i    , j + 1, k    , nx, ny, nz)] +
                        a[idx(i    , j - 1, k    , nx, ny, nz)] +
                        a[idx(i    , j    , k + 1, nx, ny, nz)] +
                        a[idx(i    , j    , k - 1, nx, ny, nz)]);
                }
            }
        }
    }
    long long t3 = clock64();
    
    milliseconds = (t3 - t2) / gpu_freq;
    printf("Computation %lld ms\n", milliseconds);

    free(a);   
}

__global__ void stencil_gpu_2(int nx, int ny, int nz, int nt, int gpu_freq) {
    printf("+++++ stencil_gpu_2 +++++\n");

    auto t1 = clock64();
    float*** a = allocate_tensor(nx, ny, nz);

    auto t2 = clock64();

    auto milliseconds = (t2 - t1) / gpu_freq;
    printf("Allocation %lld ms\n", milliseconds);


    for (int t = 0; t < nt; t++) {
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                for (int k = 1; k < nz - 1; k++) {
                    a[i][j][k] = (1./7.) * (
                        a[i    ][j    ][k    ] +
                        a[i + 1][j    ][k    ] +
                        a[i - 1][j    ][k    ] +
                        a[i    ][j + 1][k    ] +
                        a[i    ][j - 1][k    ] +
                        a[i    ][j    ][k + 1] +
                        a[i    ][j    ][k - 1]);
                }
            }
        }
    }

    auto t3 = clock64();
    
    milliseconds = (t3 - t2) / gpu_freq;
    printf("Computation %lld ms\n", milliseconds);

    deallocate_tensor(a);
}

int main() {
    int nx = 100;
    int ny = 100;
    int nz = 100;
    int nt = 10;

    stencil_cpu_1(nx, ny, nz, nt);
    stencil_cpu_2(nx, ny, nz, nt);

    int gpu_freq = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&gpu_freq, cudaDevAttrClockRate, 0));

    stencil_gpu_1<<<1,1>>>(nx, ny, nz, nt, gpu_freq);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    stencil_gpu_2<<<1,1>>>(nx, ny, nz, nt, gpu_freq);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
