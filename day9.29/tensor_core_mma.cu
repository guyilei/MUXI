#include <cuda_fp16.h>   // half、__float2half、__half2float
#include <cuda_runtime.h> // cudaMalloc、cudaMemcpy、Kernel 启动等 CUDA 运行时 API
#include <mma.h>          // NVIDIA WMMA API，用于调用 Tensor Core

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace wmma = nvcuda::wmma;

// 四个矩阵都是 16×16。
// 采用行主序时，矩阵第 row 行、第 col 列的位置是 row * N + col。
constexpr int N = 16;
constexpr int SIZE = N * N;

// 检查 CUDA API 是否执行成功。
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::cerr << cudaGetErrorString(error) << std::endl;             \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// 一个 Warp（32 个线程）共同计算 D = A × B + C。
// const 表示 Kernel 只读取 A、B、C；D 没有 const，因为要向其中写结果。
// 指针只保存矩阵第一个元素的地址，后续元素连续存放在显存中。
__global__ void tensorCoreKernel(const half* A,
                                 const half* B,
                                 const float* C,
                                 float* D) {
    // fragment 是由整个 Warp 共同持有的小矩阵。
    // fragment<用途, M, N, K, 元素类型, 内存布局>
    // matrix_a：乘法左侧的 A；matrix_b：乘法右侧的 B。
    // A、B 使用 FP16（half），累加器使用 FP32（float）。
    wmma::fragment<wmma::matrix_a, 16, 16, 16,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16,
                   half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

    // 从显存加载三个 16×16 矩阵。
    // 最后的 N 是 leading dimension：行主序矩阵每行有 16 个元素。
    wmma::load_matrix_sync(a_frag, A, N);
    wmma::load_matrix_sync(b_frag, B, N);
    wmma::load_matrix_sync(c_frag, C, N, wmma::mem_row_major);

    // 核心：使用 Tensor Core 计算 d_frag = a_frag × b_frag + c_frag。
    // 这一行由整个 Warp 同步执行，不能只让部分线程执行。
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

    // 将结果写回显存，按行存储。
    wmma::store_matrix_sync(D, d_frag, N, wmma::mem_row_major);
}

// 普通 CPU 矩阵乘加，用于检查 GPU 结果。
void cpuReference(const half* A, const half* B, const float* C, float* D) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            // 先取 C[row][col]，再计算 A 的这一行和 B 的这一列的点积。
            float sum = C[row * N + col];
            for (int k = 0; k < N; ++k) {
                // A、B 已经是 half。先转回 float，再在 CPU 上计算参考结果。
                sum += __half2float(A[row * N + k]) *
                       __half2float(B[k * N + col]);
            }
            D[row * N + col] = sum;
        }
    }
}

int main() {
    // 这些数组位于 CPU 内存中。
    // A、B 是 Tensor Core 的 FP16 输入；C、D 是 FP32。
    half A[SIZE];
    half B[SIZE];
    float C[SIZE];
    float D_gpu[SIZE];
    float D_cpu[SIZE];

    // 使用简单的小数初始化矩阵。
    // __float2half 把普通 float 转换成 FP16。
    for (int i = 0; i < SIZE; ++i) {
        A[i] = __float2half((i % 5) * 0.25f);
        B[i] = __float2half((i % 7) * 0.125f);
        C[i] = 1.0f;
    }

    // 名字中的 d_ 表示 device：这些指针将指向 GPU 显存。
    float *d_C, *d_D;
    half *d_A, *d_B;

    // cudaMalloc 在 GPU 显存中申请空间。
    // 传入指针的地址，是为了让 CUDA 把新显存的地址写入 d_A 等变量。
    CHECK_CUDA(cudaMalloc(&d_A, SIZE * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, SIZE * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_D, SIZE * sizeof(float)));

    // HostToDevice：把 CPU 数组复制到 GPU 显存。
    CHECK_CUDA(cudaMemcpy(d_A, A, SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, C, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // 1 个线程块、32 个线程，也就是 1 个 Warp。
    // <<<1, 32>>> 中的 1 是线程块数量，32 是每个线程块的线程数。
    tensorCoreKernel<<<1, 32>>>(d_A, d_B, d_C, d_D);
    // 检查启动错误，并等待 GPU 完成计算。
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // DeviceToHost：把 GPU 结果复制回 CPU 数组 D_gpu。
    CHECK_CUDA(cudaMemcpy(D_gpu, d_D, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    cpuReference(A, B, C, D_cpu);

    // 比较全部 256 个元素，找出最大的绝对误差。
    float max_error = 0.0f;
    for (int i = 0; i < SIZE; ++i) {
        float error = std::fabs(D_gpu[i] - D_cpu[i]);
        if (error > max_error) max_error = error;
    }

    std::cout << "CPU result: " << D_cpu[0] << '\n';
    std::cout << "GPU result: " << D_gpu[0] << '\n';
    std::cout << "Max error: " << max_error << '\n';
    std::cout << (max_error < 1e-2f ? "PASSED" : "FAILED") << '\n';

    // 用完后释放 GPU 显存。
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_D));

    return max_error < 1e-2f ? 0 : 1;
}
