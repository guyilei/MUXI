#define CL_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.h>
#include <iostream>

// GPU 核函数：一个 block 有 2×2=4 个线程。
const char* gpu_code = R"(
__kernel void matrix_add(__global const float* A,
                         __global const float* B,
                         __global float* C) {
    int col = get_local_id(0);  // 当前线程在 block 内的列号
    int row = get_local_id(1);  // 当前线程在 block 内的行号
    int i = row * 2 + col;      // 二维位置转换为一维数组位置

    C[i] = A[i] + B[i];         // 每个线程计算一个元素
}
)";

int main() {
    // 两个具体的 2×2 矩阵。
    // A = [1  2]    B = [10  20]
    //     [3  4]        [30  40]
    float A[4] = {1, 2, 3, 4};
    float B[4] = {10, 20, 30, 40};
    float C[4] = {0, 0, 0, 0};

    cl_int error;
    const size_t bytes = sizeof(A);

    // 1. 找到 Mac GPU，并创建运行环境。
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id gpu;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu, nullptr);

    cl_context context =
        clCreateContext(nullptr, 1, &gpu, nullptr, nullptr, &error);
    cl_command_queue queue =
        clCreateCommandQueue(context, gpu, 0, &error);

    // 2. 把字符串中的 kernel 代码交给 OpenCL 编译。
    cl_program program =
        clCreateProgramWithSource(context, 1, &gpu_code, nullptr, &error);
    clBuildProgram(program, 1, &gpu, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "matrix_add", &error);

    // 3. 在 GPU 上为 A、B、C 创建内存。
    // A、B 创建时会把 CPU 数组中的具体数字复制到 GPU。
    cl_mem gpu_A = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A, &error);
    cl_mem gpu_B = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B, &error);
    cl_mem gpu_C = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error);

    // 4. 按照 matrix_add(A, B, C) 的顺序传入三个参数。
    clSetKernelArg(kernel, 0, sizeof(gpu_A), &gpu_A);
    clSetKernelArg(kernel, 1, sizeof(gpu_B), &gpu_B);
    clSetKernelArg(kernel, 2, sizeof(gpu_C), &gpu_C);

    // 5. 总线程布局是 2×2，每个 block 也是 2×2。
    // block 数量 = (2÷2) × (2÷2) = 1。
    const size_t total_threads[2] = {2, 2};
    const size_t one_block[2] = {2, 2};

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                           total_threads, one_block,
                           0, nullptr, nullptr);

    // 6. 把 GPU 中的结果 gpu_C 复制回 CPU 数组 C。
    clEnqueueReadBuffer(queue, gpu_C, CL_TRUE, 0, bytes, C,
                        0, nullptr, nullptr);

    // 7. 直接显示四个线程计算出的结果。
    std::cout << "block(0,0)，thread(0,0)：1 + 10 = " << C[0] << '\n';
    std::cout << "block(0,0)，thread(0,1)：2 + 20 = " << C[1] << '\n';
    std::cout << "block(0,0)，thread(1,0)：3 + 30 = " << C[2] << '\n';
    std::cout << "block(0,0)，thread(1,1)：4 + 40 = " << C[3] << '\n';

    std::cout << "结果矩阵 C：\n";
    std::cout << C[0] << "  " << C[1] << '\n';
    std::cout << C[2] << "  " << C[3] << '\n';

    // 8. 释放用完的 GPU 资源。
    clReleaseMemObject(gpu_A);
    clReleaseMemObject(gpu_B);
    clReleaseMemObject(gpu_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
