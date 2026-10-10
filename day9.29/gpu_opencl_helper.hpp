#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.h>
#include <cstdlib>
#include <iostream>

inline void checkGpu(cl_int error, const char* step) {
    if (error != CL_SUCCESS) {
        std::cerr << step << "失败，错误码 " << error << '\n';
        std::exit(1);
    }
}

// 连接 GPU、传入矩阵、运行一个 2×2 线程组、取回结果。
inline void runOnGpu(const char* gpu_code, float* A, float* B, float* C) {
    cl_int error;
    cl_platform_id platform;
    checkGpu(clGetPlatformIDs(1, &platform, nullptr), "查找平台");
    cl_device_id gpu;
    checkGpu(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu, nullptr), "查找 GPU");
    cl_context context = clCreateContext(nullptr, 1, &gpu, nullptr, nullptr, &error);
    checkGpu(error, "创建运行环境");
    cl_command_queue queue = clCreateCommandQueue(context, gpu, 0, &error);
    checkGpu(error, "创建命令队列");

    cl_program program = clCreateProgramWithSource(context, 1, &gpu_code, nullptr, &error);
    checkGpu(error, "创建 GPU 程序");
    checkGpu(clBuildProgram(program, 1, &gpu, nullptr, nullptr, nullptr), "编译 GPU 程序");
    cl_kernel kernel = clCreateKernel(program, "matrix_add", &error);
    checkGpu(error, "找到核函数");

    const size_t bytes = 4 * sizeof(float);
    cl_mem gpu_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   bytes, A, &error);
    checkGpu(error, "上传 A");
    cl_mem gpu_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   bytes, B, &error);
    checkGpu(error, "上传 B");
    cl_mem gpu_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error);
    checkGpu(error, "分配 C");
    checkGpu(clSetKernelArg(kernel, 0, sizeof(gpu_A), &gpu_A), "设置 A");
    checkGpu(clSetKernelArg(kernel, 1, sizeof(gpu_B), &gpu_B), "设置 B");
    checkGpu(clSetKernelArg(kernel, 2, sizeof(gpu_C), &gpu_C), "设置 C");

    const size_t total[2] = {2, 2};
    const size_t one_block[2] = {2, 2};
    checkGpu(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, total, one_block,
                                    0, nullptr, nullptr), "运行核函数");
    checkGpu(clEnqueueReadBuffer(queue, gpu_C, CL_TRUE, 0, bytes, C,
                                 0, nullptr, nullptr), "取回结果");

    clReleaseMemObject(gpu_A);
    clReleaseMemObject(gpu_B);
    clReleaseMemObject(gpu_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
