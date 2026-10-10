#define CL_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.h>
#include <cstdlib>
#include <iostream>

// GPU 核函数：4×4 矩阵被切成 4 个 block。
// 每个 block 覆盖 2×2 个矩阵元素，内部有 2×2=4 个线程。
const char* gpu_code = R"(
__kernel void matrix_add(__global const float* A,
                         __global const float* B,
                         __global float* C) {
    // 当前是第几个 block。
    int block_col = get_group_id(0);
    int block_row = get_group_id(1);

    // 当前线程在 block 内的位置。
    int thread_col = get_local_id(0);
    int thread_row = get_local_id(1);

    // block 位置 + 组内线程位置 = 矩阵中的位置。
    int col = block_col * 2 + thread_col;
    int row = block_row * 2 + thread_row;

    // 4×4 矩阵按行存储，把二维位置换成数组位置。
    int i = row * 4 + col;

    // 每个线程只负责一个矩阵元素。
    C[i] = A[i] + B[i];
}
)";

void check(cl_int error, const char* step) {
    // if (error != CL_SUCCESS) {
    //     std::cerr << step << "失败，错误码：" << error << '\n';
    //     std::exit(1);
    // }
}

int main() {
    // 两个具体的 4×4 矩阵，内存中按行排列。
    float A[16] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    float B[16] = {
         10,  20,  30,  40,
         50,  60,  70,  80,
         90, 100, 110, 120,
        130, 140, 150, 160
    };

    float C[16] = {};
    const size_t bytes = sizeof(A);
    cl_int error;

    //==================== 1. 连接 Mac GPU ====================

    // platform 用来保存 OpenCL 平台。
    // 可以把“平台”理解为 OpenCL 驱动提供的入口。
    cl_platform_id platform;

    // 获取 1 个 OpenCL 平台：
    // 参数1：最多获取 1 个平台；
    // 参数2：把找到的平台保存到 platform；
    // 参数3：不需要返回平台总数，所以传 nullptr。
    clGetPlatformIDs(1, &platform, nullptr);

    // gpu 用来保存我们选中的 GPU 设备。
    cl_device_id gpu;

    // 从 platform 中获取 1 个 GPU：
    // CL_DEVICE_TYPE_GPU 表示只查找 GPU，不选择 CPU；
    // 找到的设备保存到 gpu。
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &gpu, nullptr);

    // 创建 OpenCL 上下文 context。
    // context 是管理 GPU、GPU 内存和 GPU 程序的运行环境。
    // 这里把刚才找到的 gpu 加入这个运行环境。
    cl_context context =
        clCreateContext(nullptr, 1, &gpu, nullptr, nullptr, &error);

    // 创建命令队列 queue。
    // CPU 通过这个队列依次向 GPU 提交命令，例如运行 kernel、复制数据。
    cl_command_queue queue =
        clCreateCommandQueue(context, gpu, 0, &error);

    //==================== 2. 编译 GPU 核函数 ====================

    // gpu_code 目前只是一段字符串。
    // clCreateProgramWithSource 把这段字符串包装成 OpenCL 程序 program。
    cl_program program =
        clCreateProgramWithSource(context, 1, &gpu_code, nullptr, &error);

    // 把 program 中的 OpenCL kernel 源代码编译成该 GPU 能执行的程序。
    clBuildProgram(program, 1, &gpu, nullptr, nullptr, nullptr);

    // 从编译完成的 program 中找到名为 matrix_add 的核函数。
    // kernel 是 CPU 端用来代表这个 GPU 核函数的对象。
    cl_kernel kernel = clCreateKernel(program, "matrix_add", &error);

    //==================== 3. 准备 GPU 内存 ====================

    // 在 GPU 上创建只读内存 gpu_A。
    // CL_MEM_COPY_HOST_PTR 表示创建时把 CPU 数组 A 的内容复制进去。
    // bytes 是矩阵 A 占用的总字节数。
    cl_mem gpu_A = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes,
        A,
        &error);

    // 在 GPU 上创建只读内存 gpu_B，同时复制 CPU 数组 B 的内容。
    cl_mem gpu_B = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes,
        B,
        &error);

    // 在 GPU 上为结果矩阵 C 分配可写内存。
    // 此时不需要复制初始数据，所以主机指针传 nullptr。
    cl_mem gpu_C = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        bytes,
        nullptr,
        &error);

    // 给 kernel 设置第 0 个参数，对应 matrix_add 的参数 A。
    clSetKernelArg(kernel, 0, sizeof(gpu_A), &gpu_A);

    // 给 kernel 设置第 1 个参数，对应 matrix_add 的参数 B。
    clSetKernelArg(kernel, 1, sizeof(gpu_B), &gpu_B);

    // 给 kernel 设置第 2 个参数，对应 matrix_add 的参数 C。
    clSetKernelArg(kernel, 2, sizeof(gpu_C), &gpu_C);

    //==================== 4. 设置 block 和线程 ====================

    // 整个线程网格是 4列×4行，共 16 个线程。
    const size_t total_threads[2] = {4, 4};

    // 每个 block 是 2列×2行，共 4 个线程。
    const size_t one_block[2] = {2, 2};

    // 因此：横向 block 数=4÷2=2，纵向 block 数=4÷2=2，
    // 总 block 数=2×2=4。

    // 把运行 matrix_add 的命令加入 queue：
    // 参数1 queue：命令提交到哪个队列；
    // 参数2 kernel：运行哪个核函数；
    // 参数3 2：线程使用二维布局；
    // 参数4 nullptr：线程编号从 (0,0) 开始；
    // 参数5 total_threads：整体线程布局为 4×4；
    // 参数6 one_block：每个 block 的线程布局为 2×2；
    // 后三个参数与事件通知有关，本实验暂时不用。
    clEnqueueNDRangeKernel(queue,
                           kernel,
                           2,
                           nullptr,
                           total_threads,
                           one_block,
                           0,
                           nullptr,
                           nullptr);

    //==================== 5. 把结果取回 CPU ====================

    // 把 GPU 内存 gpu_C 中的结果复制到 CPU 数组 C：
    // CL_TRUE 表示 CPU 在这里等待复制完成后再继续；
    // 0 表示从 gpu_C 的第 0 字节开始；
    // bytes 表示复制整个矩阵；
    // C 是接收结果的 CPU 数组。
    clEnqueueReadBuffer(queue,
                        gpu_C,
                        CL_TRUE,
                        0,
                        bytes,
                        C,
                        0,
                        nullptr,
                        nullptr);

    // 不使用循环，直接展示每个 block、每个线程负责的位置。
    std::cout << "block(0,0)：\n";
    std::cout << "  thread(0,0) -> C[0][0] = " << C[0] << '\n';
    std::cout << "  thread(0,1) -> C[0][1] = " << C[1] << '\n';
    std::cout << "  thread(1,0) -> C[1][0] = " << C[4] << '\n';
    std::cout << "  thread(1,1) -> C[1][1] = " << C[5] << '\n';

    std::cout << "block(0,1)：\n";
    std::cout << "  thread(0,0) -> C[0][2] = " << C[2] << '\n';
    std::cout << "  thread(0,1) -> C[0][3] = " << C[3] << '\n';
    std::cout << "  thread(1,0) -> C[1][2] = " << C[6] << '\n';
    std::cout << "  thread(1,1) -> C[1][3] = " << C[7] << '\n';

    std::cout << "block(1,0)：\n";
    std::cout << "  thread(0,0) -> C[2][0] = " << C[8] << '\n';
    std::cout << "  thread(0,1) -> C[2][1] = " << C[9] << '\n';
    std::cout << "  thread(1,0) -> C[3][0] = " << C[12] << '\n';
    std::cout << "  thread(1,1) -> C[3][1] = " << C[13] << '\n';

    std::cout << "block(1,1)：\n";
    std::cout << "  thread(0,0) -> C[2][2] = " << C[10] << '\n';
    std::cout << "  thread(0,1) -> C[2][3] = " << C[11] << '\n';
    std::cout << "  thread(1,0) -> C[3][2] = " << C[14] << '\n';
    std::cout << "  thread(1,1) -> C[3][3] = " << C[15] << '\n';

    std::cout << "\n结果矩阵 C：\n";
    std::cout << C[0]  << "  " << C[1]  << "  " << C[2]  << "  " << C[3]  << '\n';
    std::cout << C[4]  << "  " << C[5]  << "  " << C[6]  << "  " << C[7]  << '\n';
    std::cout << C[8]  << "  " << C[9]  << "  " << C[10] << "  " << C[11] << '\n';
    std::cout << C[12] << "  " << C[13] << "  " << C[14] << "  " << C[15] << '\n';

    // 释放 GPU 资源。
    clReleaseMemObject(gpu_A);
    clReleaseMemObject(gpu_B);
    clReleaseMemObject(gpu_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
