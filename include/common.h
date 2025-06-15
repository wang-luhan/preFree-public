#pragma once

#include <map>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <array>
#include <cstddef>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <parallel/algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusparse.h>
#include <cublas_v2.h>

#ifdef fp64
#define valT double
#define VAL_CUDA_R_TYPE CUDA_R_64F
#define BUF_CUDA_R_TYPE CUDA_R_64F
#else
#define valT half
#define BUF_CUDA_R_TYPE CUDA_R_32F
#define VAL_CUDA_R_TYPE CUDA_R_16F
#endif

#define CUDA_CHECK_ERROR(call)                                            \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

template <typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
  return (a + b - 1) / b;
}

class CudaTimer {
public:
    /**
     * @brief 构造函数，创建两个 CUDA 事件用于计时。
     */
    CudaTimer() {
        // 检查 CUDA API 调用是否成功是一种好习惯
        cudaError_t err_start = cudaEventCreate(&start_event);
        cudaError_t err_stop = cudaEventCreate(&stop_event);
        if (err_start != cudaSuccess || err_stop != cudaSuccess) {
            std::cerr << "Failed to create CUDA events!" << std::endl;
        }
    }

    /**
     * @brief 析构函数，销毁已创建的 CUDA 事件，防止资源泄漏。
     */
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    // --- 禁止拷贝和赋值 ---
    // 删除拷贝构造函数和拷贝赋值运算符，防止用户意外地拷贝计时器对象，
    // 这可能导致对同一CUDA事件的重复销毁，从而引发运行时错误。
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    /**
     * @brief 开始计时。在GPU流上记录一个起始点。
     */
    void start() {
        cudaEventRecord(start_event, 0); // 在默认流上记录事件
    }

    /**
     * @brief 停止计时并返回经过的时间。
     * @return float 从调用 start() 到调用 stop() 之间，GPU执行所花费的时间（单位：毫秒）。
     *
     * 该函数会记录一个停止事件，并同步等待该事件完成，以确保计时的准确性。
     */
    float stop() {
        float elapsed_ms = 0.0f;
        cudaEventRecord(stop_event, 0);        // 在默认流上记录结束事件
        cudaEventSynchronize(stop_event);      // 阻塞CPU，直到结束事件在GPU上完成
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event); // 计算两个事件间的耗时
        return elapsed_ms;
    }

private:
    cudaEvent_t start_event; // 起始事件句柄
    cudaEvent_t stop_event;  // 结束事件句柄
};


inline void initVec(valT *vec, int length)
{

    for (int i = 0; i < length; ++i)
    {
        vec[i] = static_cast<valT>(static_cast<float>(i % 15));
    }

}
