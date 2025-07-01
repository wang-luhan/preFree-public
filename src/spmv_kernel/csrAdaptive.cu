#include "common.h"

#define ADAPTIVE_BLOCK_DIM 256

// CSR-adaptive核心内核函数
template <int THREADS_PER_BLOCK>
__global__ void csrAdaptiveSpMV_kernel_bench(valT *__restrict__ d_val,
                                              valT *__restrict__ d_x,
                                              int *__restrict__ d_cols,
                                              int *__restrict__ d_ptr,
                                              int N,
                                              int *__restrict__ d_rowBlocks,
                                              valT *__restrict__ d_y)
{
    int startRow = d_rowBlocks[blockIdx.x];
    int nextStartRow = d_rowBlocks[blockIdx.x + 1];
    int num_rows = nextStartRow - startRow;
    int i = threadIdx.x;
    
    extern __shared__ valT LDS[];
    
    // 如果块包含多行，则运行CSR Stream模式
    if (num_rows > 1) {
        int nnz = d_ptr[nextStartRow] - d_ptr[startRow];
        int first_col = d_ptr[startRow];

        // 每个线程写入共享内存
        if (i < nnz) {
            LDS[i] = d_val[first_col + i] * __ldg(&d_x[d_cols[first_col + i]]);
        }
        __syncthreads();

        // 落在范围内的线程对部分结果求和
        for (int k = startRow + i; k < nextStartRow; k += THREADS_PER_BLOCK) {
            valT temp = 0;
            for (int j = (d_ptr[k] - first_col); j < (d_ptr[k + 1] - first_col); j++) {
                temp = temp + LDS[j];
            }
            d_y[k] = temp;
        }
    }
    // 如果块只包含一行，则运行CSR Vector模式
    else {
        int rowStart = d_ptr[startRow];
        int rowEnd = d_ptr[nextStartRow];

        valT sum = 0;

        // 使用块中所有线程累积乘法元素
        for (int j = rowStart + i; j < rowEnd; j += THREADS_PER_BLOCK) {
            int col = d_cols[j];
            sum += d_val[j] * __ldg(&d_x[col]);
        }

        LDS[i] = sum;
        __syncthreads();

        // 归约部分和
        for (int stride = THREADS_PER_BLOCK >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            if (i < stride)
                LDS[i] += LDS[i + stride];
        }
        
        // 写入结果
        if (i == 0)
            d_y[startRow] = LDS[i];
    }
}

// 计算CSR矩阵的行块分配
int csrAdaptive_rowBlocks(int *ptr, int totalRows, int *rowBlocks, int blockDim)
{
    rowBlocks[0] = 0;
    int sum = 0;
    int last_i = 0;
    int ctr = 1;
    
    for (int i = 1; i < totalRows; i++) {
        // 计算当前行的非零元素数量
        sum += ptr[i] - ptr[i - 1];
        
        if (sum == blockDim) {
            // 当前行正好填满blockDim
            last_i = i;
            rowBlocks[ctr++] = i;
            sum = 0;
        }
        else if (sum > blockDim) {
            if (i - last_i > 1) {
                // 这个额外的行不适合
                rowBlocks[ctr++] = i - 1;
                i--;
            }
            else if (i - last_i == 1) {
                // 这一行太大
                rowBlocks[ctr++] = i;
            }
            last_i = i;
            sum = 0;
        }
    }
    rowBlocks[ctr++] = totalRows;
    return ctr;
}

// 预处理函数：计算行块分配
void csrAdaptive_preprocess(int *csrRowPtr, int rowA, int **h_rowBlocks, int **d_rowBlocks, int *countRowBlocks)
{
    // 分配主机内存
    *h_rowBlocks = (int *)malloc(rowA * sizeof(int));
    
    // 计算行块分配
    *countRowBlocks = csrAdaptive_rowBlocks(csrRowPtr, rowA, *h_rowBlocks, ADAPTIVE_BLOCK_DIM);
    
    // 分配设备内存并拷贝
    cudaMalloc((void**)d_rowBlocks, (*countRowBlocks) * sizeof(int));
    cudaMemcpy(*d_rowBlocks, *h_rowBlocks, (*countRowBlocks) * sizeof(int), cudaMemcpyHostToDevice);
}

// 主函数，采用你的框架接口风格
void csrAdaptiveSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
                     valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
                     double *cdTime, double *cdPre)
{
    CudaTimer timer;
    valT *d_vecY_accu, *d_vecY_perf, *d_vecX, *d_val;
    int *d_indices, *d_ptr;
    int *h_rowBlocks, *d_rowBlocks;
    int countRowBlocks;

    // 内存分配
    cudaMalloc(&d_vecY_accu, sizeof(valT) * rowA);
    cudaMalloc(&d_vecY_perf, sizeof(valT) * rowA);
    cudaMalloc(&d_vecX, sizeof(valT) * colA);
    cudaMalloc(&d_val, sizeof(valT) * nnzA);
    cudaMalloc(&d_indices, sizeof(int) * nnzA);
    cudaMalloc(&d_ptr, sizeof(int) * (rowA + 1));

    // 数据拷贝
    cudaMemcpy(d_val, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, csrColInd, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, csrRowPtr, sizeof(int) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecX, X_val, sizeof(valT) * colA, cudaMemcpyHostToDevice);

    // 预处理：计算行块分配
    int warm_iter = 200;
    int test_iter = 4000;
    
    timer.start();
    for (int i = 0; i < test_iter; ++i) {
        csrAdaptive_preprocess(csrRowPtr, rowA, &h_rowBlocks, &d_rowBlocks, &countRowBlocks);
        // 立即释放以模拟预处理开销
        free(h_rowBlocks);
        cudaFree(d_rowBlocks);
    }
    float pre_total_time_ms = timer.stop();
    *cdPre = (double)pre_total_time_ms / test_iter;

    // 最终预处理（用于实际计算）
    csrAdaptive_preprocess(csrRowPtr, rowA, &h_rowBlocks, &d_rowBlocks, &countRowBlocks);

    // 参数设置
    const int THREADS_PER_BLOCK = ADAPTIVE_BLOCK_DIM;
    const int WORK_BLOCKS = countRowBlocks - 1;
    const int shared_mem_size = ADAPTIVE_BLOCK_DIM * sizeof(valT);

    // 清零结果向量
    cudaMemset(d_vecY_perf, 0.0, sizeof(valT) * rowA);

    // 预热
    for (int i = 0; i < warm_iter; ++i) {
        csrAdaptiveSpMV_kernel_bench<ADAPTIVE_BLOCK_DIM>
            <<<WORK_BLOCKS, THREADS_PER_BLOCK, shared_mem_size>>>
            (d_val, d_vecX, d_indices, d_ptr, rowA, d_rowBlocks, d_vecY_perf);
    }
    cudaDeviceSynchronize();

    // 正式测试
    timer.start();
    for (int i = 0; i < test_iter; ++i) {
        csrAdaptiveSpMV_kernel_bench<ADAPTIVE_BLOCK_DIM>
            <<<WORK_BLOCKS, THREADS_PER_BLOCK, shared_mem_size>>>
            (d_val, d_vecX, d_indices, d_ptr, rowA, d_rowBlocks, d_vecY_perf);
    }
    float total_loop_time_ms = timer.stop();

    // 计算准确结果
    cudaMemset(d_vecY_accu, 0.0, sizeof(valT) * rowA);
    csrAdaptiveSpMV_kernel_bench<ADAPTIVE_BLOCK_DIM>
        <<<WORK_BLOCKS, THREADS_PER_BLOCK, shared_mem_size>>>
        (d_val, d_vecX, d_indices, d_ptr, rowA, d_rowBlocks, d_vecY_accu);

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(cudaGetLastError());

    // 计算性能指标
    *cdTime = (double)total_loop_time_ms / test_iter;
    double cd_gflops = (double)((long)nnzA * 2) / (total_loop_time_ms * 1e6);

    // 输出性能信息
    printf("CSR-Adaptive SpMV Performance: %.2f GFLOPS\n", cd_gflops);
    printf("CSR-Adaptive Average Time: %.4f ms\n", *cdTime);
    printf("CSR-Adaptive Preprocess Time: %.4f ms\n", *cdPre);
    printf("CSR-Adaptive Row Blocks: %d\n", countRowBlocks - 1);

    // 拷贝结果
    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_vecY_accu, sizeof(valT) * rowA, cudaMemcpyDeviceToHost));

    // 释放内存
    free(h_rowBlocks);
    cudaFree(d_rowBlocks);
    cudaFree(d_vecY_perf);
    cudaFree(d_vecY_accu);
    cudaFree(d_vecX);
    cudaFree(d_val);
    cudaFree(d_indices);
    cudaFree(d_ptr);
}
