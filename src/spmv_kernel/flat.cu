#include "common.h"


// ==========================================================================================
// =============================   FLAT 算法 - 修正集成版   ==================================
// ==========================================================================================
//
// 核心改动：
// 1. 彻底移除所有内核定义中的 typename T, typename I 泛型模板。
// 2. 所有数据类型直接使用您框架中的 valT 和 int，与 preFreeSpMV_kernel_bench 写法一致。
// 3. 确保所有函数声明与调用处的模板参数和函数参数完全匹配。
//

/**
 * @brief FLAT 算法的预处理核函数 (已适配您的框架，移除泛型)。
 * @tparam BREAK_STRIDE 每个块处理的非零元素数。
 */
template <int BREAK_STRIDE>
__global__ void pre_startRowPerBlock_flat(const int *__restrict__ row_ptr, const int m, int *__restrict__ startRowPerBlock)
{
    const int global_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    const int global_threads_num = blockDim.x * gridDim.x;

    if (global_thread_id == 0)
    {
        startRowPerBlock[0] = 0;
    }

    for (int i = global_thread_id; i < m; i += global_threads_num)
    {
        if (row_ptr[i] / BREAK_STRIDE != row_ptr[i + 1] / BREAK_STRIDE)
        { 
            for (int b = row_ptr[i] / BREAK_STRIDE + 1; b <= row_ptr[i + 1] / BREAK_STRIDE; b++)
            {
                startRowPerBlock[b] = i;
            }
            if (row_ptr[i + 1] % BREAK_STRIDE == 0)
            {
                startRowPerBlock[row_ptr[i + 1] / BREAK_STRIDE] += 1;
            }
        }
    }
}


/**
 * @brief FLAT 算法的行归约设备函数 (已适配您的框架，移除泛型)。
 * @tparam NNZ_PER_BLOCK 每个块处理的非零元素数。
 * @tparam THREADS_PER_BLOCK 每个块的线程数。
 */
template <int NNZ_PER_BLOCK, int THREADS_PER_BLOCK>
__device__ __forceinline__ void flat_reduce_oneRow_in_thread(const int tid_in_block, const int block_id,
                                                             const int reduceStartRowId, const int reduceEndRowId,
                                                             const int *__restrict__ row_ptr,
                                                             const valT *__restrict__ smem, valT *__restrict__ y)
{
    int reduce_row_id = reduceStartRowId + tid_in_block;
    int nnz_id_before = block_id * NNZ_PER_BLOCK;

    for (; reduce_row_id < reduceEndRowId; reduce_row_id += THREADS_PER_BLOCK)
    {
        valT sum = 0;
        const int reduce_start_idx = (row_ptr[reduce_row_id] - nnz_id_before) < 0 ? 0 : (row_ptr[reduce_row_id] - nnz_id_before);
        const int reduce_end_idx = (row_ptr[reduce_row_id + 1] - nnz_id_before) > NNZ_PER_BLOCK ? NNZ_PER_BLOCK : (row_ptr[reduce_row_id + 1] - nnz_id_before);
        
        for (int i = reduce_start_idx; i < reduce_end_idx; i++)
        {
            sum += smem[i];
        }
        atomicAdd(y + reduce_row_id, sum);
    }
}


/**
 * @brief FLAT 算法的主SpMV计算核函数 (已适配您的框架，移除泛型)。
 * @tparam productNnzPerThread 每个线程处理的非零元素数。
 * @tparam THREADS_PER_BLOCK 每个块的线程数。
 */
template <int productNnzPerThread, int THREADS_PER_BLOCK>
__global__ void spmv_flat_kernel(valT *__restrict__ d_val,
                                 int *__restrict__ d_ptr,
                                 int *__restrict__ d_cols,
                                 int rowA,
                                 valT *__restrict__ d_x,
                                 valT *__restrict__ d_y,
                                 int *__restrict__ startRowPerBlock)
{
    const int tid_in_block = threadIdx.x;
    const int NNZ_PER_BLOCK = THREADS_PER_BLOCK * productNnzPerThread;
    extern __shared__ valT middle_s[];
    
    const int lastElemId = d_ptr[rowA];
    int blockNnzStart = NNZ_PER_BLOCK * blockIdx.x;

    // 1. 乘积阶段 (Product stage)
    #pragma unroll
    for (int round = 0; round < productNnzPerThread; round++)
    {
        const int sIdx = tid_in_block + round * THREADS_PER_BLOCK;
        const int gIdx = min(blockNnzStart + sIdx, lastElemId - 1);
        middle_s[sIdx] = d_val[gIdx] * d_x[d_cols[gIdx]];
    }
    __syncthreads();

    // 2. 归约阶段 (Reduction stage)
    const int reduceStartRowId = min(startRowPerBlock[blockIdx.x], rowA);
    int reduceEndRowId = min(startRowPerBlock[blockIdx.x + 1], rowA);
    reduceEndRowId = (reduceEndRowId == 0) ? rowA : reduceEndRowId;
    if (d_ptr[reduceEndRowId] % NNZ_PER_BLOCK != 0 || reduceEndRowId == reduceStartRowId)
    {
        reduceEndRowId = min(reduceEndRowId + 1, rowA);
    }
    
    // 调用适配后的归约设备函数
    flat_reduce_oneRow_in_thread<NNZ_PER_BLOCK, THREADS_PER_BLOCK>(tid_in_block, blockIdx.x,
                                                                   reduceStartRowId, reduceEndRowId,
                                                                   d_ptr, middle_s, d_y);
}


/**
 * @brief FLAT SpMV算法的基准测试主函数，严格按照您的框架进行集成。
 */
void flatSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
              valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
              double *cdTime, double *cdPre)
{
    CudaTimer timer;
    valT *d_vecY_accu, *d_vecY_perf, *d_vecX, *d_val;
    int *d_indices, *d_ptr;

    // 1. 内存分配与数据传输 (与您的框架完全一致)
    cudaMalloc(&d_vecY_accu, sizeof(valT) * rowA);
    cudaMalloc(&d_vecY_perf, sizeof(valT) * rowA);
    cudaMalloc(&d_vecX, sizeof(valT) * colA);
    cudaMalloc(&d_val, sizeof(valT) * nnzA);
    cudaMalloc(&d_indices, sizeof(int) * nnzA);
    cudaMalloc(&d_ptr, sizeof(int) * (rowA + 1));

    cudaMemcpy(d_val, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, csrColInd, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr, csrRowPtr, sizeof(int) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecX, X_val, sizeof(valT) * colA, cudaMemcpyHostToDevice);

    // 2. 配置核函数启动参数 (与您的框架逻辑一致)
    const int THREADS_PER_BLOCK = 128;
    const int productNnzPerThread = 4;
    const int productNnzPerBlock = THREADS_PER_BLOCK * productNnzPerThread;
    const int WORK_BLOCKS = (nnzA + productNnzPerBlock - 1) / productNnzPerBlock;

    const int startRowPerBlock_len = WORK_BLOCKS + 1;
    int *startRowPerBlock_accu, *startRowPerBlock_perf;
    cudaMalloc(&startRowPerBlock_accu, sizeof(int) * startRowPerBlock_len);
    cudaMalloc(&startRowPerBlock_perf, sizeof(int) * startRowPerBlock_len);

    // 3. 预处理阶段 (调用适配后的 pre_startRowPerBlock_flat)
    dim3 pre_grid(divup<uint32_t>(rowA + 1, 256), 1, 1);
    dim3 pre_block(256, 1, 1);
    
    cudaMemset(startRowPerBlock_accu, 0, sizeof(int) * startRowPerBlock_len);
    pre_startRowPerBlock_flat<productNnzPerBlock><<<pre_grid, pre_block>>>(d_ptr, rowA, startRowPerBlock_accu);
    cudaDeviceSynchronize();

    // 预热和计时
    int warm_iter = 200;
    int test_iter = 4000;
    cudaMemset(startRowPerBlock_perf, 0, sizeof(int) * startRowPerBlock_len);
    for (int i = 0; i < warm_iter; ++i)
    {
        pre_startRowPerBlock_flat<productNnzPerBlock><<<pre_grid, pre_block>>>(d_ptr, rowA, startRowPerBlock_perf);
    }
    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < test_iter; ++i)
    {
        pre_startRowPerBlock_flat<productNnzPerBlock><<<pre_grid, pre_block>>>(d_ptr, rowA, startRowPerBlock_perf);
    }
    float pre_total_time_ms = timer.stop();
    *cdPre = (double)pre_total_time_ms / test_iter;
    CUDA_CHECK_ERROR(cudaGetLastError());
    
    // 4. 主SpMV计算阶段 (调用适配后的 spmv_flat_kernel)
    const int sharedMemSize = productNnzPerBlock * sizeof(valT);
    cudaMemset(d_vecY_perf, 0.0, sizeof(valT) * rowA);
    
    // 预热
    for (int i = 0; i < warm_iter; ++i)
    {
        spmv_flat_kernel<productNnzPerThread, THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), sharedMemSize>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_perf, startRowPerBlock_accu);
    }
    cudaDeviceSynchronize();
    
    // 计时
    timer.start();
    for (int i = 0; i < test_iter; ++i)
    {
        spmv_flat_kernel<productNnzPerThread, THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), sharedMemSize>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_perf, startRowPerBlock_accu);
    }
    float total_loop_time_ms = timer.stop();
    *cdTime = (double)total_loop_time_ms / test_iter;
    
    // 5. 获取结果并清理 (与您的框架完全一致)
    cudaMemset(d_vecY_accu, 0.0, sizeof(valT) * rowA);
    spmv_flat_kernel<productNnzPerThread, THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), sharedMemSize>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_accu, startRowPerBlock_accu);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(cudaGetLastError());

    CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_vecY_accu, sizeof(valT) * rowA, cudaMemcpyDeviceToHost));
    
    cudaFree(d_vecY_perf);
    cudaFree(d_vecY_accu);
    cudaFree(d_vecX);
    cudaFree(d_val);
    cudaFree(d_indices);
    cudaFree(d_ptr);
    cudaFree(startRowPerBlock_accu);
    cudaFree(startRowPerBlock_perf);
}