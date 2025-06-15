#include "common.h"
__global__ void pre_startRowPerTile(const int *__restrict__ row_ptr,
                                    const int m,
                                    int *__restrict__ startRowPerBlock,
                                    int tileSize)
{
  const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_thread_id >= m)
    return;
  int a = row_ptr[global_thread_id];
  int b = row_ptr[min(global_thread_id + 1, (int)m)];
  int blocka = divup<int>(a, tileSize);
  int blockb = (b - 1) / static_cast<int>(tileSize);
  if (a != b)
    for (; blocka <= blockb; ++blocka)
      startRowPerBlock[blocka] = global_thread_id;
}

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ void red_row_block(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
                                              const int reduceStartRowId,
                                              const int *__restrict__ row_ptr,
                                              const valT *__restrict__ smem, valT *__restrict__ y)
{
  constexpr int num_warps = THREADS_PER_BLOCK >> 5;
  __shared__ valT warp_results[num_warps];
  const int reduce_start_idx = max((int)0, row_ptr[reduceStartRowId] - block_id * NNZ_PER_BLOCK);
  const int reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduceStartRowId + 1] - block_id * NNZ_PER_BLOCK);

  valT thread_sum = 0;
  for (int j = reduce_start_idx + tid_in_block; j < reduce_end_idx; j += THREADS_PER_BLOCK)
  {
    thread_sum += smem[j];
  }

  for (int offset = 16; offset > 0; offset >>= 1)
  {
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
  }
  const int warp_id = tid_in_block >> 5;
  const int lane_id = tid_in_block & 31;
  if (lane_id == 0)
  {
    warp_results[warp_id] = thread_sum;
  }
  __syncthreads();

  if (tid_in_block == 0)
  {
    valT final_sum = 0;
    for (int i = 0; i < num_warps; i++)
    {
      final_sum += warp_results[i];
    }
    atomicAdd(y + reduceStartRowId, final_sum);
  }
}

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ void red_row_thread(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
                                               const int reduceStartRowId, const int reduceEndRowId,
                                               const int *__restrict__ row_ptr,
                                               const valT *__restrict__ smem, valT *__restrict__ y)
{
  int reduce_row_id = reduceStartRowId + tid_in_block;
  int nnz_id_before = block_id * NNZ_PER_BLOCK;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += THREADS_PER_BLOCK)
  {
    valT sum = 0;
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - nnz_id_before);
    const int reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - nnz_id_before);
    for (int i = reduce_start_idx; i < reduce_end_idx; i++)
    {
      sum += smem[i];
    }
    atomicAdd(y + reduce_row_id, sum);
  }
}

template <int VEC_SIZE>
__device__ __forceinline__ valT warpReduceSum(valT sum)
{
  if (VEC_SIZE >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
  if (VEC_SIZE >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
  if (VEC_SIZE >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
  if (VEC_SIZE >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
  if (VEC_SIZE >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
  return sum;
}

template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
red_row_vector_inwarp(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                      const int reduceStartRowId, const int reduceEndRowId,
                      const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{

  const int vec_size = VECTOR_SIZE;
  const int vec_num = THREADS_PER_BLOCK / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block & (vec_size - 1);

  int reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    valT sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += vec_size)
    {
      sum += smem[i];
    }
    sum = warpReduceSum<vec_size>(sum);
    // store value
    if (tid_in_vec == 0)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
  }
}

template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
red_row_vector_crosswarp(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                         const int reduceStartRowId, const int reduceEndRowId,
                         const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{

  const int vec_size = VECTOR_SIZE;
  const int vec_num = THREADS_PER_BLOCK / vec_size;
  const int vec_id = tid_in_block / vec_size;
  const int tid_in_vec = tid_in_block & (vec_size - 1);
  const int warp_lane_id = tid_in_block & 31;

  int reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    valT sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += vec_size)
    {
      sum += smem[i];
    }
    // sum = warpReduceSum<vec_size>(sum);
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // store value
    if (warp_lane_id == 0)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
  }
}



template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
red_row_vector_dynamic(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                       const int reduceStartRowId, const int reduceEndRowId,
                       const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
    // 1. 设置共享内存原子计数器
    __shared__ int next_row_offset;
    if (tid_in_block == 0) {
        next_row_offset = 0;
    }
    __syncthreads();

    const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1); // 线程在Vector内的ID

    // 2. 动态获取任务的循环
    while (true) {
        int my_row_offset = -1;

        // 只有Vector的“队长”线程（例如，ID为0的线程）去获取任务
        if (tid_in_vec == 0) {
            my_row_offset = atomicAdd(&next_row_offset, 1);
        }
        
        // 使用__shfl_sync或共享内存将队长获取的 my_row_offset 广播给Vector内的其他线程
        // 对于Warp内（VEC_SIZE <= 32），__shfl_sync 是最高效的方式
        my_row_offset = __shfl_sync(0xffffffff, my_row_offset, 0, VECTOR_SIZE);
        
        int reduce_row_id = reduceStartRowId + my_row_offset;

        // 3. 检查任务是否已经取完
        if (reduce_row_id >= reduceEndRowId) {
            break; // 所有行都已被处理，退出循环
        }

        // 4. 执行计算 (与您之前的逻辑相同)
        const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
        const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
        
        valT sum = 0;
        for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += VECTOR_SIZE) {
            sum += smem[i];
        }
        
        sum = warpReduceSum<VECTOR_SIZE>(sum); // 使用Warp内reduce求和
        
        if (tid_in_vec == 0) {
            atomicAdd(y + reduce_row_id, sum);
        }
    }
}

template <int THREADS_PER_BLOCK>
__global__ void preFreeSpMV_kernel(valT *__restrict__ d_val,
                                   int *__restrict__ d_ptr,
                                   int *__restrict__ d_cols,
                                   int rowA,
                                   valT *__restrict__ d_x,
                                   valT *__restrict__ d_y,
                                   int *__restrict__ startRowPerBlock,
                                   int productNnzPerThread,
                                   int productNnzPerBlock)
{
  // int NNZ_PER_BLOCK = THREADS_PER_BLOCK * productNnzPerThread;
  extern __shared__ valT middle_s[];
  // __shared__ valT middle_s[NNZ_PER_BLOCK];
  const int last = d_ptr[rowA] - 1;
  int blockNnzStart = productNnzPerBlock * blockIdx.x;

#pragma unroll
  for (int round = 0; round < productNnzPerThread; round++)
  {
    const int sIdx = threadIdx.x + round * THREADS_PER_BLOCK;
    const int gIdx = min(blockNnzStart + sIdx, last);
    middle_s[sIdx] = d_val[gIdx] * __ldg(&d_x[d_cols[gIdx]]);
  }
  __syncthreads();

  const int reduceStartRowId = min(startRowPerBlock[blockIdx.x], rowA);
  int reduceEndRowId = min(startRowPerBlock[blockIdx.x + 1], rowA);
  reduceEndRowId = (reduceEndRowId == 0) ? rowA : reduceEndRowId;
  if (d_ptr[reduceEndRowId] % productNnzPerBlock != 0 || reduceEndRowId == reduceStartRowId)
  {
    reduceEndRowId = min(reduceEndRowId + 1, rowA);
  }

  const int n_reduce_rows_num = reduceEndRowId - reduceStartRowId;
  if (n_reduce_rows_num > 64)
  {
    red_row_thread<THREADS_PER_BLOCK>(productNnzPerBlock, threadIdx.x, blockIdx.x,
                                      reduceStartRowId, reduceEndRowId,
                                      d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num == 1)
  {
    red_row_block<THREADS_PER_BLOCK>(productNnzPerBlock, threadIdx.x, blockIdx.x,
                                     reduceStartRowId,
                                     d_ptr, middle_s, d_y);
  }

  else if (n_reduce_rows_num == 2)
  {
    red_row_vector_crosswarp<THREADS_PER_BLOCK, 64>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                    reduceStartRowId, reduceEndRowId,
                                                    d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 4)
  {
    red_row_vector_dynamic<THREADS_PER_BLOCK, 32>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                 reduceStartRowId, reduceEndRowId,
                                                 d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 8)
  {
    red_row_vector_dynamic<THREADS_PER_BLOCK, 16>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                 reduceStartRowId, reduceEndRowId,
                                                 d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 16)
  {
    red_row_vector_dynamic<THREADS_PER_BLOCK, 8>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                reduceStartRowId, reduceEndRowId,
                                                d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 32)
  {
    red_row_vector_dynamic<THREADS_PER_BLOCK, 4>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                reduceStartRowId, reduceEndRowId,
                                                d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 64)
  {
    red_row_vector_dynamic<THREADS_PER_BLOCK, 2>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                reduceStartRowId, reduceEndRowId,
                                                d_ptr, middle_s, d_y);
  }
  /*
  else if (n_reduce_rows_num == 2)
  {
    red_row_vector_crosswarp<THREADS_PER_BLOCK, 64>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                    reduceStartRowId, reduceEndRowId,
                                                    d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 4)
  {
    red_row_vector_inwarp<THREADS_PER_BLOCK, 32>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                 reduceStartRowId, reduceEndRowId,
                                                 d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 8)
  {
    red_row_vector_inwarp<THREADS_PER_BLOCK, 16>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                 reduceStartRowId, reduceEndRowId,
                                                 d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 16)
  {
    red_row_vector_inwarp<THREADS_PER_BLOCK, 8>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                reduceStartRowId, reduceEndRowId,
                                                d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 32)
  {
    red_row_vector_inwarp<THREADS_PER_BLOCK, 4>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                reduceStartRowId, reduceEndRowId,
                                                d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 64)
  {
    red_row_vector_inwarp<THREADS_PER_BLOCK, 2>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                reduceStartRowId, reduceEndRowId,
                                                d_ptr, middle_s, d_y);
  }
  */
  /*
  else
  {
    red_dynamic_queue_batch_cooperation<THREADS_PER_BLOCK>(productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
                                                          reduceStartRowId, reduceEndRowId,
                                                          d_ptr, middle_s, d_y);
  }
  */
}

void preFreeSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
                 valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
                 double *cdTime, double *cdPre)
{
  CudaTimer timer;
  valT *d_vecY_accu, *d_vecY_perf, *d_vecX, *d_val;
  int *d_indices, *d_ptr;

  cudaMalloc(&d_vecY_accu, sizeof(valT) * rowA);
  cudaMalloc(&d_vecY_perf, sizeof(valT) * rowA);
  cudaMalloc(&d_vecX, sizeof(valT) * colA);
  cudaMalloc(&d_val, sizeof(valT) * nnzA);
  cudaMalloc(&d_indices, sizeof(int) * nnzA);
  cudaMalloc(&d_ptr, sizeof(int) * (rowA + 2));

  cudaMemcpy(d_val, csrVal, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, csrColInd, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr, csrRowPtr, sizeof(int) * (rowA + 2), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vecX, X_val, sizeof(valT) * colA, cudaMemcpyHostToDevice);

  const int THREADS_PER_BLOCK = 128;

#ifdef fp64
  int productNnzPerThread = (nnzA > 200000000) ? 8 : 4;
#else
  int productNnzPerThread = (nnzA > 300000) ? 16 : 4;
#endif
  const int WORK_BLOCKS = nnzA / (productNnzPerThread * THREADS_PER_BLOCK) + ((nnzA % (productNnzPerThread * THREADS_PER_BLOCK) == 0) ? 0 : 1);

  const int startRowPerBlock_len = WORK_BLOCKS + 1;

  int *startRowPerBlock;
  cudaMalloc(&startRowPerBlock, sizeof(int) * startRowPerBlock_len);
  cudaMemset(startRowPerBlock, 0, sizeof(int) * startRowPerBlock_len);
  timer.start();
  pre_startRowPerTile<<<divup<uint32_t>(rowA + 1, 128), 128>>>(d_ptr, rowA, startRowPerBlock, productNnzPerThread * THREADS_PER_BLOCK);
  *cdPre = (double)timer.stop();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("pre_startRowPerTile kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  int productNnzPerBlock = THREADS_PER_BLOCK * productNnzPerThread;

  int warm_iter = 200;
  int test_iter = 4000;
  cudaMemset(d_vecY_perf, 0.0, sizeof(valT) * rowA);

  for (int i = 0; i < warm_iter; ++i)
  {
    preFreeSpMV_kernel<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_perf, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
  }
  cudaDeviceSynchronize();
  timer.start();
  for (int i = 0; i < test_iter; ++i)
  {
    preFreeSpMV_kernel<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_perf, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
  }
  float total_loop_time_ms = timer.stop();

  cudaMemset(d_vecY_accu, 0.0, sizeof(valT) * rowA);
  preFreeSpMV_kernel<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_accu, startRowPerBlock, productNnzPerThread, productNnzPerBlock);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR(cudaGetLastError());

  *cdTime = (double)total_loop_time_ms / test_iter;
  double cd_gflops = (double)((long)nnzA * 2) / (total_loop_time_ms * 1e6);

  CUDA_CHECK_ERROR(cudaMemcpy(Y_val, d_vecY_accu, sizeof(valT) * rowA, cudaMemcpyDeviceToHost));
  cudaFree(d_vecY_perf);
  cudaFree(d_vecY_accu);
  cudaFree(d_vecX);
  cudaFree(d_val);
  cudaFree(d_indices);
  cudaFree(d_ptr);
  cudaFree(startRowPerBlock);
}
