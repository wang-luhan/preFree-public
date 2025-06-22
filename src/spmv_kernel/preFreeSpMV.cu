#include "common.h"

__global__ void pre_startRowPerTile(const int *__restrict__ row_ptr,
                                    const int m,
                                    int *__restrict__ startRowPerBlock,
                                    int tileSize)
{
  const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id >= m)
  {
    return;
  }
  const int a = row_ptr[row_id];
  const int b = row_ptr[row_id + 1];
  if (a == b)
  {
    return;
  }
  const int start_block = divup<int>(a, tileSize);
  const int end_block = (b - 1) / tileSize;
  for (int i = start_block; i <= end_block; ++i)
  {
    if (i >= start_block)
    {
      startRowPerBlock[i] = row_id;
    }
  }
}

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ void
red_row_block(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
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
__device__ __forceinline__ void
red_row_thread(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
               const int reduceStartRowId, const int reduceEndRowId,
               const int *__restrict__ row_ptr,
               const valT *__restrict__ smem, valT *__restrict__ y)
{
  // 1. 设置共享内存原子计数器
  __shared__ int next_row_offset;
  if (tid_in_block == 0)
  {
    next_row_offset = 0;
  }
  __syncthreads();

  const int nnz_id_before = block_id * NNZ_PER_BLOCK;
  // 2. 动态获取任务的循环
  while (true)
  {
    // 每个线程都独立地去获取一个新行作为任务
    const int my_row_offset = atomicAdd(&next_row_offset, 1);
    const int reduce_row_id = reduceStartRowId + my_row_offset;

    // 3. 检查任务是否已经取完
    if (reduce_row_id >= reduceEndRowId)
    {
      break; // 所有行都已被处理，退出循环
    }

    // 4. 执行计算 (与原函数相同的串行计算)
    valT sum = 0;
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - nnz_id_before);
    const int reduce_end_idx = min(NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - nnz_id_before);

    for (int i = reduce_start_idx; i < reduce_end_idx; i++)
    {
      sum += smem[i];
    }
    // atomicAdd(y + reduce_row_id, sum);
    if (reduce_row_id == reduceStartRowId || reduce_row_id == reduceEndRowId - 1)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
    else
    {
      y[reduce_row_id] = sum;
    }
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

#define MAX_ROWS_PER_BLOCK 128
template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
red_row_vector_1(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                 const int reduceStartRowId, const int reduceEndRowId,
                 const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  __shared__ int s_cached_ptr[MAX_ROWS_PER_BLOCK + 1];
  for (int i = tid_in_block; i < n_reduce_rows_num + 1; i += THREADS_PER_BLOCK)
  {
    s_cached_ptr[i] = row_ptr[reduceStartRowId + i];
  }
  __syncthreads();

  const int vec_num = THREADS_PER_BLOCK / VECTOR_SIZE;
  const int vec_id = tid_in_block / VECTOR_SIZE;
  const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);

  for (int row_offset = vec_id; row_offset < n_reduce_rows_num; row_offset += vec_num)
  {
    const int reduce_row_id = reduceStartRowId + row_offset;

    const int row_start_ptr = s_cached_ptr[row_offset];
    const int row_end_ptr = s_cached_ptr[row_offset + 1];

    const int reduce_start_idx = max((int)0, row_start_ptr - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_end_ptr - block_id * NNZ_PER_BLOCK);

    valT sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += VECTOR_SIZE)
    {
      sum += smem[i];
    }

    sum = warpReduceSum<VECTOR_SIZE>(sum);

    if (tid_in_vec == 0)
    {
      if (n_reduce_rows_num <= 2 || reduce_row_id == reduceStartRowId || reduce_row_id == reduceEndRowId - 1)
      {
        atomicAdd(y + reduce_row_id, sum);
      }
      else
      {
        y[reduce_row_id] = sum;
      }
    }
  }
}



/**
 * @brief 针对“单一超长行 + 多个短行”优化的极致性能规约函数
 * @tparam THREADS_PER_BLOCK      块内线程总数 (例如 128)
 * @tparam VECTOR_SIZE            短行处理团队的Vector大小 (例如 16)
 * @tparam LONG_ROW_WARPS         分配给长行处理的Warp数量 (建议为块内Warp总数的一半)
 */
template <int THREADS_PER_BLOCK, int VECTOR_SIZE, int LONG_ROW_WARPS = 2>
__device__  void
red_row_vector_3(
    int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
    const int reduceStartRowId, const int reduceEndRowId,
    const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  // ====================================================================================
  // 阶段 1: 块内并行分析，识别长行，分离长短行 (逻辑不变，保持正确)
  // ====================================================================================

  // --- 共享内存声明 ---
  __shared__ int s_row_lens[MAX_ROWS_PER_BLOCK];
  __shared__ int s_long_row_info[2]; // {max_len, local_idx}
  __shared__ valT s_final_sums[MAX_ROWS_PER_BLOCK];
  __shared__ int s_short_row_map[MAX_ROWS_PER_BLOCK];
  __shared__ int s_short_row_count;
  // [修正] 为长行团队的跨Warp规约准备的共享内存
  __shared__ valT s_long_row_warp_sums[LONG_ROW_WARPS];

  // --- 并行初始化 ---
  for (int i = tid_in_block; i < n_reduce_rows_num; i += THREADS_PER_BLOCK)
  {
    s_final_sums[i] = 0.0;
  }
  if (tid_in_block == 0)
  {
    s_long_row_info[0] = -1;
    s_long_row_info[1] = -1;
    s_short_row_count = 0;
  }
  __syncthreads();

  // --- 并行计算行长并找到最长行 ---
  for (int i = tid_in_block; i < n_reduce_rows_num; i += THREADS_PER_BLOCK)
  {
    s_row_lens[i] = row_ptr[reduceStartRowId + i + 1] - row_ptr[reduceStartRowId + i];
  }
  __syncthreads();
  for (int i = tid_in_block; i < n_reduce_rows_num; i += THREADS_PER_BLOCK)
  {
    atomicMax(&s_long_row_info[0], s_row_lens[i]);
  }
  __syncthreads();
  if (tid_in_block == 0)
  {
    for (int i = 0; i < n_reduce_rows_num; i++)
    {
      if (s_row_lens[i] == s_long_row_info[0])
      {
        s_long_row_info[1] = i;
        break;
      }
    }
  }
  __syncthreads();

  // --- 创建短行索引的紧凑映射表 ---
  if (tid_in_block < 32)
  {
    for (int i = tid_in_block; i < n_reduce_rows_num; i += 32)
    {
      if (i != s_long_row_info[1])
      {
        int map_idx = atomicAdd(&s_short_row_count, 1);
        s_short_row_map[map_idx] = i;
      }
    }
  }
  __syncthreads();

  // ====================================================================================
  // 阶段 2 & 3: 静态分工与并行执行
  // ====================================================================================
  const int warp_id = tid_in_block >> 5;
  const int lane_id = tid_in_block & 31;

  // ---------------------------
  //  长行处理团队 (Long-Row Team)
  // ---------------------------
  if (warp_id < LONG_ROW_WARPS)
  {
    const int long_row_local_idx = s_long_row_info[1];
    if (long_row_local_idx != -1) // 确保长行已被找到
    {
      const int reduce_row_id = reduceStartRowId + long_row_local_idx;
      const int row_start_ptr = row_ptr[reduce_row_id];
      const int row_end_ptr = row_ptr[reduce_row_id + 1];
      const int reduce_start_idx = max((int)0, row_start_ptr - block_id * NNZ_PER_BLOCK);
      const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_end_ptr - block_id * NNZ_PER_BLOCK);

      valT thread_sum = 0;
      const int team_tid = tid_in_block;
      const int team_size = LONG_ROW_WARPS * 32;

      for (int i = reduce_start_idx + team_tid; i < reduce_end_idx; i += team_size)
      {
        thread_sum += smem[i];
      }

      // [修正] 正确的跨Warp规约
      // 1. Warp内规约
      thread_sum = warpReduceSum<32>(thread_sum); // 使用一个标准的32线程Warp规约

      // 2. Warp leader将结果写入共享内存
      if (lane_id == 0)
      {
        s_long_row_warp_sums[warp_id] = thread_sum;
      }
      __syncthreads(); // <<-- 此处同步至关重要！

      // 3. 由单个线程(0号)完成最后的加和
      if (tid_in_block == 0)
      {
        valT total_sum = 0;
        for (int i = 0; i < LONG_ROW_WARPS; i++)
        {
          total_sum += s_long_row_warp_sums[i];
        }
        s_final_sums[long_row_local_idx] = total_sum;
      }
    }
  }
  // ---------------------------
  //  短行处理团队 (Short-Row Team) - (逻辑无误，无需修改)
  // ---------------------------
  else
  {
    const int team_tid = tid_in_block - (LONG_ROW_WARPS * 32);
    constexpr int team_warps = (THREADS_PER_BLOCK / 32) - LONG_ROW_WARPS;
    constexpr int team_vectors = team_warps * (32 / VECTOR_SIZE);

    const int vec_id = team_tid / VECTOR_SIZE;
    const int tid_in_vec = team_tid & (VECTOR_SIZE - 1);

    for (int row_offset = vec_id; row_offset < s_short_row_count; row_offset += team_vectors)
    {
      const int short_row_local_idx = s_short_row_map[row_offset];
      const int reduce_row_id = reduceStartRowId + short_row_local_idx;
      const int row_start_ptr = row_ptr[reduce_row_id];
      const int row_end_ptr = row_ptr[reduce_row_id + 1];
      const int reduce_start_idx = max((int)0, row_start_ptr - block_id * NNZ_PER_BLOCK);
      const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_end_ptr - block_id * NNZ_PER_BLOCK);

      valT sum = 0;
      for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += VECTOR_SIZE)
      {
        sum += smem[i];
      }
      sum = warpReduceSum<VECTOR_SIZE>(sum);

      if (tid_in_vec == 0)
      {
        s_final_sums[short_row_local_idx] = sum;
      }
    }
  }

  // ====================================================================================
  // 阶段 4: 最终结果并行写回 (逻辑无误，无需修改)
  // ====================================================================================
  __syncthreads();

  for (int i = tid_in_block; i < n_reduce_rows_num; i += THREADS_PER_BLOCK)
  {
    const valT final_sum = s_final_sums[i];
    if (final_sum == 0.0)
      continue;

    const int final_row_id = reduceStartRowId + i;

    if (n_reduce_rows_num <= 2 || final_row_id == reduceStartRowId || final_row_id == reduceEndRowId - 1)
    {
      atomicAdd(y + final_row_id, final_sum);
    }
    else
    {
      y[final_row_id] = final_sum;
    }
  }
}

template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
dispatch_reduction_strategy(
    bool use_uneven, int productNnzPerBlock, int n_reduce_rows_num,
    int tid_in_block, int block_id, int reduceStartRowId, int reduceEndRowId,
    const int *__restrict__ d_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  if (use_uneven)
  {
    red_row_vector_3<THREADS_PER_BLOCK, VECTOR_SIZE>(
        productNnzPerBlock, n_reduce_rows_num, tid_in_block, block_id,
        reduceStartRowId, reduceEndRowId, d_ptr, smem, y);
  }
  else
  {
    red_row_vector_1<THREADS_PER_BLOCK, VECTOR_SIZE>(
        productNnzPerBlock, n_reduce_rows_num, tid_in_block, block_id,
        reduceStartRowId, reduceEndRowId, d_ptr, smem, y);
  }
}

__device__ __forceinline__ bool
is_imbalanced(const int reduceStartRowId, const int n_reduce_rows_num,
              const int *__restrict__ d_ptr,
              const int tid_in_block)
{
  const int lane_id = tid_in_block & 31;
  int my_len = 0;
  if (lane_id < n_reduce_rows_num)
  {
    const int my_row_idx = reduceStartRowId + lane_id;
    my_len = d_ptr[my_row_idx + 1] - d_ptr[my_row_idx];
  }
  int total_nnz_in_warp = my_len;
  int max_len_in_warp = my_len;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    total_nnz_in_warp += __shfl_down_sync(0xFFFFFFFF, total_nnz_in_warp, offset);
    max_len_in_warp = max(max_len_in_warp, __shfl_down_sync(0xFFFFFFFF, max_len_in_warp, offset));
  }
  bool decision = false;
  if (lane_id == 0)
  {
    const float DOMINANCE_FACTOR = 0.5f;
    decision = (max_len_in_warp > total_nnz_in_warp * DOMINANCE_FACTOR);
  }
  decision = __shfl_sync(0xFFFFFFFF, decision, 0);
  return decision;
}

__device__ __forceinline__ bool
is_imbalanced_small(const int reduceStartRowId, const int n_reduce_rows_num,
                    const int *__restrict__ d_ptr,
                    const int tid_in_block)
{
  const int lane_id = tid_in_block & 31;
  int my_len = 0;
  if (lane_id < n_reduce_rows_num)
  {
    const int my_row_idx = reduceStartRowId + lane_id;
    my_len = d_ptr[my_row_idx + 1] - d_ptr[my_row_idx];
  }
  const int len0 = __shfl_sync(0xFFFFFFFF, my_len, 0);
  const int len1 = __shfl_sync(0xFFFFFFFF, my_len, 1);
  const int len2 = __shfl_sync(0xFFFFFFFF, my_len, 2);
  const int len3 = __shfl_sync(0xFFFFFFFF, my_len, 3);
  bool decision = false;
  if (lane_id == 0)
  {
    const int total_nnz = len0 + len1 + len2 + len3;
    const int max_len = max(max(len0, len1), max(len2, len3));

    const float DOMINANCE_FACTOR = 0.5f;
    if (total_nnz > 0)
    {
      decision = (max_len > total_nnz * DOMINANCE_FACTOR);
    }
  }
  decision = __shfl_sync(0xFFFFFFFF, decision, 0);
  return decision;
}

template <int MAX_N>
__device__ __forceinline__ bool
is_imbalanced_warp(const int reduceStartRowId, const int n_reduce_rows_num,
                   const int *__restrict__ d_ptr,
                   const int tid_in_block)
{
  const int lane_id = tid_in_block & 31;
  int my_len = 0;
  if (lane_id < n_reduce_rows_num)
  {
    const int my_row_idx = reduceStartRowId + lane_id;
    my_len = d_ptr[my_row_idx + 1] - d_ptr[my_row_idx];
  }
  int total_nnz = my_len;
  int max_len = my_len;
#pragma unroll
  for (int offset = MAX_N / 2; offset > 0; offset >>= 1)
  {
    total_nnz += __shfl_down_sync(0xFFFFFFFF, total_nnz, offset);
    max_len = max(max_len, __shfl_down_sync(0xFFFFFFFF, max_len, offset));
  }
  bool decision = false;
  if (lane_id == 0)
  {
    const float DOMINANCE_FACTOR = 0.8f;
    if (n_reduce_rows_num > 1 && total_nnz > 0)
    {
      decision = (max_len > total_nnz * DOMINANCE_FACTOR);
    }
  }
  decision = __shfl_sync(0xFFFFFFFF, decision, 0);
  return decision;
}

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ unsigned int
calculate_vector_size(int n_reduce_rows_num)
{
  if (n_reduce_rows_num >= THREADS_PER_BLOCK)
  {
    return 2;
  }
  unsigned int avg_threads_per_row = THREADS_PER_BLOCK / n_reduce_rows_num;
  unsigned int highest_power_of_2 = 1 << (31 - __clz(avg_threads_per_row));
  return min(32, max(2, highest_power_of_2));
}

template <int THREADS_PER_BLOCK>
__global__ void preFreeSpMV_kernel_bench(valT *__restrict__ d_val,
                                         int *__restrict__ d_ptr,
                                         int *__restrict__ d_cols,
                                         int rowA,
                                         valT *__restrict__ d_x,
                                         valT *__restrict__ d_y,
                                         int *__restrict__ startRowPerBlock,
                                         int productNnzPerThread,
                                         int productNnzPerBlock)
{
  extern __shared__ valT middle_s[];
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

  if (n_reduce_rows_num > 128)
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
  else if (n_reduce_rows_num <= 4)
  {
    __shared__ bool use_uneven_path;
    if ((threadIdx.x >> 5) == 0)
    {
      bool decision = is_imbalanced_small(reduceStartRowId, n_reduce_rows_num, d_ptr, threadIdx.x);
      if (threadIdx.x == 0)
      {
        use_uneven_path = decision;
      }
    }
    __syncthreads();
    dispatch_reduction_strategy<THREADS_PER_BLOCK, 32>(
        use_uneven_path, productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
        reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 8)
  {
    red_row_vector_1<THREADS_PER_BLOCK, 16>(
        productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
        reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 16)
  {
    red_row_vector_1<THREADS_PER_BLOCK, 8>(
        productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
        reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 32)
  {
    red_row_vector_1<THREADS_PER_BLOCK, 4>(
        productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
        reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
  }
  else if (n_reduce_rows_num <= 128)
  {
    red_row_vector_1<THREADS_PER_BLOCK, 2>(
        productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
        reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
  }
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
  int productNnzPerThread = 4;
  const int WORK_BLOCKS = nnzA / (productNnzPerThread * THREADS_PER_BLOCK) + ((nnzA % (productNnzPerThread * THREADS_PER_BLOCK) == 0) ? 0 : 1);

  const int startRowPerBlock_len = WORK_BLOCKS + 1;

  int *startRowPerBlock;
  cudaMalloc(&startRowPerBlock, sizeof(int) * startRowPerBlock_len);
  cudaMemset(startRowPerBlock, 0, sizeof(int) * startRowPerBlock_len);

  int productNnzPerBlock = THREADS_PER_BLOCK * productNnzPerThread;


  timer.start();
  pre_startRowPerTile<<<divup<uint32_t>(rowA + 1, 128), 128>>>(d_ptr, rowA,
                                                               startRowPerBlock,
                                                               productNnzPerBlock);
  *cdPre = (double)timer.stop();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("pre_startRowPerTile kernel launch failed: %s\n", cudaGetErrorString(err));
  }
  int warm_iter = 200;
  int test_iter = 4000;
  cudaMemset(d_vecY_perf, 0.0, sizeof(valT) * rowA);

  for (int i = 0; i < warm_iter; ++i)
  {
    preFreeSpMV_kernel_bench<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_perf, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
  }
  cudaDeviceSynchronize();
  timer.start();
  for (int i = 0; i < test_iter; ++i)
  {
    preFreeSpMV_kernel_bench<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_perf, startRowPerBlock, productNnzPerThread, productNnzPerBlock);
  }
  float total_loop_time_ms = timer.stop();

  cudaMemset(d_vecY_accu, 0.0, sizeof(valT) * rowA);
  preFreeSpMV_kernel_bench<THREADS_PER_BLOCK><<<(WORK_BLOCKS), (THREADS_PER_BLOCK), productNnzPerBlock * sizeof(valT)>>>(d_val, d_ptr, d_indices, rowA, d_vecX, d_vecY_accu, startRowPerBlock, productNnzPerThread, productNnzPerBlock);

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
