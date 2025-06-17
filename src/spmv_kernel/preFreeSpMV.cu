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

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ void
red_row_thread_dynamic(int NNZ_PER_BLOCK, const int tid_in_block, const int block_id,
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
  if (tid_in_block == 0)
  {
    next_row_offset = 0;
  }
  __syncthreads();

  const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);

  // 2. 动态获取任务的循环
  while (true)
  {
    int my_row_offset = -1;

    // 只有Vector的"队长"线程（例如，ID为0的线程）去获取任务
    if (tid_in_vec == 0)
    {
      my_row_offset = atomicAdd(&next_row_offset, 1);
    }

    // 使用__shfl_sync或共享内存将队长获取的 my_row_offset 广播给Vector内的其他线程
    // 对于Warp内（VEC_SIZE <= 32），__shfl_sync 是最高效的方式
    my_row_offset = __shfl_sync(0xffffffff, my_row_offset, 0, VECTOR_SIZE);

    int reduce_row_id = reduceStartRowId + my_row_offset;

    // 3. 检查任务是否已经取完
    if (reduce_row_id >= reduceEndRowId)
    {
      break; // 所有行都已被处理，退出循环
    }

    // 4. 执行计算 (与您之前的逻辑相同)
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);

    valT sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += VECTOR_SIZE)
    {
      sum += smem[i];
    }

    sum = warpReduceSum<VECTOR_SIZE>(sum); // 使用Warp内reduce求和

    if (tid_in_vec == 0)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
  }
}

// ===================================================================================
//      >>> 全新重构，极致性能的 red_row_vector_dynamic_uneven <<<
// ===================================================================================

// 定义一个“任务”结构体，用于统一表示所有工作
struct SpmvTask
{
  int row_id;    // 任务所属的原始行ID
  int start_idx; // 任务在smem中的起始索引
  int end_idx;   // 任务在smem中的结束索引
};

template <
    int THREADS_PER_BLOCK,
    int VECTOR_SIZE,
    // CHUNK_SIZE 定义了长行切分的基本粒度，也是区分长短行的阈值
    int CHUNK_SIZE = 32,
    // 线程块能处理的最大任务数，影响共享内存使用量
    // 这个值需要根据矩阵特性和资源限制进行调整
    // (reduceEndRowId - reduceStartRowId) + (最长行的长度 / CHUNK_SIZE)
    int MAX_TASKS_PER_BLOCK = 128>
__device__ __forceinline__ void
red_row_vector_dynamic_uneven(
    int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
    const int reduceStartRowId, const int reduceEndRowId,
    const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  // =================================================================
  // 阶段 1: 共享内存初始化 和 任务队列定义
  // =================================================================

  // 任务队列，存放所有需要处理的工作单元
  __shared__ SpmvTask s_tasks[MAX_TASKS_PER_BLOCK];
  // 任务队列的头指针（任务生成时使用）
  __shared__ int s_task_queue_head;
  // 当前待领取的任务ID（任务执行时使用）
  __shared__ int s_task_fetch_idx;

  const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);

  if (tid_in_block == 0)
  {
    s_task_queue_head = 0;
    s_task_fetch_idx = 0;
  }
  __syncthreads();

  // =================================================================
  // 阶段 2: 合作式任务生成 (Cooperative Task Generation)
  // 线程块内所有线程并行地扫描行，并生成任务列表
  // =================================================================

  // 使用grid-stride loop的方式让所有线程参与任务生成
  for (int row_idx = reduceStartRowId + tid_in_block; row_idx < reduceEndRowId; row_idx += THREADS_PER_BLOCK)
  {
    const int row_start = max(0, row_ptr[row_idx] - block_id * NNZ_PER_BLOCK);
    const int row_end = min(NNZ_PER_BLOCK, row_ptr[row_idx + 1] - block_id * NNZ_PER_BLOCK);
    const int row_len = row_end - row_start;

    if (row_len <= 0)
      continue;

    // 如果是“短行”，则生成一个任务
    if (row_len <= CHUNK_SIZE)
    {
      int task_idx = atomicAdd(&s_task_queue_head, 1);
      if (task_idx < MAX_TASKS_PER_BLOCK)
      {
        s_tasks[task_idx] = {row_idx, row_start, row_end};
      }
    }
    // 如果是“长行”，则切分成多个任务
    else
    {
      for (int offset = 0; offset < row_len; offset += CHUNK_SIZE)
      {
        int task_idx = atomicAdd(&s_task_queue_head, 1);
        if (task_idx < MAX_TASKS_PER_BLOCK)
        {
          s_tasks[task_idx] = {row_idx, row_start + offset, min(row_start + offset + CHUNK_SIZE, row_end)};
        }
      }
    }
  }
  // 必须同步，确保所有线程都完成了任务生成，s_tasks 和 s_task_queue_head 都已就绪
  __syncthreads();

  // =================================================================
  // 阶段 3: 动态任务拾取与执行 (Dynamic Task Execution)
  // =================================================================

  const int total_tasks = s_task_queue_head;

  while (true)
  {
    int my_task_idx = -1;

    // Vector的队长线程(leader)负责获取任务ID
    if (tid_in_vec == 0)
    {
      my_task_idx = atomicAdd(&s_task_fetch_idx, 1);
    }
    // 将任务ID广播给Vector内的所有线程
    my_task_idx = __shfl_sync(0xffffffff, my_task_idx, 0, VECTOR_SIZE);

    // 如果任务已经取完，则退出循环
    if (my_task_idx >= total_tasks)
    {
      break;
    }

    // Vector内的所有线程从共享内存中读取同一个任务
    const SpmvTask my_task = s_tasks[my_task_idx];

    // --- 开始执行计算 ---
    valT sum = 0;
    for (int i = my_task.start_idx + tid_in_vec; i < my_task.end_idx; i += VECTOR_SIZE)
    {
      sum += smem[i];
    }

    // 高效的Warp内归约
    sum = warpReduceSum<VECTOR_SIZE>(sum);

    // Vector队长线程将归约结果原子地加到最终输出向量y
    if (tid_in_vec == 0)
    {
      atomicAdd(y + my_task.row_id, sum);
    }
  }
}

template <int THREADS_PER_BLOCK>
__device__ __forceinline__ bool
is_imbalanced(const int reduceStartRowId, const int n_reduce_rows_num,
              const int productNnzPerBlock, const int *__restrict__ d_ptr,
              const int tid_in_block)
{
  __shared__ bool is_imbalanced_decision;
  if (tid_in_block == 0)
  {
    is_imbalanced_decision = false;
  }
  const int warp_id = tid_in_block >> 5;
  if (warp_id == 0) // warp-level decision
  {
    const int lane_id = tid_in_block & 31;

    const int k_samples = min(n_reduce_rows_num, 32);
    const unsigned int active_mask = (k_samples == 32) ? 0xffffffff : (1U << k_samples) - 1;
    int my_nnz;
    if (lane_id < k_samples)
    {
      // K点等距采样
      const int my_row_idx_offset = (k_samples <= 1) ? 0 : ((long long)lane_id * (n_reduce_rows_num - 1)) / (k_samples - 1);
      const int my_row_idx = reduceStartRowId + my_row_idx_offset;
      my_nnz = d_ptr[my_row_idx + 1] - d_ptr[my_row_idx];
    }
    else
    {
      my_nnz = 0;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      my_nnz = max(my_nnz, __shfl_down_sync(active_mask, my_nnz, offset));
    }
    const int max_nnz = __shfl_sync(active_mask, my_nnz, 0);
    if (lane_id == 0)
    {
      const float len_exp = (float)productNnzPerBlock / n_reduce_rows_num;
      const float THRESHOLD_RATIO = 10.0f;
      if ((float)max_nnz > THRESHOLD_RATIO * len_exp)
      {
        is_imbalanced_decision = true;
      }
    }
  }
  __syncthreads();
  return is_imbalanced_decision;
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
    red_row_vector_dynamic_uneven<THREADS_PER_BLOCK, VECTOR_SIZE>(
        productNnzPerBlock, n_reduce_rows_num, tid_in_block, block_id,
        reduceStartRowId, reduceEndRowId, d_ptr, smem, y);
  }
  else
  {
    red_row_vector_dynamic<THREADS_PER_BLOCK, VECTOR_SIZE>(
        productNnzPerBlock, n_reduce_rows_num, tid_in_block, block_id,
        reduceStartRowId, reduceEndRowId, d_ptr, smem, y);
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
  bool use_uneven = is_imbalanced<THREADS_PER_BLOCK>(reduceStartRowId, n_reduce_rows_num, productNnzPerBlock, d_ptr, threadIdx.x);

  if (n_reduce_rows_num > 64)
  {

    red_row_thread_dynamic<THREADS_PER_BLOCK>(productNnzPerBlock, threadIdx.x, blockIdx.x,
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

  else
  {
    // 根据行数选择一个合适的VECTOR_SIZE
    int vector_size_selector;
    if (n_reduce_rows_num <= 4)
      vector_size_selector = 32;
    else if (n_reduce_rows_num <= 8)
      vector_size_selector = 16;
    else if (n_reduce_rows_num <= 16)
      vector_size_selector = 8;
    else if (n_reduce_rows_num <= 32)
      vector_size_selector = 4;
    else
      vector_size_selector = 2; // for n_reduce_rows_num <= 64

    // 使用switch语句分发，代码清晰，且易于编译器优化
    switch (vector_size_selector)
    {
    case 32:
      dispatch_reduction_strategy<THREADS_PER_BLOCK, 32>(
          use_uneven, productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 16:
      dispatch_reduction_strategy<THREADS_PER_BLOCK, 16>(
          use_uneven, productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 8:
      dispatch_reduction_strategy<THREADS_PER_BLOCK, 8>(
          use_uneven, productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 4:
      dispatch_reduction_strategy<THREADS_PER_BLOCK, 4>(
          use_uneven, productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 2:
      dispatch_reduction_strategy<THREADS_PER_BLOCK, 2>(
          use_uneven, productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    }
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
  /*
  #ifdef fp64
    int productNnzPerThread = (nnzA > 200000000) ? 8 : 4;
  #else
    int productNnzPerThread = (nnzA > 300000) ? 16 : 4;
  #endif
  */
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
  /*
    int *h_startRowPerBlock = new int[startRowPerBlock_len];
    cudaMemcpy(h_startRowPerBlock, startRowPerBlock, sizeof(int) * startRowPerBlock_len, cudaMemcpyDeviceToHost);
    std::ofstream csv_file("n_reduce_rows_num_Patents.csv");
    csv_file << "block_id,n_reduce_rows_num\n"; // CSV头部
     for (int block_id = 0; block_id < WORK_BLOCKS; block_id++) {
      // 重现kernel中的计算逻辑
      int reduceStartRowId = min(h_startRowPerBlock[block_id], rowA);
      int reduceEndRowId = min(h_startRowPerBlock[block_id + 1], rowA);
      reduceEndRowId = (reduceEndRowId == 0) ? rowA : reduceEndRowId;

      if (csrRowPtr[reduceEndRowId] % productNnzPerBlock != 0 || reduceEndRowId == reduceStartRowId) {
        reduceEndRowId = min(reduceEndRowId + 1, rowA);
      }

      int n_reduce_rows_num = reduceEndRowId - reduceStartRowId;

      // 写入CSV文件
      csv_file << block_id << "," << n_reduce_rows_num << "\n";
    }
    csv_file.close();
    delete[] h_startRowPerBlock;
    printf("n_reduce_rows_num data saved to n_reduce_rows_num.csv\n");
  */

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
