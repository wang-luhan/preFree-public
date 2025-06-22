#include "common.h"
const float THRESH_MAX_AVG = 30.0f;
#define CHUNK 8
#define TASK 128
/*
__global__ void pre_startRowPerTile(const int *__restrict__ row_ptr,
                                    const int m,
                                    int *__restrict__ startRowPerBlock,
                                    bool *__restrict__ d_is_block_imbalanced,
                                    int tileSize,
                                    int long_row_threshold)
{
  const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_thread_id >= m)
  {
    return;
  }
  int a = row_ptr[global_thread_id];
  int b = row_ptr[min(global_thread_id + 1, (int)m)];
  const int row_len = b - a;
  int blocka = divup<int>(a, tileSize);
  int blockb = (b - 1) / static_cast<int>(tileSize);
  if (a != b)
  {
    for (; blocka <= blockb; ++blocka)
    {
      startRowPerBlock[blocka] = global_thread_id;
    }
  }
  ////////////////////////////////////////////////////////////////////////
  if (row_len > long_row_threshold)
  {
    // 我们需要精确计算超长行所跨越的块区间
    int start_block_for_flagging = a / tileSize;
    int end_block_for_flagging = (b - 1) / tileSize;

    if (a != b)
    { // 再次确认行不为空
      for (int i = start_block_for_flagging; i <= end_block_for_flagging; ++i)
      {
        // 良性竞态：多个线程写入true，结果依然是true，无需原子操作
        d_is_block_imbalanced[i] = true;
      }
    }
  }
}
*/

/**
 * @brief 最终合并版预处理核函数
 *
 * 严格保留两种不同的起始块计算逻辑，并将它们合并到单一循环中，
 * 解决了代码冗余，同时保证了与您后续计算逻辑的兼容性。
 */
__global__ void pre_startRowPerTile(const int *__restrict__ row_ptr,
                                    const int m,
                                    int *__restrict__ startRowPerBlock,
                                    bool *__restrict__ d_is_block_imbalanced,
                                    int tileSize,
                                    int long_row_threshold)
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
  const int sRPB_start_block = divup<int>(a, tileSize);
  const int flag_start_block = a / tileSize;
  const int end_block = (b - 1) / tileSize;
  const bool is_long_row = (b - a) > long_row_threshold;
  for (int i = min(sRPB_start_block, flag_start_block); i <= end_block; ++i)
  {
    if (i >= sRPB_start_block)
    {
      startRowPerBlock[i] = row_id;
    }
    if (is_long_row && i >= flag_start_block)
    {
      d_is_block_imbalanced[i] = true;
    }
  }
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

template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
red_row_vector_inwarp(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                      const int reduceStartRowId, const int reduceEndRowId,
                      const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  const int vec_num = THREADS_PER_BLOCK / VECTOR_SIZE;
  const int vec_id = tid_in_block / VECTOR_SIZE;
  const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);

  int reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);
    // reduce LDS via vectors.
    valT sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += VECTOR_SIZE)
    {
      sum += smem[i];
    }
    sum = warpReduceSum<VECTOR_SIZE>(sum);
    // store value
    if (tid_in_vec == 0)
    {
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
}


// 需要一个宏来定义一个Block能处理的最大行数，这决定了共享内存的开销
#define MAX_ROWS_PER_BLOCK 128
template <int THREADS_PER_BLOCK, int VECTOR_SIZE>
__device__ __forceinline__ void
red_row_vector_inwarp_prefetched(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                                    const int reduceStartRowId, const int reduceEndRowId,
                                    const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
    // --- 新增：为“零冗余预取”准备的共享内存 ---
    // 我们需要n_reduce_rows_num个行，所以需要n_reduce_rows_num+1个指针
    __shared__ int s_cached_ptr[MAX_ROWS_PER_BLOCK + 1];
    
    // --- 1. 高效的“零冗余”协同预取阶段 ---
    // Block内的所有线程参与，以合并方式，精确地只读取一次所需要的数据
    for (int i = tid_in_block; i < n_reduce_rows_num + 1; i += THREADS_PER_BLOCK) {
        s_cached_ptr[i] = row_ptr[reduceStartRowId + i];
    }

    // 必须同步，确保所有行指针都已加载到共享内存
    __syncthreads();

    // --- 2. 计算阶段 (无全局访存延迟，无冗余) ---
    const int vec_num = THREADS_PER_BLOCK / VECTOR_SIZE;
    const int vec_id = tid_in_block / VECTOR_SIZE;
    const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);

    for (int row_offset = vec_id; row_offset < n_reduce_rows_num; row_offset += vec_num)
    {
        const int reduce_row_id = reduceStartRowId + row_offset;
        
        // 从共享内存中获取行指针，并计算在smem中的相对索引
        const int row_start_ptr = s_cached_ptr[row_offset];
        const int row_end_ptr   = s_cached_ptr[row_offset + 1];
        
        const int reduce_start_idx = max((int)0, row_start_ptr - block_id * NNZ_PER_BLOCK);
        const int reduce_end_idx   = min((int)NNZ_PER_BLOCK, row_end_ptr - block_id * NNZ_PER_BLOCK);
        
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


template <
    int THREADS_PER_BLOCK,
    int VECTOR_SIZE,
    // CHUNK_SIZE 定义了长行切分的基本粒度
    int CHUNK_SIZE = CHUNK,
    // 线程块能处理的最大任务数
    int MAX_TASKS_PER_BLOCK = TASK>
__device__ __forceinline__ void
red_row_vector_2(
    int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
    const int reduceStartRowId, const int reduceEndRowId,
    const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  // =================================================================
  // 阶段 1: 共享内存声明与初始化
  // =================================================================

  // 任务队列，存放所有需要处理的工作单元
  __shared__ SpmvTask s_tasks[MAX_TASKS_PER_BLOCK];
  // 任务队列的头指针（任务生成时使用）
  __shared__ int s_task_queue_head;
  // 当前待领取的任务ID（任务执行时使用）
  __shared__ int s_task_fetch_idx;
  // 【优化点】: 用于块内结果预聚合的部分和数组
  __shared__ valT s_partial_sums[MAX_ROWS_PER_BLOCK];

  // 由块内第一个线程初始化任务队列指针
  if (tid_in_block == 0)
  {
    s_task_queue_head = 0;
    s_task_fetch_idx = 0;
  }

  // 所有线程并行初始化部分和数组
  // 确保 n_reduce_rows_num 不超过 MAX_ROWS_PER_BLOCK 是调用者的责任
  for (int i = tid_in_block; i < n_reduce_rows_num; i += THREADS_PER_BLOCK)
  {
    s_partial_sums[i] = 0.0;
  }
  __syncthreads();

  // =================================================================
  // 阶段 2: 【优化点】单Warp合作式任务生成
  // =================================================================

  const int warp_id = tid_in_block >> 5; // 计算当前线程所属的Warp ID

  // 仅委托第一个Warp (warp_id == 0) 来执行任务生成
  if (warp_id == 0)
  {
    const int lane_id = tid_in_block & 31; // Warp内的线程ID (0-31)

    // 第一个Warp内的32个线程合作，扫描所有行并生成任务
    for (int row_idx = reduceStartRowId + lane_id; row_idx < reduceEndRowId; row_idx += 32)
    {
      const int row_start = max(0, row_ptr[row_idx] - block_id * NNZ_PER_BLOCK);
      const int row_end = min(NNZ_PER_BLOCK, row_ptr[row_idx + 1] - block_id * NNZ_PER_BLOCK);
      const int row_len = row_end - row_start;

      if (row_len <= 0)
        continue;

      // 短行：生成一个任务
      if (row_len <= CHUNK_SIZE)
      {
        // atomicAdd争用被限制在32个线程内，开销大幅降低
        int task_idx = atomicAdd(&s_task_queue_head, 1);
        if (task_idx < MAX_TASKS_PER_BLOCK)
        {
          s_tasks[task_idx] = {row_idx, row_start, row_end};
        }
      }
      // 长行：切分成多个任务
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
  }

  // 必须同步，确保任务生成阶段完成，s_tasks 和 s_task_queue_head 对所有线程可见
  __syncthreads();

  // =================================================================
  // 阶段 3: 动态任务拾取与执行
  // =================================================================
  const int total_tasks = s_task_queue_head;
  const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);

  while (true)
  {
    int my_task_idx = -1;

    // Vector的leader线程负责获取任务ID
    if (tid_in_vec == 0)
    {
      my_task_idx = atomicAdd(&s_task_fetch_idx, 1);
    }
    my_task_idx = __shfl_sync(0xffffffff, my_task_idx, 0, VECTOR_SIZE);

    if (my_task_idx >= total_tasks)
    {
      break; // 所有任务完成
    }

    const SpmvTask my_task = s_tasks[my_task_idx];

    valT sum = 0;
    for (int i = my_task.start_idx + tid_in_vec; i < my_task.end_idx; i += VECTOR_SIZE)
    {
      sum += smem[i];
    }

    sum = warpReduceSum<VECTOR_SIZE>(sum);

    // 【优化点】: 将结果原子地累加到【共享内存】中
    if (tid_in_vec == 0)
    {
      int relative_row_idx = my_task.row_id - reduceStartRowId;
      atomicAdd(&s_partial_sums[relative_row_idx], sum);
    }
  }
  __syncthreads(); // 确保所有任务对 s_partial_sums 的写操作都已完成

  // =================================================================
  // 阶段 4: 【优化点】最终结果并行写回
  // =================================================================

  // 所有线程合作，将共享内存中的最终聚合结果写回到全局内存y
  for (int row_offset = tid_in_block; row_offset < n_reduce_rows_num; row_offset += THREADS_PER_BLOCK)
  {
    const valT final_sum = s_partial_sums[row_offset];

    // 如果这个行的部分和为0，则无需写回，减少无效内存访问
    if (final_sum == 0.0)
    {
      continue;
    }

    const int final_row_id = reduceStartRowId + row_offset;

    // 只有当行是整个矩阵的边界行，或者可能被多个Block处理的行时，才使用昂贵的全局原子操作
    // 否则，直接写入即可
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
red_row_vector_crosswarp(int NNZ_PER_BLOCK, const int n_reduce_rows_num, const int tid_in_block, const int block_id,
                         const int reduceStartRowId, const int reduceEndRowId,
                         const int *__restrict__ row_ptr, const valT *__restrict__ smem, valT *__restrict__ y)
{
  const int vec_num = THREADS_PER_BLOCK / VECTOR_SIZE;
  const int vec_id = tid_in_block / VECTOR_SIZE;
  const int tid_in_vec = tid_in_block & (VECTOR_SIZE - 1);
  const int warp_lane_id = tid_in_block & 31;

  int reduce_row_id = reduceStartRowId + vec_id;
  for (; reduce_row_id < reduceEndRowId; reduce_row_id += vec_num)
  {
    const int reduce_start_idx = max((int)0, row_ptr[reduce_row_id] - block_id * NNZ_PER_BLOCK);
    const int reduce_end_idx = min((int)NNZ_PER_BLOCK, row_ptr[reduce_row_id + 1] - block_id * NNZ_PER_BLOCK);

    valT sum = 0;
    for (int i = reduce_start_idx + tid_in_vec; i < reduce_end_idx; i += VECTOR_SIZE)
    {
      sum += smem[i];
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
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

    ///////////////////////////////////////////////////////////////////////////////////////////////

    sum = warpReduceSum<VECTOR_SIZE>(sum); // 使用Warp内reduce求和

    if (tid_in_vec == 0)
    {
      atomicAdd(y + reduce_row_id, sum);
    }
  }
}

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
    int CHUNK_SIZE = CHUNK,
    // 线程块能处理的最大任务数，影响共享内存使用量
    // 这个值需要根据矩阵特性和资源限制进行调整
    // (reduceEndRowId - reduceStartRowId) + (最长行的长度 / CHUNK_SIZE)
    int MAX_TASKS_PER_BLOCK = TASK>
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
      if ((float)max_nnz > THRESH_MAX_AVG * len_exp)
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
__device__ __forceinline__ unsigned int
calculate_vector_size_round_up(int n_reduce_rows_num)
{
  if (n_reduce_rows_num <= 0)
    return THREADS_PER_BLOCK;
  unsigned int v = THREADS_PER_BLOCK / n_reduce_rows_num;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return min(32, max(2, v));
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
  // 阶段1：MA - 乘加并将中间结果存入共享内存 (与您原来代码一致)
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

  // -----------------------------------------------------------------------------------
  // 阶段2：Reduction - 规约 (*** 这是被优化的核心部分 ***)
  // -----------------------------------------------------------------------------------

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

    red_row_thread_dynamic<THREADS_PER_BLOCK>(productNnzPerBlock, threadIdx.x, blockIdx.x,
                                      reduceStartRowId, reduceEndRowId,
                                      d_ptr, middle_s, d_y);
  }
  // --- 新的、简化的分发逻辑 ---
  else if (n_reduce_rows_num == 1)
  {
    // 情况1: Block内所有非零元属于同一行。这是最优情况，使用最高效的块内规约。
    red_row_block<THREADS_PER_BLOCK>(productNnzPerBlock, threadIdx.x, blockIdx.x,
                                     reduceStartRowId,
                                     d_ptr, middle_s, d_y);
  }
  else
  {
    // 情况2: Block内非零元属于多行。
    // 我们将 red_row_vector_dynamic 提升为处理所有此类情况的通用、高效策略。
    // 它通过Vector级的动态任务拾取，本身就具备了很好的负载均衡能力，且开销远小于uneven策略。

    // 动态计算一个合适的Vector Size (沿用您的优秀逻辑)
    unsigned int vector_size_selector = calculate_vector_size<THREADS_PER_BLOCK>(n_reduce_rows_num);

    // 我们不再需要 is_imbalanced 判断和 uneven 策略，直接分发
    switch (vector_size_selector)
    {
    case 32:
      red_row_vector_inwarp_prefetched<THREADS_PER_BLOCK, 32>(
          productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 16:
      red_row_vector_inwarp_prefetched<THREADS_PER_BLOCK, 16>(
          productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 8:
      red_row_vector_inwarp_prefetched<THREADS_PER_BLOCK, 8>(
          productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 4:
      red_row_vector_inwarp_prefetched<THREADS_PER_BLOCK, 4>(
          productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    case 2:
      red_row_vector_inwarp_prefetched<THREADS_PER_BLOCK, 2>(
          productNnzPerBlock, n_reduce_rows_num, threadIdx.x, blockIdx.x,
          reduceStartRowId, reduceEndRowId, d_ptr, middle_s, d_y);
      break;
    }
  }
  // 注意：n_reduce_rows_num <= 0 的情况自然跳过，无需处理。
}

template <int THREADS_PER_BLOCK>
__global__ void preFreeSpMV_kernel_hpc(valT *__restrict__ d_val,
                                       int *__restrict__ d_ptr,
                                       int *__restrict__ d_cols,
                                       int rowA,
                                       valT *__restrict__ d_x,
                                       valT *__restrict__ d_y,
                                       int *__restrict__ startRowPerBlock,
                                       bool *__restrict__ d_is_block_imbalanced,
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
  // bool use_uneven = is_imbalanced<THREADS_PER_BLOCK>(reduceStartRowId, n_reduce_rows_num, productNnzPerBlock, d_ptr, threadIdx.x);
  bool use_uneven = d_is_block_imbalanced[blockIdx.x];

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

  else //(2, 64]
  {
    unsigned int vector_size_selector = calculate_vector_size<THREADS_PER_BLOCK>(n_reduce_rows_num);
    /*
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
    */
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

  bool *d_is_block_imbalanced;
  cudaMalloc(&d_is_block_imbalanced, sizeof(bool) * WORK_BLOCKS);
  cudaMemset(d_is_block_imbalanced, 0, sizeof(bool) * WORK_BLOCKS);

  int productNnzPerBlock = THREADS_PER_BLOCK * productNnzPerThread;
  const int LONG_ROW_THRESHOLD = 128;

  timer.start();
  pre_startRowPerTile<<<divup<uint32_t>(rowA + 1, 128), 128>>>(d_ptr, rowA,
                                                               startRowPerBlock,
                                                               d_is_block_imbalanced,
                                                               productNnzPerBlock,
                                                               LONG_ROW_THRESHOLD);
  *cdPre = (double)timer.stop();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("pre_startRowPerTile kernel launch failed: %s\n", cudaGetErrorString(err));
  }

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
