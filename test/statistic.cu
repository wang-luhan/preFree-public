#include "PreFree.h"
#include "mmio.h"
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

struct TileStatistics {
    int tile_id;
    int n_reduce_rows_num;
    int max_row_length;
    int min_row_length;
    double avg_row_length;
    double variance;
    int start_row;
    int end_row;
};

// 从文件路径提取矩阵名称（不包含路径和扩展名）
std::string extractMatrixName(const std::string& filepath) {
    // 找到最后一个'/'或'\'的位置
    size_t last_slash = filepath.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? filepath : filepath.substr(last_slash + 1);
    
    // 移除扩展名
    size_t last_dot = filename.find_last_of(".");
    if (last_dot != std::string::npos) {
        filename = filename.substr(0, last_dot);
    }
    
    return filename;
}

// 计算tile的统计信息
TileStatistics calculateTileStatistics(int tile_id, int start_row, int end_row, 
                                     const int* csrRowPtr, int productNnzPerBlock) {
    TileStatistics stats;
    stats.tile_id = tile_id;
    stats.start_row = start_row;
    stats.end_row = end_row;
    stats.n_reduce_rows_num = end_row - start_row;
    
    if (stats.n_reduce_rows_num <= 0) {
        stats.max_row_length = 0;
        stats.min_row_length = 0;
        stats.avg_row_length = 0.0;
        stats.variance = 0.0;
        return stats;
    }
    
    // 计算tile的非零元素范围
    int tile_nnz_start = tile_id * productNnzPerBlock;
    int tile_nnz_end = (tile_id + 1) * productNnzPerBlock;
    
    // 计算每行在tile内部的长度
    std::vector<int> row_lengths_in_tile(stats.n_reduce_rows_num);
    int total_length = 0;
    
    for (int i = 0; i < stats.n_reduce_rows_num; i++) {
        int row_idx = start_row + i;
        int row_global_start = csrRowPtr[row_idx];
        int row_global_end = csrRowPtr[row_idx + 1];
        
        // 计算该行在当前tile范围内的非零元个数
        int row_start_in_tile = std::max(row_global_start, tile_nnz_start);
        int row_end_in_tile = std::min(row_global_end, tile_nnz_end);
        
        int row_length_in_tile = std::max(0, row_end_in_tile - row_start_in_tile);
        row_lengths_in_tile[i] = row_length_in_tile;
        total_length += row_length_in_tile;
    }
    
    // 计算最大、最小和平均长度
    stats.max_row_length = *std::max_element(row_lengths_in_tile.begin(), row_lengths_in_tile.end());
    stats.min_row_length = *std::min_element(row_lengths_in_tile.begin(), row_lengths_in_tile.end());
    stats.avg_row_length = static_cast<double>(total_length) / stats.n_reduce_rows_num;
    
    // 计算方差
    double variance_sum = 0.0;
    for (int length : row_lengths_in_tile) {
        double diff = length - stats.avg_row_length;
        variance_sum += diff * diff;
    }
    stats.variance = variance_sum / stats.n_reduce_rows_num;
    
    return stats;
}

// GPU kernel用于计算startRowPerBlock（与原算法相同）
__global__ void pre_startRowPerTile_statistic(const int *__restrict__ row_ptr,
                                             const int m,
                                             int *__restrict__ startRowPerBlock,
                                             int tileSize,
                                             int long_row_threshold)
{
    const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= m) {
        return;
    }
    
    const int a = row_ptr[row_id];
    const int b = row_ptr[row_id + 1];
    if (a == b) {
        return;
    }
    
    const int start_block = (a + tileSize - 1) / tileSize;  // divup
    const int end_block = (b - 1) / tileSize;
    
    for (int i = start_block; i <= end_block; ++i) {
        if (i >= start_block) {
            startRowPerBlock[i] = row_id;
        }
    }
}

void analyzeMatrixTiles(const char* filename) {
    printf("正在分析稀疏矩阵文件: %s\n", filename);
    
    // 读取矩阵文件
    int m, n, nnz, isSymmetric;
    int *csrRowPtr, *csrColInd;
    valT *csrVal;
    
    int ret = mmio_allinone(&m, &n, &nnz, &isSymmetric, 
                           &csrRowPtr, &csrColInd, &csrVal, 
                           const_cast<char*>(filename));
    
    if (ret != 0) {
        printf("错误：无法读取矩阵文件 %s\n", filename);
        return;
    }
    
    printf("矩阵信息: %d x %d, nnz = %d, 对称性 = %d\n", m, n, nnz, isSymmetric);
    
    // 设置算法参数（与原算法相同）
    const int THREADS_PER_BLOCK = 128;
    int productNnzPerThread = 4;
    int productNnzPerBlock = THREADS_PER_BLOCK * productNnzPerThread;
    const int WORK_BLOCKS = nnz / productNnzPerBlock + 
                           ((nnz % productNnzPerBlock == 0) ? 0 : 1);
    const int startRowPerBlock_len = WORK_BLOCKS + 1;
    const int LONG_ROW_THRESHOLD = 2048;
    
    printf("算法参数: productNnzPerBlock = %d, WORK_BLOCKS = %d\n", 
           productNnzPerBlock, WORK_BLOCKS);
    
    // 分配GPU内存
    int *d_ptr, *startRowPerBlock, *h_startRowPerBlock;
    
    CUDA_CHECK_ERROR(cudaMalloc(&d_ptr, sizeof(int) * (m + 2)));
    CUDA_CHECK_ERROR(cudaMalloc(&startRowPerBlock, sizeof(int) * startRowPerBlock_len));
    CUDA_CHECK_ERROR(cudaMemcpy(d_ptr, csrRowPtr, sizeof(int) * (m + 2), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemset(startRowPerBlock, 0, sizeof(int) * startRowPerBlock_len));
    
    // 执行tile划分kernel
    pre_startRowPerTile_statistic<<<(m + 127) / 128, 128>>>(
        d_ptr, m, startRowPerBlock, productNnzPerBlock, LONG_ROW_THRESHOLD);
    
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // 将startRowPerBlock拷贝回CPU
    h_startRowPerBlock = new int[startRowPerBlock_len];
    CUDA_CHECK_ERROR(cudaMemcpy(h_startRowPerBlock, startRowPerBlock, 
                                sizeof(int) * startRowPerBlock_len, cudaMemcpyDeviceToHost));
    
    // 计算每个tile的统计信息
    std::vector<TileStatistics> tileStats;
    
    for (int tile_id = 0; tile_id < WORK_BLOCKS; tile_id++) {
        int reduceStartRowId = std::min(h_startRowPerBlock[tile_id], m);
        int reduceEndRowId = std::min(h_startRowPerBlock[tile_id + 1], m);
        
        if (reduceEndRowId == 0) {
            reduceEndRowId = m;
        }
        
        // 应用与原算法相同的逻辑来调整reduceEndRowId
        if (csrRowPtr[reduceEndRowId] % productNnzPerBlock != 0 || 
            reduceEndRowId == reduceStartRowId) {
            reduceEndRowId = std::min(reduceEndRowId + 1, m);
        }
        
        if (reduceStartRowId < m && reduceEndRowId > reduceStartRowId) {
            TileStatistics stats = calculateTileStatistics(
                tile_id, reduceStartRowId, reduceEndRowId, csrRowPtr, productNnzPerBlock);
            tileStats.push_back(stats);
        }
    }
    
    // 创建stat目录（如果不存在）
    system("mkdir -p stat");
    
    // 写入CSV文件
    std::string matrixName = extractMatrixName(filename);
    std::string csvFilename = std::string("stat/") + matrixName + ".csv";
    
    std::ofstream csvFile(csvFilename);
    if (!csvFile.is_open()) {
        printf("错误：无法创建CSV文件 %s\n", csvFilename.c_str());
        goto cleanup;
    }
    
    // 写入CSV头部
    csvFile << "tile_id,n_reduce_rows_num,max_row_length,min_row_length,avg_row_length,variance,start_row,end_row\n";
    
    // 写入每个tile的统计信息
    for (const auto& stats : tileStats) {
        csvFile << stats.tile_id << ","
                << stats.n_reduce_rows_num << ","
                << stats.max_row_length << ","
                << stats.min_row_length << ","
                << std::fixed << std::setprecision(2) << stats.avg_row_length << ","
                << std::fixed << std::setprecision(2) << stats.variance << ","
                << stats.start_row << ","
                << stats.end_row << "\n";
    }
    
    csvFile.close();
    
    printf("分析完成！统计信息已保存到: %s\n", csvFilename.c_str());
    printf("总共分析了 %zu 个tiles\n", tileStats.size());
    
    // 打印一些摘要信息
    if (!tileStats.empty()) {
        int total_tiles = tileStats.size();
        double avg_n_reduce_rows = 0, avg_max_length = 0, avg_variance = 0;
        
        for (const auto& stats : tileStats) {
            avg_n_reduce_rows += stats.n_reduce_rows_num;
            avg_max_length += stats.max_row_length;
            avg_variance += stats.variance;
        }
        
        avg_n_reduce_rows /= total_tiles;
        avg_max_length /= total_tiles;
        avg_variance /= total_tiles;
        
        printf("\n===== 摘要统计 =====\n");
        printf("平均行跨度: %.2f\n", avg_n_reduce_rows);
        printf("平均最大行长度: %.2f\n", avg_max_length);
        printf("平均方差: %.2f\n", avg_variance);
    }
    
cleanup:
    // 清理资源
    CUDA_CHECK_ERROR(cudaFree(d_ptr));
    CUDA_CHECK_ERROR(cudaFree(startRowPerBlock));
    delete[] h_startRowPerBlock;
    free(csrRowPtr);
    free(csrColInd);
    free(csrVal);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("用法: %s <矩阵文件路径>\n", argv[0]);
        printf("示例: %s ../../../rootdata/mtx/poisson3Db/poisson3Db.mtx\n", argv[0]);
        return 1;
    }
    
    // 初始化CUDA设备
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("错误：未找到CUDA设备\n");
        return 1;
    }
    
    CUDA_CHECK_ERROR(cudaSetDevice(0));
    
    printf("PreFree SpMV Tile 结构分析工具\n");
    printf("================================\n");
    
    analyzeMatrixTiles(argv[1]);
    
    return 0;
} 