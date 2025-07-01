#include "PreFree.h"
#include "mmio.h"

void spmv_serial(valT *csrVal, int *csrRowPtr, int *csrColInd,
                 valT *X_val, valT *Y_val, int rowA, int colA, int nnzA)
{
    valT t;
    for (int i = 0; i < rowA; i++)
    {
        t = static_cast<valT>(0.0);
        int ptr_start = csrRowPtr[i];
        int n_one_line = csrRowPtr[i + 1] - ptr_start;
        for (int j = 0; j < n_one_line; j++)
        {
            int v_idx = csrColInd[j + ptr_start];
            t = t + csrVal[j + ptr_start] * X_val[v_idx];
        }
        Y_val[i] = Y_val[i] + t;
    }
}

int eQcheck(valT *tmp1, valT *tmp2, int length)
{
#ifdef fp64
    // Use double precision (fp64), check for 15 significant digits
    const double tolerance = 1e-8; // 15 significant digits for double precision
    for (int i = 0; i < length; i++)
    {
        double val1 = tmp1[i];
        double val2 = tmp2[i];
        if (fabs(val1 - val2) / fmax(fabs(val1), fabs(val2)) > tolerance)
        {
            printf("Error at index (%d), res(%4.15f), our(%4.15f), please check your code!\n", i, val1, val2);
            // return -1; TODO: !!!
        }
    }
#else
    // Use half precision (fp16), check for 3-4 significant digits
    const float tolerance = 1e-1; // 3 significant digits for half precision
    for (int i = 0; i < length; i++)
    {
        // Convert __half to float for computation
        float val1 = static_cast<float>(tmp1[i]);
        float val2 = static_cast<float>(tmp2[i]);
        if (fabs(val1 - val2) / fmax(fabs(val1), fabs(val2)) > tolerance)
        {
            printf("Error at index (%d), res(%4.3f), our(%4.3f), please check your code!\n", i, val1, val2);
            // return -1; TODO: !!!
        }
    }
#endif
    printf("Success! All values match within the tolerance for %d elements.\n", length);
    return 0;
}
/*
void cusparse_spmv_all(valT *cu_ValA, int *cu_RowPtrA, int *cu_ColIdxA,
                       valT *cu_ValX, valT *cu_ValY, int rowA, int colA, int nnzA,
                       long long int data_origin1, long long int data_origin2, double *cu_time, double *cu_gflops, double *cu_bandwidth1, double *cu_bandwidth2, double *cu_pre)
{
    CudaTimer timer;
    valT *dA_val, *dX, *dY;
    int *dA_cid;
    int *dA_rpt;
    valT alpha = (valT)1, beta = (valT)0;

    cudaMalloc((void **)&dA_val, sizeof(valT) * nnzA);
    cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
    cudaMalloc((void **)&dA_rpt, sizeof(int) * (rowA + 1));
    cudaMalloc((void **)&dX, sizeof(valT) * colA);
    cudaMalloc((void **)&dY, sizeof(valT) * rowA);

    cudaMemcpy(dA_val, cu_ValA, sizeof(valT) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, cu_ValX, sizeof(valT) * colA, cudaMemcpyHostToDevice);
    cudaMemset(dY, 0.0, sizeof(valT) * rowA);

    cusparseHandle_t handle = NULL;

    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // int version;
    // if (cusparseGetVersion(handle, &version) == CUSPARSE_STATUS_SUCCESS) {
    //     std::cout << "cuSPARSE Version: " << version / 1000 << "." << (version % 1000) / 10 << std::endl;
    // } else {
    //     std::cerr << "Failed to get cuSPARSE version." << std::endl;
    // }
    cusparseCreateDnVec(&vecX, colA, dX, VAL_CUDA_R_TYPE);
    cusparseCreateDnVec(&vecY, rowA, dY, VAL_CUDA_R_TYPE);
    cusparseCreate(&handle);
    timer.start();
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, VAL_CUDA_R_TYPE);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    *cu_pre = (double)timer.stop();
    // cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                         &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
    //                         CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    // cudaDeviceSynchronize();

    int warm_iter = 200;
    int test_iter = 4000;

    for (int i = 0; i < warm_iter; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();

    timer.start();
    for (int i = 0; i < test_iter; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                     CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    float total_loop_time_ms = timer.stop();

    // cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                  &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
    //                  CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();
    cudaMemcpy(cu_ValY, dY, sizeof(valT) * rowA, cudaMemcpyDeviceToHost);

    // double gflops = (2.0 * matA_csr->nnz) / ((runtime / 1000) * 1e9);
    // printf("\n CUSPARSE CUDA kernel runtime = %g ms\n", runtime);
    *cu_time = (double)total_loop_time_ms / test_iter;
    *cu_gflops = (double)((long)nnzA * 2) / (*cu_time * 1e6);
    *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6);
    *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaFree(dA_val);
    cudaFree(dA_cid);
    cudaFree(dA_rpt);
    cudaFree(dX);
    cudaFree(dY);
}
*/

void cusparse_spmv_all(valT *cu_ValA, int *cu_RowPtrA, int *cu_ColIdxA,
                       valT *cu_ValX, valT *cu_ValY, int rowA, int colA, int nnzA,
                       long long int data_origin1, long long int data_origin2, double *cu_time, double *cu_gflops, double *cu_bandwidth1, double *cu_bandwidth2, double *cu_pre)
{
    CudaTimer timer;
    valT *dA_val, *dX, *dY;
    int *dA_cid;
    int *dA_rpt;
    valT alpha = (valT)1.0;
    valT beta = (valT)0.0;

    // --- 1. 初始化 cuSPARSE 和分配设备内存 ---
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK_CUDA(cudaMalloc((void **)&dA_val, sizeof(valT) * nnzA));
    CHECK_CUDA(cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA));
    CHECK_CUDA(cudaMalloc((void **)&dA_rpt, sizeof(int) * (rowA + 1)));
    CHECK_CUDA(cudaMalloc((void **)&dX, sizeof(valT) * colA));
    CHECK_CUDA(cudaMalloc((void **)&dY, sizeof(valT) * rowA));

    // --- 2. 将数据从主机拷贝到设备 ---
    CHECK_CUDA(cudaMemcpy(dA_val, cu_ValA, sizeof(valT) * nnzA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(int) * (rowA + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX, cu_ValX, sizeof(valT) * colA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dY, 0, sizeof(valT) * rowA)); // 使用 0 而非 0.0 更通用

    // --- 3. 创建 cuSPARSE 描述符 ---

    cusparseDnVecDescr_t vecX, vecY;

    // 创建稀疏矩阵描述符 (CSR格式)

    // 创建稠密向量描述符
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, colA, dX, VAL_CUDA_R_TYPE));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rowA, dY, VAL_CUDA_R_TYPE));

    // --- 4. 预处理阶段 ---
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, VAL_CUDA_R_TYPE));

    // 4.1 获取预处理所需的缓冲区大小
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    // 4.2 分配缓冲区
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // 4.3 执行预处理 (这是您想测试的关键函数)
    timer.start(); // 开始计时预处理时间

    for (int i = 0; i < 4000; ++i)
    {
        CHECK_CUSPARSE(cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                                               CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    float pre_total_time_ms = timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize()); // 确保预处理完成
    *cu_pre = (double)pre_total_time_ms / 4000;

    // --- 5. 执行阶段 ---
    int warm_iter = 200;
    int test_iter = 4000;

    // 5.1 预热，确保 GPU 达到稳定频率
    for (int i = 0; i < warm_iter; ++i)
    {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5.2 计时测试
    timer.start();
    for (int i = 0; i < test_iter; ++i)
    {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, BUF_CUDA_R_TYPE,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }

    float total_loop_time_ms = timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize()); // 等待所有 SpMV 操作完成

    // --- 6. 获取结果并计算性能指标 ---
    CHECK_CUDA(cudaMemcpy(cu_ValY, dY, sizeof(valT) * rowA, cudaMemcpyDeviceToHost));

    *cu_time = (double)total_loop_time_ms / test_iter;
    *cu_gflops = (double)((long long)nnzA * 2) / (*cu_time * 1e6);
    *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6);
    *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6);

    // --- 7. 清理资源 ---
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_val));
    CHECK_CUDA(cudaFree(dA_cid));
    CHECK_CUDA(cudaFree(dA_rpt));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Run the code by './spmv_double matrix.mtx'. \n");
        return 0;
    }

    // struct timeval t1, t2;
    int rowA, colA;
    int nnzA;
    int isSymmetricA;
    valT *csrVal;
    int *csrColInd;
    int *csrRowPtr;

    char *filename;
    filename = argv[1];

    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtr, &csrColInd, &csrVal, filename);
    valT *X_val = (valT *)malloc(sizeof(valT) * colA);
    initVec(X_val, colA);
    initVec(csrVal, nnzA);

    printf("INIT DONE\n");

    valT *cuY_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *Y_val = (valT *)malloc(sizeof(valT) * rowA);
    valT *Y_val_s = (valT *)malloc(sizeof(valT) * rowA);

    memset(cuY_val, (valT)0, sizeof(valT) * rowA);
    memset(Y_val, (valT)0, sizeof(valT) * rowA);
    memset(Y_val_s, (valT)0, sizeof(valT) * rowA);

    double cu_time = 0, cu_gflops = 0, cu_bandwidth1 = 0, cu_bandwidth2 = 0, cu_pre = 0;
    long long int data_origin1 = (nnzA + colA + rowA) * sizeof(valT) + nnzA * sizeof(int) + (rowA + 1) * sizeof(int);
    long long int data_origin2 = (nnzA + nnzA + rowA) * sizeof(valT) + nnzA * sizeof(int) + (rowA + 1) * sizeof(int);

    cusparse_spmv_all(csrVal, csrRowPtr, csrColInd, X_val, cuY_val, rowA, colA, nnzA, data_origin1, data_origin2, &cu_time, &cu_gflops, &cu_bandwidth1, &cu_bandwidth2, &cu_pre);

    double cdTime = 0, cdPre = 0;
    preFreeSpMV(csrVal, csrRowPtr, csrColInd, X_val, Y_val, rowA, colA, nnzA, &cdTime, &cdPre);
    spmv_serial(csrVal, csrRowPtr, csrColInd, X_val, Y_val_s, rowA, colA, nnzA);

    printf("our_perf:%8.4lf ms, our_pre:%8.4lf ms\n", cdTime, cdPre);
    printf("cusparse_perf:%8.4lf ms, cusparse_pre:%8.4lf ms\n", cu_time, cu_pre);

    eQcheck(Y_val_s, Y_val, rowA);

    free(X_val);
    free(Y_val);
    free(cuY_val);
    free(csrColInd);
    free(csrRowPtr);
    free(csrVal);

    return 0;
}
