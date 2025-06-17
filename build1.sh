#!/bin/bash

# 清理并构建项目
echo "Building project..."
rm -rf build
mkdir build
cd build
cmake ..
make -j

# 检查构建是否成功
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"

# 定义测试矩阵列表
matrices=(
    "../../../rootdata/mtx/poisson3Db/poisson3Db.mtx"
    "../../../rootdata/mtx/circuit5M/circuit5M.mtx"
    "../../../rootdata/mtx/nlpkkt240/nlpkkt240.mtx"
    "../../../rootdata/mtx/audikw_1/audikw_1.mtx"
    "../../../rootdata/mtx/mouse_gene/mouse_gene.mtx"
    "../../../rootdata/mtx/Freescale1/Freescale1.mtx"
    "../../../rootdata/mtx/web-Google/web-Google.mtx"
    "../../../rootdata/mtx/web-Stanford/web-Stanford.mtx"
    "../../../rootdata/mtx/cnr-2000/cnr-2000.mtx"
    "../../../rootdata/mtx/cit-Patents/cit-Patents.mtx"
    "../../../rootdata/mtx/uk-2002/uk-2002.mtx"
    "../../../rootdata/mtx/soc-LiveJournal1/soc-LiveJournal1.mtx"
)

# 创建CSV文件并写入标题
csv_file="../performance_results_dynamic_25666666.csv"
echo "Matrix,Our_Perf(ms),cuSPAESE_Perf(ms),Our_Pre(ms),cuSPAESE_Pre(ms),Elements" > $csv_file

echo ""
echo "Running performance tests..."
echo "Results will be saved to: $csv_file"
echo ""

# 运行测试并解析结果
for matrix in "${matrices[@]}"; do
    echo "Testing: $matrix"
    
    # 运行测试并捕获输出
    output=$(./cuda_perftest "$matrix" 2>&1)
    
    # 检查测试是否成功运行
    if [ $? -ne 0 ]; then
        echo "Error running test for $matrix"
        continue
    fi
    
    # 提取矩阵名称（去掉路径和扩展名）
    matrix_name=$(basename "$matrix" .mtx)
    
    # 使用正则表达式提取性能数据
    our_perf=$(echo "$output" | grep -o "our_perf: *[0-9.]*" | grep -o "[0-9.]*")
    our_pre=$(echo "$output" | grep -o "our_pre: *[0-9.]*" | grep -o "[0-9.]*")
    cusparse_perf=$(echo "$output" | grep -o "cusparse_perf: *[0-9.]*" | grep -o "[0-9.]*")
    cusparse_pre=$(echo "$output" | grep -o "cusparse_pre: *[0-9.]*" | grep -o "[0-9.]*")
    elements=$(echo "$output" | grep -o "for [0-9]* elements" | grep -o "[0-9]*")
    
    # 验证是否成功提取所有数据
    if [[ -n "$our_perf" && -n "$our_pre" && -n "$cusparse_perf" && -n "$cusparse_pre" && -n "$elements" ]]; then
        # 将结果写入CSV
        echo "$matrix_name,$our_perf,$cusparse_perf,$our_pre,$cusparse_pre,$elements" >> $csv_file
        echo "  ✓ Our: ${our_perf}ms, cuSPAESE: ${cusparse_perf}ms"
    else
        echo "  ✗ Failed to parse results for $matrix_name"
        echo "$matrix_name,ERROR,ERROR,ERROR,ERROR,ERROR" >> $csv_file
    fi
    
    echo ""
done

echo "Performance testing completed!"
echo "Results saved to: $(pwd)/$csv_file"
echo ""
echo "CSV Summary:"
echo "============"
column -t -s, $csv_file


echo ""
echo "Detailed results: $csv_file"