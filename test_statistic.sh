#!/bin/bash

echo "=== PreFree 统计工具编译测试 ==="
echo

# 创建build目录
echo "1. 创建build目录..."
mkdir -p build
cd build

# 运行cmake
echo "2. 运行cmake配置..."
cmake .. -DUSE_FP64=ON

if [ $? -ne 0 ]; then
    echo "错误：cmake配置失败！"
    exit 1
fi

# 编译
echo "3. 编译项目..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "错误：编译失败！"
    exit 1
fi

# 检查可执行文件
echo "4. 检查生成的可执行文件..."
if [ -f "./statistic" ]; then
    echo "✓ statistic 可执行文件生成成功"
    ls -la ./statistic
else
    echo "✗ statistic 可执行文件未生成"
    exit 1
fi

if [ -f "./cuda_perftest" ]; then
    echo "✓ cuda_perftest 可执行文件生成成功"
    ls -la ./cuda_perftest
else
    echo "✗ cuda_perftest 可执行文件未生成"
fi

echo
echo "=== 编译测试完成 ==="
echo "现在可以运行："
echo "  ./statistic <矩阵文件路径>"
echo "例如："
echo "  ./build/statistic ../../rootdata/mtx/poisson3Db/poisson3Db.mtx" 