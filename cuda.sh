# 使用官方卸载脚本
if [ -f /usr/local/cuda-12.0/bin/cuda-uninstaller ]; then
    sudo /usr/local/cuda-12.0/bin/cuda-uninstaller
else
    # 如果是通过apt安装的，则用apt卸载
    sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda-repo-*"
    sudo apt-get autoremove
    sudo apt-get autoclean
    sudo rm -f /etc/apt/sources.list.d/cuda*
fi
