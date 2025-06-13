#!/bin/bash

rm -rf build
mkdir build
cd build
cmake ..
make -j

./cuda_perftest ../../../rootdata/mtx/amazon-2008/amazon-2008.mtx
./cuda_perftest ../../../rootdata/mtx/road_central/road_central.mtx
./cuda_perftest ../../../rootdata/mtx/cit-Patents/cit-Patents.mtx
./cuda_perftest ../../../rootdata/mtx/Cube_Coup_dt0/Cube_Coup_dt0.mtx
#./cuda_perftest ../../../rootdata/mtx/nlpkkt240/nlpkkt240.mtx
./cuda_perftest ../../../rootdata/mtx/rgg_n_2_23_s0/rgg_n_2_23_s0.mtx
#./cuda_perftest ../../../rootdata/mtx/af_shell10/af_shell10.mtx
#./cuda_perftest ../../../rootdata/mtx/uk-2002/uk-2002.mtx


