#!/bin/bash

rm -rf build
mkdir build
cd build
cmake ..
make -j


./cuda_perftest ../../../rootdata/mtx/poisson3Db/poisson3Db.mtx
./cuda_perftest ../../../rootdata/mtx/mouse_gene/mouse_gene.mtx
./cuda_perftest ../../../rootdata/mtx/web-Google/web-Google.mtx
./cuda_perftest ../../../rootdata/mtx/web-Stanford/web-Stanford.mtx
./cuda_perftest ../../../rootdata/mtx/cnr-2000/cnr-2000.mtx
./cuda_perftest ../../../rootdata/mtx/cit-Patents/cit-Patents.mtx
./cuda_perftest ../../../rootdata/mtx/Freescale1/Freescale1.mtx
./cuda_perftest ../../../rootdata/mtx/circuit5M/circuit5M.mtx
./cuda_perftest ../../../rootdata/mtx/uk-2002/uk-2002.mtx
./cuda_perftest ../../../rootdata/mtx/soc-LiveJournal1/soc-LiveJournal1.mtx
./cuda_perftest ../../../rootdata/mtx/nlpkkt240/nlpkkt240.mtx
./cuda_perftest ../../../rootdata/mtx/audikw_1/audikw_1.mtx





