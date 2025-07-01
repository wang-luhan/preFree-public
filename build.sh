#!/bin/bash

rm -rf build
mkdir build
cd build
cmake ..
make -j


#./cuda_perftest ../../../rootdata/mtx/msdoor/msdoor.mtx
#./cuda_perftest ../../../rootdata/mtx/road_central/road_central.mtx
#./cuda_perftest ../../../rootdata/mtx/wikipedia-20060925/wikipedia-20060925.mtx
#./cuda_perftest ../../../rootdata/mtx/inline_1/inline_1.mtx
#./cuda_perftest ../../../rootdata/mtx/F1/F1.mtx
# ./cuda_perftest ../../../rootdata/mtx/Geo_1438/Geo_1438.mtx
# ./cuda_perftest ../../../rootdata/mtx/Flan_1565/Flan_1565.mtx
# ./cuda_perftest ../../../rootdata/mtx/europe_osm/europe_osm.mtx
# ./cuda_perftest ../../../rootdata/mtx/cage15/cage15.mtx
# ./cuda_perftest ../../../rootdata/mtx/ldoor/ldoor.mtx
# ./cuda_perftest ../../../rootdata/mtx/af_shell10/af_shell10.mtx
# ./cuda_perftest ../../../rootdata/mtx/msdoor/msdoor.mtx
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





