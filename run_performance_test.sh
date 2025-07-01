#!/bin/bash

cd build

CSV_FILE="../perf/performance_results_complete_2.csv"
echo "Matrix,our_perf,cusparse_perf,our_pre,cusparse_pre" > "$CSV_FILE"

MTX_FILES=""
for dir in /root/rootdata/mtx/*/; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        mtx_file="$dir$folder_name.mtx"
        if [ -f "$mtx_file" ]; then
            MTX_FILES="$MTX_FILES$mtx_file"$'\n'
        fi
    fi
done
MTX_FILES=$(echo "$MTX_FILES" | sort)

echo "Found $(echo "$MTX_FILES" | wc -l) matrix files"

for mtx_file in $MTX_FILES; do
    matrix_name=$(basename "$mtx_file" .mtx)
    
    echo "Testing: $matrix_name"
    
    output=$(./cuda_perftest "$mtx_file" 2>&1)
    
    if [ $? -eq 0 ]; then
        our_perf=$(echo "$output" | grep "our_perf:" | sed 's/.*our_perf: *\([0-9.]*\) *ms.*/\1/')
        our_pre=$(echo "$output" | grep "our_perf:" | sed 's/.*our_pre: *\([0-9.]*\) *ms.*/\1/')
        cusparse_perf=$(echo "$output" | grep "cusparse_perf:" | sed 's/.*cusparse_perf: *\([0-9.]*\) *ms.*/\1/')
        cusparse_pre=$(echo "$output" | grep "cusparse_perf:" | sed 's/.*cusparse_pre: *\([0-9.]*\) *ms.*/\1/')
        
        if [[ -n "$our_perf" && -n "$cusparse_perf" && -n "$our_pre" && -n "$cusparse_pre" ]]; then
            echo "$matrix_name,$our_perf,$cusparse_perf,$our_pre,$cusparse_pre" >> "$CSV_FILE"
            sync
            echo "  Done"
        else
            echo "  Parse failed"
        fi
    else
        echo "  Run failed"
    fi
done

echo "Test completed. Results saved to: $CSV_FILE"
echo "Total tested: $(tail -n +2 "$CSV_FILE" | wc -l) matrices" 