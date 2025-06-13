#!/bin/bash

# output csv file
output_file="results_merge_fp64.csv"
echo "Matrix,merge (ms)" > "$output_file"

# Iterate through all subfolders in the ../../data/mtx/ directory
for dir in /home/v-wangtuowei/wangluhan/data/mtx/*/; do
    if [ -d "$dir" ]; then
        matrix_name=$(basename "$dir")
        mtx_file="$dir/$matrix_name.mtx"
        
        if [ -f "$mtx_file" ]; then
            echo "Processing $mtx_file..."
            
            # Run ./gpu_spmv and extract the setup and avg ms time
            output=$(./gpu_spmv --mtx="$mtx_file" 2>&1)
            
            # Look for "fp64: ... setup ms, ... avg ms" pattern and extract avg ms
            spmv_time=$(echo "$output" | grep -oP "fp64: [0-9.]+ setup ms, \K[0-9.]+(?= avg ms)")
            
            # Ensure that the extracted value is not null
            if [[ -n "$spmv_time" ]]; then
                echo "$matrix_name,$spmv_time" >> "$output_file"
                echo "Recorded: $matrix_name -> $spmv_time ms"
            else
                echo "$matrix_name,OOM" >> "$output_file"
                echo "Warning: Failed to extract GPU SpMV time for $matrix_name, marked as OOM"
            fi
        else
            echo "Warning: No matching .mtx file found for $matrix_name"
        fi
    fi
done

echo "Results saved to $output_file"
