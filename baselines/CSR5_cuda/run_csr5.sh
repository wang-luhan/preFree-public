#!/bin/bash

# output csv file
output_file="results_csr5.csv"
echo "Matrix,CSR5 SpMV Time (ms)" > "$output_file"

# Iterate through all subfolders in the ../../../data/mtx/ directory
for dir in ../../../../data/mtx/*/; do
    if [ -d "$dir" ]; then
        matrix_name=$(basename "$dir")
        mtx_file="$dir/$matrix_name.mtx"
        
        if [ -f "$mtx_file" ]; then
            echo "Processing $mtx_file..."
            
            # Run . /spmv and extract the CSR5-based SpMV time
            output=$(./spmv "$mtx_file" 2>&1)
            spmv_time=$(echo "$output" | grep -oP "CSR5-based SpMV time = \K[0-9.]+(?= ms)")

            # Ensure that the extracted value is not null
            if [[ -n "$spmv_time" ]]; then
                echo "$matrix_name,$spmv_time" >> "$output_file"
                echo "Recorded: $matrix_name -> $spmv_time ms"
            else
                echo "$matrix_name,OOM" >> "$output_file"
                echo "Warning: Failed to extract CSR5-based SpMV time for $matrix_name, marked as OOM"
            fi
        else
            echo "Warning: No matching .mtx file found for $matrix_name"
        fi
    fi
done

echo "Results saved to $output_file"
