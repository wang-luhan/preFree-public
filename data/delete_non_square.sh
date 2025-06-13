#!/bin/bash

root_dir="./mtx"

log_file="./check_issquare_log.txt"

> "$log_file"

find "$root_dir" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    folder_name=$(basename "$dir")
    
    mtx_file="$dir/$folder_name.mtx"
    
    if [ -f "$mtx_file" ]; then
        read rows cols _ < <(grep -v '^%' "$mtx_file" | head -n 1)
        
        if [[ "$rows" =~ ^[0-9]+$ && "$cols" =~ ^[0-9]+$ ]]; then

            if [ "$rows" -ne "$cols" ]; then
                echo "Matrix rows and columns are not equal: $mtx_file" | tee -a "$log_file"
                rm -rf "$dir"
            fi
        else
            echo "Unable to parse file: $mtx_file" | tee -a "$log_file"
        fi
    else
        echo "File does not exist: $mtx_file" | tee -a "$log_file"
    fi
done

echo "Check completed, log saved in $log_file"
