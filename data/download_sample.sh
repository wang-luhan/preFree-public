#!/bin/bash

# Define the log file containing the URLs
log_file="sample_url.log"

# Define the target directory for downloaded files
target_dir="targz"

# Ensure the target directory exists
mkdir -p "$target_dir"

# Loop through each line in the log file
while IFS= read -r url; do
  # Extract the filename from the URL
  file_name=$(basename "$url")
  
  # Check if the file already exists in the target directory
  if [[ -f "$target_dir/$file_name" ]]; then
    echo "File $file_name already exists in $target_dir. Skipping download."
  else
    echo "Downloading $file_name to $target_dir..."
    # Use wget to download the file to the target directory
    wget -O "$target_dir/$file_name" "$url"
  fi
done < "$log_file"

echo "Download process completed."
