#!/bin/bash

# Check if the required arguments are provided
export_file_name="$1"
kernel_name="$2"
filename="$3"

# Execute the ncu command
ncu --config-file off --export "$export_file_name" --force-overwrite --kernel-name "$kernel_name" --section-folder /tmp/var/sections --set full "./$filename"
