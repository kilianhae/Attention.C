filename=$(basename "$1" .cu)
nvcc "$1" -o "$filename" -lcublas
./run_ncu.sh "$2" "$2" "$filename"
