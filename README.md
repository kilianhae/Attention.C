A forward pass implementation in Cuda C of a simplified (no layernorm, no mask) Attention layer beating PyTorch performance in forward only.

The operation consists of 3 Operations which we implement as seperate kernels:

- Matmul
- Softmax
- Transpose (can be fused into matmul)

Each of these is implemented in increasing complexity and performance.
For testing we bind these kernels into pytorch and call them from there.

### Benchmarking
- To profile the individual kernels run the bench.py files in the respective directory.
- To benchmark the full attention pass, run the outermost bench.py file.
