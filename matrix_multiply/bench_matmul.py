import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_matmul = load(name='matmul', sources=['main.cpp', 'matmul.cu'], extra_cuda_cflags=['-O2'])

batch_size = 1
n_head = 8
seq_len = 10490
head_embd = 64
torch.cuda.empty_cache()

// Wether to use on transposed matrices (For QK^T)
transpose = True

if not transpose:
    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, head_embd, seq_len).cuda()
else:
    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, head_embd, seq_len).cuda()

print('=== profiling manual attention ===')


# Compare to Pytroch's matmul
def manual_matmul(q, k):
    if transpose:
        y = torch.matmul(q, k.transpose(-2, -1))
    else:
        y = torch.matmul(q, k)
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_matmul_transpose(q, k)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_matmul = minimal_matmul.forward(q, k, transpose)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(minimal_matmul.cpu())
print(manual_result.cpu())
print('attn values sanity check:', torch.allclose(minimal_matmul, manual_result, rtol=0, atol=1e-02))

