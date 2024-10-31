'''
    A Triton implementation for ESMM
'''

import torch
import triton
import triton.language as tl
import constant

constant.MAX_TOKEN_CHUNK = 16

'''
    BLOCK_SIZE here cannot be very large. In ESMM, it must be lower than MAX_TOKEN_CHUNK (16).
'''

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': constant.MAX_TOKEN_CHUNK, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_autotune_config():
    return get_cuda_autotune_config()
    
@triton.autotune(
    configs=get_autotune_config(),
    key=['num_expanded_toks', 'num_out_dims', 'num_in_dims'],
)
@triton.jit
def esmm_kernel_bias(
        # Pointers to matrices
        r_ptr, tok_ptr, w_ptr, b_ptr, tok_ids_list, exp_ids_list,
        # Matrix dimensions
        num_expanded_toks, num_in_dims, num_out_dims,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_itok_num, stride_itok_dim, #
        stride_w_in, stride_w_out, #
        stride_b_exp, stride_b_out, #
        stride_otok_num, stride_otok_dim,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    """
        Kernel for computing expert-specific matrix multiplication.
        tok has shape (num_toks + 1, num_in_dims), # padding with zeros
        w has shape (num_experts * num_in_dims, num_out_dims),
        b has shape (num_experts, num_out_dims),
        result has shape (num_toks + 1, num_out_dims), # padding with zeros
        tok_ids_list has shape (num_expanded_toks,),
        exp_ids_list has shape (num_expanded_toks // BLOCK_SIZE_M,)
    """
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(num_expanded_toks, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(num_out_dims, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offsets_toks = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % num_expanded_toks
    
    local_exp_idx = tl.load(exp_ids_list + pid_m)
    offs_tok = tl.load(tok_ids_list + offsets_toks)
    offs_out = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_out_dims
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    tok_ptrs = tok_ptr + (offs_tok[:, None] * stride_itok_num + offs_k[None, :] * stride_itok_dim)
    w_ptrs = w_ptr + ((offs_k + local_exp_idx * num_in_dims)[:, None] * stride_w_in + offs_out[None, :] * stride_w_out)
    b_ptrs = b_ptr + (local_exp_idx * stride_b_exp + offs_out * stride_b_out)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=r_ptr.dtype.element_ty)
    bias_mask = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < num_out_dims
    # bias_tmp = tl.load(b_ptrs, mask=bias_mask, other=0.0).expand_dims(0).to(r_ptr.dtype.element_ty)
    bias_tmp = tl.load(b_ptrs, mask=bias_mask, other=0.0).expand_dims(0).to(tl.float32)
    accumulator = accumulator + tl.broadcast_to(bias_tmp, (BLOCK_SIZE_M, BLOCK_SIZE_N))

    for k in range(0, tl.cdiv(num_in_dims, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
        b = tl.load(w_ptrs, mask=offs_k[:, None] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
        # a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(r_ptr.dtype.element_ty)
        # b = tl.load(w_ptrs, mask=offs_k[:, None] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(r_ptr.dtype.element_ty)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        tok_ptrs += BLOCK_SIZE_K * stride_itok_dim
        w_ptrs += BLOCK_SIZE_K * stride_w_in
    c = accumulator.to(r_ptr.dtype.element_ty)
    # c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_tok = offs_tok - 1
    c_ptrs = r_ptr + stride_otok_num * offs_tok[:, None] + stride_otok_dim * offs_cn[None, :]
    c_mask = (offs_tok[:, None] >= 0) & (offs_cn[None, :] < num_out_dims)
    # init_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)

    tl.store(c_ptrs, c, mask=c_mask)

@triton.autotune(
    configs=get_autotune_config(),
    key=['num_expanded_toks', 'num_out_dims', 'num_in_dims'],
)
@triton.jit
def esmm_kernel_no_bias(
        # Pointers to matrices
        r_ptr, tok_ptr, w_ptr, tok_ids_list, exp_ids_list,
        # Matrix dimensions
        num_expanded_toks, num_in_dims, num_out_dims,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_itok_num, stride_itok_dim, #
        stride_w_in, stride_w_out, #
        stride_otok_num, stride_otok_dim,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    """
        Kernel for computing expert-specific matrix multiplication.
        tok has shape (num_toks + 1, num_in_dims), # padding with zeros
        w has shape (num_experts * num_in_dims, num_out_dims),
        b has shape (num_experts, num_out_dims),
        result has shape (num_toks + 1, num_out_dims), # padding with zeros
        tok_ids_list has shape (num_expanded_toks,),
        exp_ids_list has shape (num_expanded_toks // BLOCK_SIZE_M,)
    """
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(num_expanded_toks, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(num_out_dims, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offsets_toks = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % num_expanded_toks
    
    local_exp_idx = tl.load(exp_ids_list + pid_m)
    offs_tok = tl.load(tok_ids_list + offsets_toks)
    offs_out = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_out_dims
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    tok_ptrs = tok_ptr + (offs_tok[:, None] * stride_itok_num + offs_k[None, :] * stride_itok_dim)
    w_ptrs = w_ptr + ((offs_k + local_exp_idx * num_in_dims)[:, None] * stride_w_in + offs_out[None, :] * stride_w_out)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=r_ptr.dtype.element_ty)

    for k in range(0, tl.cdiv(num_in_dims, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
        b = tl.load(w_ptrs, mask=offs_k[:, None] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
        # a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(r_ptr.dtype.element_ty)
        # b = tl.load(w_ptrs, mask=offs_k[:, None] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(r_ptr.dtype.element_ty)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        tok_ptrs += BLOCK_SIZE_K * stride_itok_dim
        w_ptrs += BLOCK_SIZE_K * stride_w_in
    c = accumulator.to(r_ptr.dtype.element_ty)
    # c = accumulator

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_tok = offs_tok - 1
    c_ptrs = r_ptr + stride_otok_num * offs_tok[:, None] + stride_otok_dim * offs_cn[None, :]
    c_mask = (offs_tok[:, None] >= 0) & (offs_cn[None, :] < num_out_dims)
    # init_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)

    tl.store(c_ptrs, c, mask=c_mask)

def ESMM(toks, w, b, tok_ids_list, exp_ids_list):
    '''
        toks has shape (num_toks, num_in_dims),
        w has shape (num_exps, num_in_dims),
        b has shape (num_exps, num_out_dims) or null,
        tok_ids_list has shape (num_expanded_toks,), 
        exp_ids_list has shape (num_expanded_toks // MAX_TOKEN_CHUNK,)
    '''
    # Check constraints.
    toks_dtype = toks.dtype
    assert toks.is_contiguous() and w.is_contiguous() and toks.shape[1] == w.shape[1] and \
        tok_ids_list.shape[0] == constant.MAX_TOKEN_CHUNK * exp_ids_list.shape[0] and \
        tok_ids_list.is_contiguous() and exp_ids_list.is_contiguous(), "Matrix A must be contiguous"
    num_toks, num_in_dims = toks.shape
    num_exps, _, num_out_dims = w.shape
    num_expanded_toks = tok_ids_list.shape[0]
    # Pad input tokens.
    toks_pad = torch.zeros((1, num_in_dims), device=toks.device, dtype=toks_dtype)
    toks_padded = torch.cat([toks_pad, toks], dim=0).contiguous()
    w = w.reshape(num_exps * num_in_dims, num_out_dims).contiguous()
    # Allocates output.
    c = torch.zeros((num_toks, num_out_dims), device=toks.device, dtype=toks_dtype)
    if not b.numel():
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(num_expanded_toks, META['BLOCK_SIZE_M']) * triton.cdiv(num_out_dims, META['BLOCK_SIZE_N']),)
        esmm_kernel_no_bias[grid](
            c, toks_padded, w, tok_ids_list, exp_ids_list, #
            num_expanded_toks, num_in_dims, num_out_dims, #
            toks_padded.stride(0), toks_padded.stride(1), #
            w.stride(0), w.stride(1), #
            c.stride(0), c.stride(1)
        )
        return c
    else:
        assert num_out_dims == b.shape[1] and num_exps == b.shape[0]
        b = b.to(torch.float32)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(num_expanded_toks, META['BLOCK_SIZE_M']) * triton.cdiv(num_out_dims, META['BLOCK_SIZE_N']),)
        esmm_kernel_bias[grid](
            c, toks_padded, w, b, tok_ids_list, exp_ids_list, #
            num_expanded_toks, num_in_dims, num_out_dims, #
            toks_padded.stride(0), toks_padded.stride(1), #
            w.stride(0), w.stride(1), #
            b.stride(0), b.stride(1), #
            c.stride(0), c.stride(1)
        )
        return c
