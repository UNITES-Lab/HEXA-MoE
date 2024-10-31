'''
    A Triton implementation for Fused Ops of ESMM, ESTMM and ESS.
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
def fused_kernel_bias(
        # Pointers to matrices
        r_ptr_ess, r_ptr_esmm, r_ptr_estmm, tok_ptr_1, tok_ptr_2, w_ptr, tok_ids_list, exp_ids_list, exp_start_list,
        # Matrix dimensions
        num_expanded_toks, num_toks, num_experts, num_in_dims, num_out_dims,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_itok1_num, stride_itok1_dim, #
        stride_itok2_num, stride_itok2_dim, #
        stride_w_out, stride_w_in, #
        stride_ess_exp, stride_ess_dim, #
        stride_esmm_num, stride_esmm_dim, #
        stride_estmm_dim1, stride_estmm_dim2, #
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    '''
        Inputs:
            tok_1 has shape (num_toks + 1, num_out_dims), # padding with zeros
            tok_2 has shape (num_toks + 1, num_in_dims), # padding with zeros
            w has shape (num_experts * num_out_dims, num_in_dims),
            tok_ids_list has shape (num_expanded_toks,),
            exp_ids_list has shape (num_expanded_toks // BLOCK_SIZE_M,),
        Outputs:
            result_ess has shape (num_experts, num_out_dims),
            result_esmm has shape (num_toks, num_in_dims),
            result_estmm has shape (num_experts * num_out_dims, num_in_dims)
    '''
    pid = tl.program_id(axis=0)

    '''
        pid splitting:
            ess shares num_experts * (num_out_dims // BLOCK_SIZE_N) processes
            esmm shares (num_expanded_toks // BLOCK_SIZE_M) * (num_in_dims // BLOCK_SIZE_N) processes
            estmm shares num_experts * (num_out_dims // BLOCK_SIZE_K) * (num_in_dims // BLOCK_SIZE_N) processes
    '''
    # pid splitting
    num_pid_ess = tl.cdiv(num_out_dims, BLOCK_SIZE_N)
    num_pid_esmm_1 = tl.cdiv(num_expanded_toks, BLOCK_SIZE_M)
    num_pid_esmm_2 = tl.cdiv(num_in_dims, BLOCK_SIZE_N)
    num_pid_estmm_1 = tl.cdiv(num_out_dims, BLOCK_SIZE_K)
    num_pid_estmm_2 = tl.cdiv(num_in_dims, BLOCK_SIZE_N)

    procs_ess = num_experts * num_pid_ess
    procs_esmm = num_pid_esmm_1 * num_pid_esmm_2
    procs_estmm = num_experts * num_pid_estmm_1 * num_pid_estmm_2

    if pid < procs_ess:
        num_pid_out_group = GROUP_SIZE_M * num_experts
        first_pid_out = (pid // num_pid_out_group) * GROUP_SIZE_M
        group_size_out = min(num_pid_ess - first_pid_out, GROUP_SIZE_M)
        pid_out = first_pid_out + ((pid % num_pid_out_group) % group_size_out)
        pid_exp = (pid % num_pid_out_group) // group_size_out

        exp_start_idx = tl.load(exp_start_list + pid_exp)
        exp_end_idx = tl.load(exp_start_list + pid_exp + 1)

        if (exp_start_idx < num_expanded_toks):
            offs_toks_chunk = tl.arange(0, BLOCK_SIZE_M)
            offs_tok = tl.load(tok_ids_list + exp_start_idx + offs_toks_chunk)
            offs_out = (pid_out * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_out_dims

            tok1_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_out[None, :] * stride_itok1_dim)
            accumulator = tl.zeros((1, BLOCK_SIZE_N), dtype=tok_ptr_1.dtype.element_ty)

            offs_out_mask = (pid_out * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < num_out_dims
            for delta_tok in range(0, tl.cdiv(exp_end_idx - exp_start_idx, BLOCK_SIZE_M)):
                tok_vals = tl.load(tok1_ptrs, mask=offs_out_mask, other=0.0).to(tok_ptr_1.dtype.element_ty)
                accumulator = accumulator + tl.sum(tok_vals, axis=0, keep_dims=True)
                offs_tok = tl.load(tok_ids_list + exp_start_idx + delta_tok * BLOCK_SIZE_M + offs_toks_chunk)
                tok1_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_out[None, :] * stride_itok1_dim)
            accumulator = accumulator.to(r_ptr_ess.dtype.element_ty)

            # -----------------------------------------------------------
            # Write back to the result of ess
            ess_r_ptrs = r_ptr_ess + \
                ((pid_exp + tl.arange(0, 1))[:, None] * stride_ess_exp + offs_out[None, :] * stride_ess_dim)
            tl.store(ess_r_ptrs, accumulator, mask=offs_out_mask)
    elif pid < procs_ess + procs_esmm:
        this_pid = pid - (procs_ess)
        num_pid_in_group = GROUP_SIZE_M * num_pid_esmm_2
        first_pid_m = (this_pid // num_pid_in_group) * GROUP_SIZE_M
        group_size_m = min(num_pid_esmm_1 - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((this_pid % num_pid_in_group) % group_size_m)
        pid_n = (this_pid % num_pid_in_group) // group_size_m

        offsets_toks = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        local_exp_idx = tl.load(exp_ids_list + pid_m)
        offs_tok = tl.load(tok_ids_list + offsets_toks)
        offs_out = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_in_dims
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        tok_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_k[None, :] * stride_itok1_dim)
        w_ptrs = w_ptr + ((offs_k + local_exp_idx * num_out_dims)[:, None] * stride_w_out + offs_out[None, :] * stride_w_in)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=r_ptr_esmm.dtype.element_ty)
        for k in range(0, tl.cdiv(num_out_dims, BLOCK_SIZE_K)):
            a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_out_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
            b = tl.load(w_ptrs, mask=offs_k[:, None] < num_out_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
            # a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_in_dims - k * BLOCK_SIZE_K, other=0.0)
            # b = tl.load(w_ptrs, mask=offs_k[:, None] < num_in_dims - k * BLOCK_SIZE_K, other=0.0)

            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # Advance the ptrs to the next K block.
            tok_ptrs += BLOCK_SIZE_K * stride_itok1_dim
            w_ptrs += BLOCK_SIZE_K * stride_w_out
        # c = accumulator
        c = accumulator.to(r_ptr_esmm.dtype.element_ty)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_tok = offs_tok - 1
        c_ptrs = r_ptr_esmm + stride_esmm_num * offs_tok[:, None] + stride_esmm_dim * offs_cn[None, :]
        c_mask = (offs_tok[:, None] >= 0) & (offs_cn[None, :] < num_in_dims)

        tl.store(c_ptrs, c, mask=c_mask)
    else:
        this_pid = pid - (procs_ess + procs_esmm)
        num_pid_in_group = GROUP_SIZE_M * num_pid_estmm_2
        first_pid_m = (this_pid // num_pid_in_group) * GROUP_SIZE_M
        group_size_m = min(num_experts * num_pid_estmm_1 - first_pid_m, GROUP_SIZE_M)
        pid_toks_1 = first_pid_m + ((this_pid % num_pid_in_group) % group_size_m)
        pid_toks_2 = (this_pid % num_pid_in_group) // group_size_m

        pid_exp = pid_toks_1 // num_pid_estmm_1
        pid_toks_1 = pid_toks_1 % num_pid_estmm_1

        exp_start_idx = tl.load(exp_start_list + pid_exp)
        exp_end_idx = tl.load(exp_start_list + pid_exp + 1)

        if (exp_start_idx < num_expanded_toks):
            offs_toks_chunk = tl.arange(0, BLOCK_SIZE_M)
            offs_tok = tl.load(tok_ids_list + exp_start_idx + offs_toks_chunk)
            offs_1 = (pid_toks_1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % num_out_dims
            offs_2 = (pid_toks_2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_in_dims

            tok1_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_1[None, :] * stride_itok1_dim)
            tok2_ptrs = tok_ptr_2 + (offs_tok[:, None] * stride_itok2_num + offs_2[None, :] * stride_itok2_dim)

            accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
            # accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=r_ptr_estmm.dtype.element_ty)

            offs_mask_1 = (pid_toks_1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] < num_out_dims
            offs_mask_2 = (pid_toks_2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < num_in_dims
            for delta_tok in range(0, tl.cdiv(exp_end_idx - exp_start_idx, BLOCK_SIZE_M)):
                tok1_vals = tl.load(tok1_ptrs, mask=offs_mask_1, other=0.0).to(tl.float16)
                tok2_vals = tl.load(tok2_ptrs, mask=offs_mask_2, other=0.0).to(tl.float16)
                # tok1_vals = tl.load(tok1_ptrs, mask=offs_mask_1, other=0.0).to(r_ptr_estmm.dtype.element_ty)
                # tok2_vals = tl.load(tok2_ptrs, mask=offs_mask_2, other=0.0).to(r_ptr_estmm.dtype.element_ty)

                accumulator = tl.dot(tok1_vals.trans(1, 0), tok2_vals, accumulator)

                offs_tok = tl.load(tok_ids_list + exp_start_idx + delta_tok * BLOCK_SIZE_M + offs_toks_chunk)
                tok1_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_1[None, :] * stride_itok1_dim)
                tok2_ptrs = tok_ptr_2 + (offs_tok[:, None] * stride_itok2_num + offs_2[None, :] * stride_itok2_dim)
            r_estmm = accumulator.to(r_ptr_estmm.dtype.element_ty)
            # r_estmm = accumulator

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_1 = (pid_toks_1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
            offs_2 = (pid_toks_2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
            offs_mask_1 = offs_1[:, None] < num_out_dims
            offs_mask_2 = offs_2[None, :] < num_in_dims
            estmm_ptrs = r_ptr_estmm + \
                stride_estmm_dim1 * (offs_1 + pid_exp * num_out_dims)[:, None] + stride_estmm_dim2 * offs_2[None, :]
            estmm_mask = offs_mask_1 & offs_mask_2

            tl.store(estmm_ptrs, r_estmm, mask=estmm_mask)

@triton.autotune(
    configs=get_autotune_config(),
    key=['num_expanded_toks', 'num_out_dims', 'num_in_dims'],
)
@triton.jit
def fused_kernel_no_bias(
        # Pointers to matrices
        r_ptr_esmm, r_ptr_estmm, tok_ptr_1, tok_ptr_2, w_ptr, tok_ids_list, exp_ids_list, exp_start_list,
        # Matrix dimensions
        num_expanded_toks, num_toks, num_experts, num_in_dims, num_out_dims,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_itok1_num, stride_itok1_dim, #
        stride_itok2_num, stride_itok2_dim, #
        stride_w_out, stride_w_in, #
        stride_esmm_num, stride_esmm_dim, #
        stride_estmm_dim1, stride_estmm_dim2, #
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr
):
    '''
        Inputs:
            tok_1 has shape (num_toks + 1, num_out_dims), # padding with zeros
            tok_2 has shape (num_toks + 1, num_in_dims), # padding with zeros
            w has shape (num_experts * num_out_dims, num_in_dims),
            tok_ids_list has shape (num_expanded_toks,),
            exp_ids_list has shape (num_expanded_toks // BLOCK_SIZE_M,),
        Outputs:
            result_esmm has shape (num_toks, num_in_dims),
            result_estmm has shape (num_experts * num_out_dims, num_in_dims)
    '''
    pid = tl.program_id(axis=0)

    '''
        pid splitting:
            esmm shares (num_expanded_toks // BLOCK_SIZE_M) * (num_in_dims // BLOCK_SIZE_N) processes
            estmm shares num_experts * (num_out_dims // BLOCK_SIZE_K) * (num_in_dims // BLOCK_SIZE_N) processes
    '''
    # pid splitting
    num_pid_esmm_1 = tl.cdiv(num_expanded_toks, BLOCK_SIZE_M)
    num_pid_esmm_2 = tl.cdiv(num_in_dims, BLOCK_SIZE_N)
    num_pid_estmm_1 = tl.cdiv(num_out_dims, BLOCK_SIZE_K)
    num_pid_estmm_2 = tl.cdiv(num_in_dims, BLOCK_SIZE_N)

    procs_esmm = num_pid_esmm_1 * num_pid_esmm_2
    procs_estmm = num_experts * num_pid_estmm_1 * num_pid_estmm_2

    if pid < procs_esmm:
        this_pid = pid
        num_pid_in_group = GROUP_SIZE_M * num_pid_esmm_2
        first_pid_m = (this_pid // num_pid_in_group) * GROUP_SIZE_M
        group_size_m = min(num_pid_esmm_1 - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((this_pid % num_pid_in_group) % group_size_m)
        pid_n = (this_pid % num_pid_in_group) // group_size_m

        offsets_toks = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        local_exp_idx = tl.load(exp_ids_list + pid_m)
        offs_tok = tl.load(tok_ids_list + offsets_toks)
        offs_out = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_in_dims
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        tok_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_k[None, :] * stride_itok1_dim)
        w_ptrs = w_ptr + ((offs_k + local_exp_idx * num_out_dims)[:, None] * stride_w_out + offs_out[None, :] * stride_w_in)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=r_ptr_esmm.dtype.element_ty)
        for k in range(0, tl.cdiv(num_out_dims, BLOCK_SIZE_K)):
            a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_out_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
            b = tl.load(w_ptrs, mask=offs_k[:, None] < num_out_dims - k * BLOCK_SIZE_K, other=0.0).to(tl.float16)
            # a = tl.load(tok_ptrs, mask=offs_k[None, :] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(r_ptr_esmm.dtype.element_ty)
            # b = tl.load(w_ptrs, mask=offs_k[:, None] < num_in_dims - k * BLOCK_SIZE_K, other=0.0).to(r_ptr_esmm.dtype.element_ty)

            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # Advance the ptrs to the next K block.
            tok_ptrs += BLOCK_SIZE_K * stride_itok1_dim
            w_ptrs += BLOCK_SIZE_K * stride_w_out

        # c = accumulator
        c = accumulator.to(r_ptr_esmm.dtype.element_ty)

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_tok = offs_tok - 1
        c_ptrs = r_ptr_esmm + stride_esmm_num * offs_tok[:, None] + stride_esmm_dim * offs_cn[None, :]
        c_mask = (offs_tok[:, None] >= 0) & (offs_cn[None, :] < num_in_dims)

        tl.store(c_ptrs, c, mask=c_mask)
    else:
        this_pid = pid - (procs_esmm)
        num_pid_in_group = GROUP_SIZE_M * num_pid_estmm_2
        first_pid_m = (this_pid // num_pid_in_group) * GROUP_SIZE_M
        group_size_m = min(num_experts * num_pid_estmm_1 - first_pid_m, GROUP_SIZE_M)
        pid_toks_1 = first_pid_m + ((this_pid % num_pid_in_group) % group_size_m)
        pid_toks_2 = (this_pid % num_pid_in_group) // group_size_m

        pid_exp = pid_toks_1 // num_pid_estmm_1
        pid_toks_1 = pid_toks_1 % num_pid_estmm_1

        exp_start_idx = tl.load(exp_start_list + pid_exp)
        exp_end_idx = tl.load(exp_start_list + pid_exp + 1)

        if (exp_start_idx < num_expanded_toks):
            offs_toks_chunk = tl.arange(0, BLOCK_SIZE_M)
            offs_tok = tl.load(tok_ids_list + exp_start_idx + offs_toks_chunk)
            offs_1 = (pid_toks_1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % num_out_dims
            offs_2 = (pid_toks_2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % num_in_dims

            tok1_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_1[None, :] * stride_itok1_dim)
            tok2_ptrs = tok_ptr_2 + (offs_tok[:, None] * stride_itok2_num + offs_2[None, :] * stride_itok2_dim)

            accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
            # accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=r_ptr_estmm.dtype.element_ty)

            offs_mask_1 = (pid_toks_1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] < num_out_dims
            offs_mask_2 = (pid_toks_2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] < num_in_dims
            for delta_tok in range(0, tl.cdiv(exp_end_idx - exp_start_idx, BLOCK_SIZE_M)):
                tok1_vals = tl.load(tok1_ptrs, mask=offs_mask_1, other=0.0).to(tl.float16)
                tok2_vals = tl.load(tok2_ptrs, mask=offs_mask_2, other=0.0).to(tl.float16)
                # tok1_vals = tl.load(tok1_ptrs, mask=offs_mask_1, other=0.0).to(r_ptr_estmm.dtype.element_ty)
                # tok2_vals = tl.load(tok2_ptrs, mask=offs_mask_2, other=0.0).to(r_ptr_estmm.dtype.element_ty)

                accumulator = tl.dot(tok1_vals.trans(1, 0), tok2_vals, accumulator)

                offs_tok = tl.load(tok_ids_list + exp_start_idx + delta_tok * BLOCK_SIZE_M + offs_toks_chunk)
                tok1_ptrs = tok_ptr_1 + (offs_tok[:, None] * stride_itok1_num + offs_1[None, :] * stride_itok1_dim)
                tok2_ptrs = tok_ptr_2 + (offs_tok[:, None] * stride_itok2_num + offs_2[None, :] * stride_itok2_dim)
            # r_estmm = accumulator
            r_estmm = accumulator.to(r_ptr_estmm.dtype.element_ty)

            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_1 = (pid_toks_1 * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
            offs_2 = (pid_toks_2 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
            offs_mask_1 = offs_1[:, None] < num_out_dims
            offs_mask_2 = offs_2[None, :] < num_in_dims
            estmm_ptrs = r_ptr_estmm + \
                stride_estmm_dim1 * (offs_1 + pid_exp * num_out_dims)[:, None] + stride_estmm_dim2 * offs_2[None, :]
            estmm_mask = offs_mask_1 & offs_mask_2

            tl.store(estmm_ptrs, r_estmm, mask=estmm_mask)

def es_fused(
        tok1, tok2, w, tok_ids_list, exp_ids_list, exp_start_list, num_exps, bias=True
):
    '''
        tok1 has shape (num_toks, num_out_dims),
        tok2 has shape (num_toks, num_in_dims),
        w has shape (num_exps, num_out_dims, num_in_dims),
        tok_ids_list has shape (num_expanded_toks,), 
        exp_ids_list has shape (num_expanded_toks // MAX_TOKEN_CHUNK,),
        exp_start_list has shape (1 + num_exps)
    '''
    w_dtype = w.dtype
    assert tok1.dtype == tok2.dtype
    assert tok1.shape[0] == tok2.shape[0] and tok_ids_list.shape[0] == constant.MAX_TOKEN_CHUNK * exp_ids_list.shape[0]
    assert tok2.shape[1] == w.shape[2] and num_exps == w.shape[0] and tok1.shape[1] == w.shape[1]
    assert exp_start_list.shape[0] == 1 + num_exps
    assert tok1.is_contiguous() and tok2.is_contiguous()
    assert tok_ids_list.is_contiguous() and exp_ids_list.is_contiguous() and exp_start_list.is_contiguous()

    num_toks = tok1.shape[0]
    num_expanded_toks = tok_ids_list.shape[0]
    num_out_dims = tok1.shape[1]
    num_in_dims = tok2.shape[1]

    w = w.reshape(num_exps * num_out_dims, num_in_dims).contiguous()

    r_esmm = torch.empty((num_toks, num_in_dims), device=tok1.device, dtype=tok1.dtype)
    r_estmm = torch.empty((num_exps * num_out_dims, num_in_dims), device=w.device, dtype=w_dtype)

    tok1_pad_zero = torch.zeros((1, num_out_dims), device=tok1.device, dtype=tok1.dtype)
    tok2_pad_zero = torch.zeros((1, num_in_dims), device=tok2.device, dtype=tok2.dtype)

    tok1_padded = torch.cat([tok1_pad_zero, tok1], dim=0).contiguous()
    tok2_padded = torch.cat([tok2_pad_zero, tok2], dim=0).contiguous()

    if bias:
        r_ess = torch.empty((num_exps, num_out_dims), device=w.device, dtype=w_dtype)

        grid = lambda META: (num_exps * triton.cdiv(num_out_dims, META['BLOCK_SIZE_N']) + \
            triton.cdiv(num_expanded_toks, META['BLOCK_SIZE_M']) * triton.cdiv(num_in_dims, META['BLOCK_SIZE_N']) + \
            num_exps * triton.cdiv(num_out_dims, META['BLOCK_SIZE_K']) * triton.cdiv(num_in_dims, META['BLOCK_SIZE_N']),)
        
        fused_kernel_bias[grid](
            r_ess, r_esmm, r_estmm, tok1_padded, tok2_padded, w, tok_ids_list, exp_ids_list, exp_start_list,
            num_expanded_toks, num_toks, num_exps, num_in_dims, num_out_dims,
            tok1_padded.stride(0), tok1_padded.stride(1),
            tok2_padded.stride(0), tok2_padded.stride(1),
            w.stride(0), w.stride(1),
            r_ess.stride(0), r_ess.stride(1),
            r_esmm.stride(0), r_esmm.stride(1),
            r_estmm.stride(0), r_estmm.stride(1),
        )

        return r_ess, r_esmm, r_estmm.reshape(num_exps, num_out_dims, num_in_dims)
    else:
        grid = lambda META: (triton.cdiv(num_expanded_toks, META['BLOCK_SIZE_M']) * triton.cdiv(num_in_dims, META['BLOCK_SIZE_N']) + \
            num_exps * triton.cdiv(num_out_dims, META['BLOCK_SIZE_K']) * triton.cdiv(num_in_dims, META['BLOCK_SIZE_N']),)
        
        fused_kernel_no_bias[grid](
            r_esmm, r_estmm, tok1_padded, tok2_padded, w, tok_ids_list, exp_ids_list, exp_start_list,
            num_expanded_toks, num_toks, num_exps, num_in_dims, num_out_dims,
            tok1_padded.stride(0), tok1_padded.stride(1),
            tok2_padded.stride(0), tok2_padded.stride(1),
            w.stride(0), w.stride(1),
            r_esmm.stride(0), r_esmm.stride(1),
            r_estmm.stride(0), r_estmm.stride(1),
        )

        return r_esmm, r_estmm.reshape(num_exps, num_out_dims, num_in_dims)
