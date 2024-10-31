'''
    A Triton implementation for token_count and token_assign functions.
'''

import time
import torch
import triton
import triton.language as tl
import constant

constant.MAX_TOKEN_CHUNK = 16

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

@triton.jit
def token_count_kernel(
    r_ptr, routing,
    num_experts, num_toks,
    stride_output, stride_input
):
    '''
        Inputs:
            routing has shape (num_toks,)
        Outputs:
            result has shape (num_experts,)
    '''
    pid = tl.program_id(axis=0)

    if (pid >= 0 and pid < num_toks):
        routed_expert = tl.load(routing + pid * stride_input)
        if (routed_expert >= 0 and routed_expert < num_experts):
            tl.atomic_add(r_ptr + routed_expert * stride_output, 1)

@triton.jit
def padding_kernel(
    padded_counter, counter,
    num_experts, chunk_size,
    stride_output, stride_input
):
    '''
        Making the counter divisible by constant.MAX_TOKEN_CHUNK.
        Inputs:
            counter has shape (num_experts,)
        Outputs:
            padded_counter has shape (num_experts,)
    '''
    pid = tl.program_id(axis=0)

    if (pid >= 0 and pid < num_experts):
        val = tl.load(counter + pid * stride_input)
        padded_val = tl.cdiv(val, chunk_size) * chunk_size
        tl.store(padded_counter + pid * stride_output, padded_val)

@triton.jit
def assign_tok_kernel(
    tok_ids_list, exp_start_list, routing,
    num_experts, num_toks,
    stride_output, stride_input, stride_input_routing
):
    pid = tl.program_id(axis=0)

    if (pid >= 0 and pid < num_toks):
        routed_expert = tl.load(routing + pid * stride_input_routing)
        if (routed_expert >= 0 and routed_expert < num_experts):
            write_pos = tl.atomic_add(exp_start_list + routed_expert * stride_input, 1)
            tl.store(tok_ids_list + write_pos * stride_output, pid)

@triton.jit
def assign_exp_kernel(
    exp_ids_list, tok_ids_list, routing,
    num_experts, num_expanded_toks,
    stride_output, stride_input, stride_input_routing,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    if (pid * BLOCK_SIZE >= 0 and pid * BLOCK_SIZE < num_expanded_toks):
        this_tok = tl.load(tok_ids_list + pid * BLOCK_SIZE * stride_input)
        this_exp = tl.load(routing + this_tok * stride_input_routing)
        if (this_exp >= 0 and this_exp < num_experts):
            tl.store(exp_ids_list + pid * stride_output, this_exp)

def counter(routing, num_exps):
    num_toks = routing.shape[0]
    result = torch.zeros(num_exps, device=routing.device, dtype=torch.int32)
    result_padded = torch.zeros(num_exps, device=routing.device, dtype=torch.int32)
    
    grid_1 = lambda meta: (triton.cdiv(num_toks, 1),)
    token_count_kernel[grid_1](result, routing, num_exps, num_toks, result.stride(0), routing.stride(0))
    grid_2 = lambda meta: (triton.cdiv(num_exps, 1),)
    padding_kernel[grid_2](result_padded, result, num_exps, constant.MAX_TOKEN_CHUNK, result_padded.stride(0), result.stride(0))

    accum_list = [0]
    for item in result_padded:
        accum_list.append(accum_list[-1] + item.item())
    exp_start_list = torch.Tensor(accum_list).to(torch.int32).to(routing.device)
    exp_start_list_kept = torch.Tensor(accum_list).to(torch.int32).to(routing.device)

    num_expanded_toks = torch.sum(result_padded).item()
    tok_ids_list = torch.zeros(num_expanded_toks, device=routing.device, dtype=torch.int32) - 1
    tok_ids_list.to(torch.int32).to(routing.device)
    grid_3 = lambda meta: (triton.cdiv(num_toks, 1),)
    assign_tok_kernel[grid_3](tok_ids_list, exp_start_list, routing, num_exps, num_toks, \
                              tok_ids_list.stride(0), exp_start_list.stride(0), routing.stride(0))

    exp_ids_list = torch.zeros(num_expanded_toks // constant.MAX_TOKEN_CHUNK, device=routing.device, dtype=torch.int32) - 1
    exp_ids_list.to(torch.int32).to(routing.device)
    grid_4 = lambda meta: (triton.cdiv(num_expanded_toks, meta['BLOCK_SIZE']), )
    assign_exp_kernel[grid_4](exp_ids_list, tok_ids_list, routing, num_exps, num_expanded_toks, \
                              exp_ids_list.stride(0), tok_ids_list.stride(0), routing.stride(0), BLOCK_SIZE=constant.MAX_TOKEN_CHUNK)

    return num_expanded_toks, exp_start_list_kept, tok_ids_list, exp_ids_list
