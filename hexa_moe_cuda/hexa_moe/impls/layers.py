from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import time
import logging
import torch
from torch import Tensor
from torch.autograd.functional import jacobian as jacobian
from easydict import EasyDict
import torch.nn.functional as F
import torch.distributed as dist

import hexa_moe_cuda

'''
    We only consider these cases for MoE computing:
    (Input, Weight, Bias) -> Output
        1. full: (float32, float32, float32) -> float32
        2. mix: (float16, float32, float32) -> float16
        3. half: (float16, float16, float16) -> float16
'''

from . import losses

class MLP_Implement(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, tokens_input: Tensor, fc1_w, fc2_w, fc1_bias, fc2_bias):
        # Shape of tokens_input is [num_tokens, num_in_dims]
        ctx.config = config
        assert tokens_input.shape[1] == fc1_w.shape[1]
        num_tokens, num_in_dims = tokens_input.shape
        num_out_dims = fc2_w.shape[1]
        assert fc1_w.shape[0] == fc2_w.shape[0]
        num_hid_dims = fc1_w.shape[0]

        hid_feat = torch.matmul(tokens_input, fc1_w.permute(1, 0).to(tokens_input.dtype))

        if ctx.config.has_fc1_bias:
            assert fc1_bias.shape[0] == num_hid_dims
            hid_feat = torch.add(hid_feat, fc1_bias.reshape(1, num_hid_dims).to(tokens_input.dtype))
        
        ctx.save_for_backward(tokens_input, hid_feat, fc1_w, fc2_w)
        
        y = ctx.config.activation_fn(hid_feat)
        y = torch.matmul(y, fc2_w.to(tokens_input.dtype))
        if ctx.config.has_fc2_bias:
            assert fc2_bias.shape[0] == num_out_dims
            y = torch.add(y, fc2_bias.reshape(1, num_out_dims).to(tokens_input.dtype))

        return y.reshape(num_tokens, num_out_dims)

    @staticmethod
    def backward(ctx: Any, j_tokens_output: Tensor): 
        def activate(x):
            return ctx.config.activation_fn(x).sum()
        diff_activate = lambda x: jacobian(activate, x) # In-place differential operation

        tokens_input, hid_feat, fc1_w, fc2_w = ctx.saved_tensors
        
        assert j_tokens_output.shape[0] == tokens_input.shape[0] and tokens_input.shape[1] == fc1_w.shape[1]
        assert fc1_w.shape[0] == fc2_w.shape[0] and hid_feat.shape[1] == fc1_w.shape[0]
        num_tokens, num_in_dims = tokens_input.shape
        num_out_dims = fc2_w.shape[1]
        num_hid_dims = fc1_w.shape[0]
        
        y2 = ctx.config.activation_fn(hid_feat)
        j_y2 = torch.matmul(j_tokens_output, fc2_w.permute(1, 0).to(j_tokens_output.dtype)).reshape(num_tokens, num_hid_dims)
        j_w2 = torch.matmul(y2.permute(1, 0), j_tokens_output).reshape(num_hid_dims, num_out_dims).to(fc2_w.dtype)
        w2_op = dist.all_reduce(j_w2, op=dist.ReduceOp.AVG, async_op=True)
        if ctx.config.has_fc2_bias:
            j_b2 = torch.sum(j_tokens_output, dim=0, keepdim=False).reshape(num_out_dims).to(fc2_w.dtype)
            b2_op = dist.all_reduce(j_b2, op=dist.ReduceOp.AVG, async_op=True)

        j_y1 = j_y2.mul(diff_activate(hid_feat))
        j_x = torch.matmul(j_y1, fc1_w.to(j_tokens_output.dtype)).reshape(num_tokens, num_in_dims)
        j_w1 = torch.matmul(j_y1.permute(1, 0), tokens_input).reshape(num_hid_dims, num_in_dims).to(fc1_w.dtype)
        w1_op = dist.all_reduce(j_w1, op=dist.ReduceOp.AVG, async_op=True)
        if ctx.config.has_fc1_bias:
            j_b1 = torch.sum(j_y1, dim=0, keepdim=False).reshape(num_hid_dims).to(fc1_w.dtype)
            b1_op = dist.all_reduce(j_b1, op=dist.ReduceOp.AVG, async_op=True)

        if ctx.config.has_fc2_bias:
            b2_op.wait()
        else:
            j_b2 = None

        if ctx.config.has_fc1_bias:
            b1_op.wait()
        else:
            j_b1 = None

        w2_op.wait()
        w1_op.wait()

        return (None, j_x, j_w1, j_w2, j_b1, j_b2)
    
class FusedMoE_Shared_CudaCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, tokens_input: Tensor, fc1_w, fc2_w, fc1_bias, fc2_bias):        
        # Shape of tokens_input is [num_tokens, num_in_dims]
        ctx.config = config

        ctx.config.fc1_w_dtypes[ctx.config.stage] = fc1_w.dtype
        ctx.config.fc2_w_dtypes[ctx.config.stage] = fc2_w.dtype
        if ctx.config.has_fc1_bias:
            ctx.config.fc1_b_dtypes[ctx.config.stage] = fc1_bias.dtype
        if ctx.config.has_fc2_bias:
            ctx.config.fc2_b_dtypes[ctx.config.stage] = fc2_bias.dtype

        tokens_input = tokens_input.contiguous()
        assert tokens_input.shape[1] == fc1_w.shape[2]
        num_tokens, num_in_dims = tokens_input.shape
        num_out_dims = fc2_w.shape[2]
        num_hid_dims = ctx.config.hidden_size_list_moe[ctx.config.stage]

        if (ctx.config.data_centric):
            # Receive parameters from different devices.
            ctx.config.fc1_w_op.wait()
            batched_fc1_w = torch.cat(tuple(ctx.config.batched_fc1_w_cache), dim=1).permute(0, 2, 1).contiguous()
            assert num_hid_dims == batched_fc1_w.shape[2]
            ctx.config.fc2_w_op.wait()
            batched_fc2_w = torch.cat(tuple(ctx.config.batched_fc2_w_cache), dim=1).contiguous()
            if ctx.config.has_fc1_bias:
                ctx.config.fc1_b_op.wait()
                batched_fc1_bias = torch.cat(tuple(ctx.config.batched_fc1_bias_cache), dim=1).reshape(-1).contiguous()
            else:
                batched_fc1_bias = torch.empty(0, dtype=fc1_w.dtype, device=tokens_input.device)
            if ctx.config.has_fc2_bias:
                batched_fc2_bias = fc2_bias.reshape(-1).contiguous()
            else:
                batched_fc2_bias = torch.empty(0, dtype=fc2_w.dtype, device=tokens_input.device)

        routing_ids = [item.reshape(-1, 1).to(tokens_input.device) for item in ctx.config.indices_]
        routing_ids = torch.cat(tuple(routing_ids), dim=1)
        tokens_per_exp_ = torch.zeros(ctx.config.num_global_experts, dtype=torch.int32, device=tokens_input.device)
        expanded_tokens_per_exp_ = torch.zeros(ctx.config.num_global_experts + 1, dtype=torch.int32, device=tokens_input.device)
        token_idx_list, expert_idx_list = [], []
        
        # Prepare tensor to store data from all devices to facilitate all_gather
        if not ctx.config.data_centric:
            local_bs_list = [None for _ in range(ctx.config.world_size)]
            dist.all_gather_object(local_bs_list, num_tokens, group=dist.group.WORLD)
            num_tokens_whole = sum(local_bs_list)

            routing_ids_whole = [torch.zeros(bs, routing_ids.shape[1], dtype=torch.int32, device=tokens_input.device) for bs in local_bs_list]
            dist.all_gather(routing_ids_whole, routing_ids, group=dist.group.WORLD, async_op=False)
            routing_ids_whole = torch.cat(tuple(routing_ids_whole), dim=0).contiguous()
            tokens_input_whole = [torch.zeros(bs, num_in_dims, dtype=tokens_input.dtype, device=tokens_input.device) for bs in local_bs_list]
            dist.all_gather(tokens_input_whole, tokens_input, group=dist.group.WORLD, async_op=False)
            tokens_input_whole = torch.cat(tuple(tokens_input_whole), dim=0).contiguous()
            
            loc_dim = ctx.config.local_dim_list[ctx.config.stage]
            
            tokens_out_ = torch.zeros(num_tokens_whole, num_out_dims, dtype=torch.float32, device=tokens_input.device)
            tokens_hid = torch.zeros(routing_ids.shape[1], num_tokens_whole, loc_dim, dtype=torch.float32, device=tokens_input.device)
            
            fc1_w = fc1_w.permute(0, 2, 1).contiguous()
            assert fc1_w.shape[2] == loc_dim
            
            fc2_w = fc2_w.contiguous()
            assert fc2_w.shape[1] == loc_dim
            
            if ctx.config.has_fc1_bias:
                fc1_bias = fc1_bias.contiguous()
                assert fc1_bias.shape[1] == loc_dim
            else:
                fc1_bias = torch.empty(0, dtype=fc1_w.dtype, device=tokens_input.device)
            if ctx.config.has_fc2_bias:
                fc2_bias = fc2_bias.contiguous()
            else:
                fc2_bias = torch.empty(0, dtype=fc2_w.dtype, device=tokens_input.device)
            
            for i in range(routing_ids_whole.shape[1]):
                tokens_per_exp_.zero_()
                expanded_tokens_per_exp_.zero_()
                hexa_moe_cuda.TOKEN_COUNT(tokens_per_exp_, expanded_tokens_per_exp_, \
                                        routing_ids_whole[:, i].contiguous(), num_tokens_whole, ctx.config.num_global_experts)

                token_idx_list.append(torch.zeros(expanded_tokens_per_exp_[-1].item(), dtype=torch.int32, device=tokens_input.device) - 1)
                expert_idx_list.append(torch.zeros(expanded_tokens_per_exp_[-1].item(), dtype=torch.int32, device=tokens_input.device) - 1)
                                
                hexa_moe_cuda.TOKEN_ASSIGN(token_idx_list[i], expert_idx_list[i], \
                                        expanded_tokens_per_exp_, routing_ids_whole[:, i].contiguous(), \
                                        num_tokens_whole, ctx.config.num_global_experts)

            cum_ids = torch.zeros(1 + routing_ids_whole.shape[1], dtype=torch.int32, device=tokens_input.device)
            for i in range(routing_ids_whole.shape[1]):
                assert len(token_idx_list[i]) == len(expert_idx_list[i])
                cum_ids[i + 1] = cum_ids[i] + len(expert_idx_list[i])

            token_idx_list = torch.cat(tuple(token_idx_list), dim=0).contiguous().reshape(-1)
            expert_idx_list = torch.cat(tuple(expert_idx_list), dim=0).contiguous().reshape(-1)

            assert len(token_idx_list) == cum_ids[routing_ids_whole.shape[1]] and \
                len(expert_idx_list) == cum_ids[routing_ids_whole.shape[1]]

            hexa_moe_cuda.ESMM_TENSOR_CORE(tokens_hid.reshape(-1), tokens_input_whole.reshape(-1), fc1_w.reshape(-1), fc1_bias.reshape(-1), \
                token_idx_list, expert_idx_list, cum_ids, \
                tokens_input.dtype == torch.float32, fc1_w.dtype == torch.float32, fc1_bias.dtype == torch.float32, \
                False, num_tokens_whole, routing_ids.shape[1], num_in_dims, loc_dim, ctx.config.num_global_experts)
            tokens_hid = tokens_hid.to(tokens_input.dtype).contiguous()

            ctx.save_for_backward(tokens_input, routing_ids_whole, tokens_hid, cum_ids, token_idx_list, expert_idx_list, \
                    fc1_w.reshape(ctx.config.num_global_experts, num_in_dims, loc_dim), \
                    fc2_w.reshape(ctx.config.num_global_experts, loc_dim, num_out_dims))

            tokens_hid_ = ctx.config.activation_fn(tokens_hid)

            if (ctx.config.local_rank == 0):
                hexa_moe_cuda.ESMM_TENSOR_CORE(tokens_out_.reshape(-1), tokens_hid_.reshape(-1), fc2_w.reshape(-1), fc2_bias.reshape(-1), \
                    token_idx_list, expert_idx_list, cum_ids, \
                    tokens_hid_.dtype == torch.float32, fc2_w.dtype == torch.float32, fc2_bias.dtype == torch.float32, \
                    True, num_tokens_whole, routing_ids.shape[1], loc_dim, num_out_dims, ctx.config.num_global_experts)
            else:
                null_bias = torch.empty(0, dtype=fc2_w.dtype, device=tokens_input.device)
                hexa_moe_cuda.ESMM_TENSOR_CORE(tokens_out_.reshape(-1), tokens_hid_.reshape(-1), fc2_w.reshape(-1), null_bias, \
                    token_idx_list, expert_idx_list, cum_ids, \
                    tokens_hid_.dtype == torch.float32, fc2_w.dtype == torch.float32, null_bias.dtype == torch.float32, \
                    True, num_tokens_whole, routing_ids.shape[1], loc_dim, num_out_dims, ctx.config.num_global_experts)
            tokens_out_ = tokens_out_.to(tokens_input.dtype).contiguous()
            
            ctr = 0
            for k in range(ctx.config.world_size):
                comm_data = tokens_out_[ctr : ctr + local_bs_list[k], :].contiguous()
                ctr += local_bs_list[k]
                dist.reduce(comm_data, dst=k, op=dist.ReduceOp.SUM, async_op=False)
                if (ctx.config.local_rank == k):
                    tokens_out = comm_data
        else:
            tokens_out = torch.zeros(num_tokens, num_out_dims, dtype=torch.float32, device=tokens_input.device)
            tokens_hid = torch.zeros(routing_ids.shape[1], num_tokens, num_hid_dims, dtype=torch.float32, device=tokens_input.device)
            
            for i in range(routing_ids.shape[1]):
                tokens_per_exp_.zero_()
                expanded_tokens_per_exp_.zero_()
                hexa_moe_cuda.TOKEN_COUNT(tokens_per_exp_, expanded_tokens_per_exp_, \
                                        routing_ids[:, i].contiguous(), num_tokens, ctx.config.num_global_experts)
                token_idx_list.append(torch.zeros(expanded_tokens_per_exp_[-1].item(), dtype=torch.int32, device=tokens_input.device) - 1)
                expert_idx_list.append(torch.zeros(expanded_tokens_per_exp_[-1].item(), dtype=torch.int32, device=tokens_input.device) - 1)
                
                hexa_moe_cuda.TOKEN_ASSIGN(token_idx_list[i], expert_idx_list[i], \
                                        expanded_tokens_per_exp_, routing_ids[:, i].contiguous(), \
                                        num_tokens, ctx.config.num_global_experts)

            cum_ids = torch.zeros(1 + routing_ids.shape[1], dtype=torch.int32, device=tokens_input.device)
            for i in range(routing_ids.shape[1]):
                assert len(token_idx_list[i]) == len(expert_idx_list[i])
                cum_ids[i + 1] = cum_ids[i] + len(expert_idx_list[i])

            token_idx_list = torch.cat(tuple(token_idx_list), dim=0).contiguous().reshape(-1)
            expert_idx_list = torch.cat(tuple(expert_idx_list), dim=0).contiguous().reshape(-1)

            assert len(token_idx_list) == cum_ids[routing_ids.shape[1]] and \
                len(expert_idx_list) == cum_ids[routing_ids.shape[1]]

            hexa_moe_cuda.ESMM_TENSOR_CORE(tokens_hid.reshape(-1), tokens_input.reshape(-1), batched_fc1_w.reshape(-1), batched_fc1_bias, \
                token_idx_list, expert_idx_list, cum_ids, \
                tokens_input.dtype == torch.float32, batched_fc1_w.dtype == torch.float32, batched_fc1_bias.dtype == torch.float32, \
                False, num_tokens, routing_ids.shape[1], num_in_dims, num_hid_dims, ctx.config.num_global_experts)
            tokens_hid = tokens_hid.to(tokens_input.dtype).contiguous()

            if ctx.config.stage >= 1:
                ctx.save_for_backward(tokens_input, routing_ids, tokens_hid, cum_ids, \
                        token_idx_list, expert_idx_list, ctx.config.prev_fc1_w, ctx.config.prev_fc2_w)
            else:
                ctx.save_for_backward(tokens_input, routing_ids, tokens_hid, cum_ids, \
                        token_idx_list, expert_idx_list, None, None)

            tokens_hid_ = ctx.config.activation_fn(tokens_hid)

            hexa_moe_cuda.ESMM_TENSOR_CORE(tokens_out.reshape(-1), tokens_hid_.reshape(-1), batched_fc2_w.reshape(-1), batched_fc2_bias, \
                token_idx_list, expert_idx_list, cum_ids, \
                tokens_hid_.dtype == torch.float32, batched_fc2_w.dtype == torch.float32, batched_fc2_bias.dtype == torch.float32, \
                True, num_tokens, routing_ids.shape[1], num_hid_dims, num_out_dims, ctx.config.num_global_experts)
            tokens_out = tokens_out.to(tokens_input.dtype).contiguous()

        return tokens_out.reshape(num_tokens, num_out_dims)

    @staticmethod
    def backward(ctx: Any, j_tokens_output: Tensor):        
        def activate(x):
            return ctx.config.activation_fn(x).sum()
        diff_activate = lambda x: jacobian(activate, x) # In-place per-element differentiation.
        
        j_tokens_output = j_tokens_output.contiguous()
        if ctx.config.data_centric:
            tokens_input, routing_ids, y1, cum_ids, tok_idx_list, exp_idx_list, prev_fc1_w, prev_fc2_w = ctx.saved_tensors
        else:
            tokens_input, routing_ids_whole, y1, cum_ids, tok_idx_list, exp_idx_list, fc1_w, fc2_w = ctx.saved_tensors
        
        assert j_tokens_output.shape[0] == tokens_input.shape[0] and ctx.config.input_dim_list_moe[ctx.config.stage] == tokens_input.shape[1]
        num_global_experts = ctx.config.num_global_experts
        num_tokens = tokens_input.shape[0]
        num_in_dims = ctx.config.input_dim_list_moe[ctx.config.stage]
        num_hid_dims, num_out_dims = ctx.config.hidden_size_list_moe[ctx.config.stage], ctx.config.output_dim_list_moe[ctx.config.stage]

        ctr = 0
        local_start_list, local_end_list = [], []
        for i in range(ctx.config.world_size):
            local_start_list.append(ctr)
            ctr += ctx.config.local_dim_list_all[i][ctx.config.stage]
            local_end_list.append(ctr)

        if ctx.config.data_centric:            
            # Receive parameters from different devices.
            ctx.config.fc1_w_op.wait()
            batched_fc1_w = torch.cat(tuple(ctx.config.batched_fc1_w_cache), dim=1).permute(0, 2, 1).contiguous()
            ctx.config.fc2_w_op.wait()
            batched_fc2_w = torch.cat(tuple(ctx.config.batched_fc2_w_cache), dim=1).contiguous()
            assert batched_fc1_w.shape[0] == batched_fc2_w.shape[0] and batched_fc1_w.shape[0] == num_global_experts
            assert num_hid_dims == batched_fc2_w.shape[1] and num_out_dims == batched_fc2_w.shape[2]

            # Gather all the param chunks for the next stage (previous one in the pipeline)
            if ctx.config.stage >= 1:
                # All gather
                depth_idx = ctx.config.stage - 1 # Send the previous layer.

                ctx.config.batched_fc1_w_cache_pool.zero_()
                ctx.config.batched_fc1_w_cache = \
                    [ctx.config.batched_fc1_w_cache_pool[:, ctx.config.ids_list_global[i][depth_idx], \
                        :ctx.config.input_dim_list_moe[depth_idx]].contiguous() for i in range(ctx.config.world_size)]
                
                ctx.config.fc1_w_op = dist.all_gather(ctx.config.batched_fc1_w_cache, \
                    prev_fc1_w, group=dist.group.WORLD, async_op=True)

                ctx.config.batched_fc2_w_cache_pool.zero_()
                ctx.config.batched_fc2_w_cache = \
                    [ctx.config.batched_fc2_w_cache_pool[:, ctx.config.ids_list_global[i][depth_idx], \
                        :ctx.config.output_dim_list_moe[depth_idx]].contiguous() for i in range(ctx.config.world_size)]
                ctx.config.fc2_w_op = dist.all_gather(ctx.config.batched_fc2_w_cache, \
                    prev_fc2_w, group=dist.group.WORLD, async_op=True)

        if not ctx.config.data_centric:
            loc_dim = ctx.config.local_dim_list[ctx.config.stage]
            
            local_bs_list = [None for _ in range(ctx.config.world_size)]
            dist.all_gather_object(local_bs_list, num_tokens, group=dist.group.WORLD)
            num_tokens_whole = sum(local_bs_list)
            
            local_bs_start, local_bs_end = 0, local_bs_list[0]
            if ctx.config.local_rank > 0:
                local_bs_start = sum(local_bs_list[:ctx.config.local_rank])
                local_bs_end = local_bs_start + local_bs_list[ctx.config.local_rank]
                        
            j_tokens_output_whole = [torch.zeros(bs, num_out_dims, dtype=j_tokens_output.dtype, device=j_tokens_output.device) for bs in local_bs_list]
            dist.all_gather(j_tokens_output_whole, j_tokens_output, group=dist.group.WORLD, async_op=False)
            j_tokens_output_whole = torch.cat(tuple(j_tokens_output_whole), dim=0)
            tokens_input_whole = [torch.zeros(bs, num_in_dims, dtype=tokens_input.dtype, device=tokens_input.device) for bs in local_bs_list]
            dist.all_gather(tokens_input_whole, tokens_input, group=dist.group.WORLD, async_op=False)
            tokens_input_whole = torch.cat(tuple(tokens_input_whole), dim=0)
            
            y1_grad = torch.zeros(routing_ids_whole.shape[1], num_tokens_whole, loc_dim, dtype=torch.float32, device=j_tokens_output.device)
            fc2_w_grad = torch.zeros(num_global_experts, num_out_dims, loc_dim, dtype=torch.float32, device=j_tokens_output.device)
            y2_tmp = ctx.config.activation_fn(y1).contiguous()
            
            if ctx.config.has_fc2_bias:
                fc2_b_grad = torch.zeros(num_global_experts, num_out_dims, dtype=torch.float32, device=j_tokens_output.device)
                hexa_moe_cuda.FUSED_GRAD(fc2_b_grad.reshape(-1), y1_grad.reshape(-1), fc2_w_grad.reshape(-1), \
                        j_tokens_output_whole.reshape(-1), fc2_w.permute(0, 2, 1).contiguous().reshape(-1), y2_tmp.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output_whole.dtype == torch.float32, fc2_w.dtype == torch.float32, \
                        False, num_tokens_whole, routing_ids_whole.shape[1], loc_dim, num_out_dims, num_global_experts)
                fc2_b_grad = fc2_b_grad.to(j_tokens_output.dtype).contiguous()
            else:
                fc2_b_grad = None
                hexa_moe_cuda.FUSED_GRAD_NO_BIAS(y1_grad.reshape(-1), fc2_w_grad.reshape(-1), \
                        j_tokens_output_whole.reshape(-1), fc2_w.permute(0, 2, 1).contiguous().reshape(-1), y2_tmp.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output_whole.dtype == torch.float32, fc2_w.dtype == torch.float32, \
                        False, num_tokens_whole, routing_ids_whole.shape[1], loc_dim, num_out_dims, num_global_experts)
            y1_grad = y1_grad.to(j_tokens_output.dtype).contiguous()
            fc2_w_grad = fc2_w_grad.to(ctx.config.fc2_w_dtypes[ctx.config.stage]).contiguous()

            y1_grad = y1_grad.mul(diff_activate(y1)).contiguous()
            fc2_w_grad = fc2_w_grad.permute(0, 2, 1).contiguous()

            fc1_w_grad = torch.zeros(num_global_experts, loc_dim, num_in_dims, dtype=torch.float32, device=j_tokens_output.device)
            j_tokens_input_ = torch.zeros(num_tokens_whole, num_in_dims, dtype=torch.float32, device=j_tokens_output.device)
            
            if ctx.config.has_fc1_bias:
                fc1_b_grad = torch.zeros(num_global_experts, loc_dim, dtype=torch.float32, device=j_tokens_output.device)
                hexa_moe_cuda.FUSED_GRAD(fc1_b_grad.reshape(-1), j_tokens_input_.reshape(-1), fc1_w_grad.reshape(-1), \
                        y1_grad.contiguous().reshape(-1), fc1_w.permute(0, 2, 1).contiguous().reshape(-1), tokens_input_whole.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output.dtype == torch.float32, fc1_w.dtype == torch.float32, \
                        True, num_tokens_whole, routing_ids_whole.shape[1], num_in_dims, loc_dim, num_global_experts)
                fc1_b_grad = fc1_b_grad.to(j_tokens_output.dtype).contiguous()
            else:
                fc1_b_grad = None
                hexa_moe_cuda.FUSED_GRAD_NO_BIAS(j_tokens_input_.reshape(-1), fc1_w_grad.reshape(-1), \
                        y1_grad.contiguous().reshape(-1), fc1_w.permute(0, 2, 1).contiguous().reshape(-1), tokens_input_whole.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output.dtype == torch.float32, fc1_w.dtype == torch.float32, \
                        True, num_tokens_whole, routing_ids_whole.shape[1], num_in_dims, loc_dim, num_global_experts)
            j_tokens_input_ = j_tokens_input_.to(j_tokens_output.dtype).contiguous()
            fc1_w_grad = fc1_w_grad.to(ctx.config.fc1_w_dtypes[ctx.config.stage]).contiguous()
            
            for k in range(ctx.config.world_size):
                comm_data = j_tokens_input_[local_bs_start : local_bs_end, :].contiguous()
                dist.reduce(comm_data, dst=k, op=dist.ReduceOp.SUM, async_op=False)
                if (ctx.config.local_rank == k):
                    j_tokens_input = comm_data
        else:
            y1_grad = torch.zeros(routing_ids.shape[1], num_tokens, num_hid_dims, dtype=torch.float32, device=j_tokens_output.device)
            fc2_w_grad_whole = torch.zeros(num_global_experts, num_out_dims, num_hid_dims, dtype=torch.float32, device=j_tokens_output.device)
            
            y2_tmp = ctx.config.activation_fn(y1).contiguous()
            
            if ctx.config.has_fc2_bias:
                fc2_b_grad = torch.zeros(num_global_experts, num_out_dims, dtype=torch.float32, device=j_tokens_output.device)
                hexa_moe_cuda.FUSED_GRAD(fc2_b_grad.reshape(-1), y1_grad.reshape(-1), fc2_w_grad_whole.reshape(-1), \
                        j_tokens_output.reshape(-1), batched_fc2_w.permute(0, 2, 1).contiguous().reshape(-1), y2_tmp.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output.dtype == torch.float32, batched_fc2_w.dtype == torch.float32, \
                        False, num_tokens, routing_ids.shape[1], num_hid_dims, num_out_dims, num_global_experts)
                fc2_b_grad = fc2_b_grad.to(j_tokens_output.dtype).contiguous()
                fc2_b_op = dist.all_reduce(fc2_b_grad, op=dist.ReduceOp.AVG, async_op=True)
            else:
                hexa_moe_cuda.FUSED_GRAD_NO_BIAS(y1_grad.reshape(-1), fc2_w_grad_whole.reshape(-1), \
                        j_tokens_output.reshape(-1), batched_fc2_w.permute(0, 2, 1).contiguous().reshape(-1), y2_tmp.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output.dtype == torch.float32, batched_fc2_w.dtype == torch.float32, \
                        False, num_tokens, routing_ids.shape[1], num_hid_dims, num_out_dims, num_global_experts)
            y1_grad = y1_grad.to(j_tokens_output.dtype).contiguous()
            fc2_w_grad_whole = fc2_w_grad_whole.to(ctx.config.fc2_w_dtypes[ctx.config.stage]).contiguous()
            
            y1_grad = y1_grad.mul(diff_activate(y1)).contiguous()
            fc2_w_grad_whole = fc2_w_grad_whole.permute(0, 2, 1).contiguous()

            fc2_w_op = dist.all_reduce(fc2_w_grad_whole, op=dist.ReduceOp.AVG, async_op=True)

            fc1_w_grad_whole = torch.zeros(num_global_experts, num_hid_dims, num_in_dims, dtype=torch.float32, device=j_tokens_output.device)
            j_tokens_input = torch.zeros(num_tokens, num_in_dims, dtype=torch.float32, device=j_tokens_output.device)
            
            if ctx.config.has_fc1_bias:
                fc1_b_grad_whole = torch.zeros(num_global_experts, num_hid_dims, dtype=torch.float32, device=j_tokens_output.device)
                hexa_moe_cuda.FUSED_GRAD(fc1_b_grad_whole.reshape(-1), j_tokens_input.reshape(-1), fc1_w_grad_whole.reshape(-1), \
                        y1_grad.contiguous().reshape(-1), batched_fc1_w.permute(0, 2, 1).contiguous().reshape(-1), tokens_input.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output.dtype == torch.float32, batched_fc1_w.dtype == torch.float32, \
                        True, num_tokens, routing_ids.shape[1], num_in_dims, num_hid_dims, num_global_experts)
                fc1_b_grad_whole = fc1_b_grad_whole.to(j_tokens_output.dtype).contiguous()
                fc1_bias_op = dist.all_reduce(fc1_b_grad_whole, op=dist.ReduceOp.AVG, async_op=True)
            else:
                hexa_moe_cuda.FUSED_GRAD_NO_BIAS(j_tokens_input.reshape(-1), fc1_w_grad_whole.reshape(-1), \
                        y1_grad.contiguous().reshape(-1), batched_fc1_w.permute(0, 2, 1).contiguous().reshape(-1), tokens_input.reshape(-1), \
                        tok_idx_list, exp_idx_list, cum_ids, \
                        j_tokens_output.dtype == torch.float32, batched_fc1_w.dtype == torch.float32, \
                        True, num_tokens, routing_ids.shape[1], num_in_dims, num_hid_dims, num_global_experts)
            fc1_w_grad_whole = fc1_w_grad_whole.to(ctx.config.fc1_w_dtypes[ctx.config.stage]).contiguous()
            j_tokens_input = j_tokens_input.to(j_tokens_output.dtype).contiguous()

            fc1_w_op = dist.all_reduce(fc1_w_grad_whole, op=dist.ReduceOp.AVG, async_op=True)

            if ctx.config.has_fc2_bias:
                fc2_b_op.wait()
            else:
                fc2_b_grad = None
            
            if ctx.config.has_fc1_bias:
                fc1_bias_op.wait()
                fc1_b_grad = fc1_b_grad_whole[:, local_start_list[ctx.config.local_rank] : local_end_list[ctx.config.local_rank]]
            else:
                fc1_b_grad = None

            fc2_w_op.wait()
            fc2_w_grad = fc2_w_grad_whole[:, local_start_list[ctx.config.local_rank] : local_end_list[ctx.config.local_rank], :]

            fc1_w_op.wait()
            fc1_w_grad = fc1_w_grad_whole[:, local_start_list[ctx.config.local_rank] : local_end_list[ctx.config.local_rank], :]

        ctx.config.stage = ctx.config.stage - 1

        return (None, j_tokens_input, fc1_w_grad, fc2_w_grad, fc1_b_grad, fc2_b_grad)

def simplified_extract_critical(scores, top_k, loss_fn=losses.gshard_loss):
    num_global_experts = int(scores.size(1))
    top_k, top_k_original = min(top_k, num_global_experts), top_k
    topk_indices = torch.topk(scores, top_k, dim=1).indices

    indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)]
    
    l_loss = loss_fn(scores, topk_indices) if loss_fn is not None else None
    
    return indices_s, l_loss