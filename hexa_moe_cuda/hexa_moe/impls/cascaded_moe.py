from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import copy
import os
import re
import time
import logging 
import collections
import importlib

import time
import torch
from torch import Tensor
from easydict import EasyDict
from timm.models.layers import trunc_normal_
import torch.distributed as dist
from torch.nn import ModuleList
import torch.nn.functional as F

from ..impls.layers import simplified_extract_critical, FusedMoE_Shared_CudaCore, MLP_Implement
from . import losses

def cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        # casts inputs to autocast dtype which enables all2all to be done in low precision
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        elif tensor.device.type == 'xpu':
            dtype = torch.xpu.get_autocast_xpu_dtype()  # type: ignore[attr-defined]
        elif tensor.device.type == 'hpu':
            dtype = torch.hpu.get_autocast_hpu_dtype()  # type: ignore[attr-defined]
        else:
            raise RuntimeError('User specified autocast device_type must be \'cuda\' or \'cpu\'')
        return tensor.to(dtype=dtype)

class MoE_Buffer(torch.nn.Module):
    """
        CPU Buffer for tensor parallelized MoE
    """
    def __init__(
            self, 
            model_dim_list_moe: list,
            num_global_experts: int, 
            total_depth_moe: int,
            mlp_ratio=4.,
            output_dim_list_moe=None,
            mlp_fc1_bias=True, 
            mlp_fc2_bias=True
    ):
        super(MoE_Buffer, self).__init__()
        self.total_depth_moe = total_depth_moe
        self.input_dim_list_moe = model_dim_list_moe
        self.hidden_size_list_moe = [int(mlp_ratio * dim) for dim in model_dim_list_moe]
        self.num_global_experts = num_global_experts
        self.output_dim_list_moe = output_dim_list_moe or model_dim_list_moe
        self.has_fc1_bias = mlp_fc1_bias
        self.has_fc2_bias = mlp_fc2_bias

        self.batched_fc1_w = \
            torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                self.hidden_size_list_moe[i], self.input_dim_list_moe[i])) for i in range(self.total_depth_moe)])
        self.batched_fc2_w = \
            torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                self.hidden_size_list_moe[i], self.output_dim_list_moe[i])) for i in range(self.total_depth_moe)])
        if self.has_fc1_bias:
            self.batched_fc1_bias = \
                torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                    self.hidden_size_list_moe[i])) for i in range(self.total_depth_moe)])
        else:
            self.batched_fc1_bias = \
                torch.nn.ParameterList([None for _ in range(self.total_depth_moe)])
        if self.has_fc2_bias:
            self.batched_fc2_bias = \
                torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                    self.output_dim_list_moe[i])) for i in range(self.total_depth_moe)])
        else:
            self.batched_fc2_bias = \
                torch.nn.ParameterList([None for _ in range(self.total_depth_moe)])

class MoE_Cascaded(torch.nn.Module):
    """
        Tutel optimized MOELayer
    """
    def __init__(
        self,
        gate_type,
        model_dim_list: list,
        moe_idx_list: list,
        num_global_experts: int,
        total_depth: int,
        data_centric: True,
        mlp_ratio=4.,
        mlp_proportion=None,
        output_dim_list=None,
        mlp_fc1_bias=True, 
        mlp_fc2_bias=True,
        activation_fn=None, 
        activation_fn_with_self=None,
        result_func=None,
        group=None,
        seeds=None,
        a2a_ffn_overlap_degree=1,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        parallel_type='adaptive:1',
        use_2dh=False
    ):
        super().__init__()
        group = group or dist.group.WORLD
        self.group = group
        self.result_func = result_func
        self.skip_moe = (int(os.environ.get('SKIP_MOE', '0')) != 0)

        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss

        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh

        if seeds is not None and seeds[1] is not None:
            torch.manual_seed(seeds[1])

        mlp_idx_list = []
        for i in range(total_depth):
            if i not in moe_idx_list:
                mlp_idx_list.append(i)

        model_dim_list_moe = [model_dim_list[i] for i in moe_idx_list]
        if output_dim_list:
            output_dim_list_moe = [output_dim_list[i] for i in moe_idx_list]
        else:
            output_dim_list_moe = [model_dim_list[i] for i in moe_idx_list]

        model_dim_list_mlp = [model_dim_list[i] for i in mlp_idx_list]
        if output_dim_list:
            output_dim_list_mlp = [output_dim_list[i] for i in mlp_idx_list]
        else:
            output_dim_list_mlp = [model_dim_list[i] for i in mlp_idx_list]

        total_depth_mlp = len(mlp_idx_list)
        total_depth_moe = len(moe_idx_list)

        # Variables for cascaded MoE
        self.is_cuda = None
        self.kernel_pool = dict()
        self.has_fc1_bias = mlp_fc1_bias
        self.has_fc2_bias = mlp_fc2_bias
        self.moe_idx_list = moe_idx_list
        self.mlp_idx_list = mlp_idx_list
        self.total_depth_moe = total_depth_moe
        self.total_depth_mlp = total_depth_mlp
        self.input_dim_list_moe = model_dim_list_moe
        self.output_dim_list_moe = output_dim_list_moe or model_dim_list_moe
        self.hidden_size_list_moe = [int(dim * mlp_ratio) for dim in model_dim_list_moe]
        self.input_dim_list_mlp = model_dim_list_mlp
        self.output_dim_list_mlp = output_dim_list_mlp or model_dim_list_mlp
        self.hidden_size_list_mlp = [int(dim * mlp_ratio) for dim in model_dim_list_mlp]
        self.num_global_experts = num_global_experts
        self.world_size = dist.get_world_size(self.group)
        self.local_rank = dist.get_rank(self.group)
        if self.num_global_experts < self.world_size:
            self.sharded_count = self.world_size // self.num_global_experts
        else:
            self.sharded_count = 1
        
        if mlp_proportion == None:
            self.mlp_proportion = [float(1 / self.world_size)] * self.world_size # Equal division by default.
        else:
            assert len(mlp_proportion) == self.world_size and abs(sum(mlp_proportion) - 1) < 0.001, "incorrect mlp_proportion"
            self.mlp_proportion = mlp_proportion

        self.auto_parallel, self.adaptive_degree, self.use_model_parallel = False, self.sharded_count, True
        self.valid_rs = [0] + [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]

        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            self.adaptive_degree = min(max(self.adaptive_degree, 0), self.sharded_count)
            if self.adaptive_degree not in self.valid_rs:
                raise Exception("Unexpected value of adaptive_degree: %d, expecting a candidate within %s." % (self.adaptive_degree, self.valid_rs))
        elif self.sharded_count == 1:
            pass
        elif parallel_type in ('data', 'model'):
            self.adaptive_degree = 1 if parallel_type == 'data' else self.sharded_count
        elif parallel_type == 'auto':
            self.adaptive_degree = 1
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)

        self.local_dim_list_all, self.ids_list_global, self.local_dim_bound_list = self.dim_split(model_dim_list_moe, mlp_ratio)
        self.local_dim_list = self.local_dim_list_all[self.local_rank]

        for device_idx in range(len(self.local_dim_list_all)):
            print("Device ", device_idx, " ", self.local_dim_list_all[device_idx])

        self.input_dim_bound = max(self.input_dim_list_moe)
        self.hidden_size_bound = max(self.hidden_size_list_moe)
        self.output_dim_bound = max(self.output_dim_list_moe)
        self.local_dim_bound = max(self.local_dim_list)

        self.batched_fc1_w = \
            torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                self.local_dim_bound_list[i], self.input_dim_list_moe[i])) for i in range(self.total_depth_moe)])
        self.batched_fc2_w = \
            torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                self.local_dim_bound_list[i], self.output_dim_list_moe[i])) for i in range(self.total_depth_moe)])
        if self.has_fc1_bias:
            self.batched_fc1_bias = \
                torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                    self.local_dim_bound_list[i])) for i in range(self.total_depth_moe)])
        else:
            self.batched_fc1_bias = torch.nn.ParameterList([None for _ in range(self.total_depth_moe)])
        if self.has_fc2_bias:
            self.batched_fc2_bias = \
                torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.num_global_experts, \
                    self.output_dim_list_moe[i])) for i in range(self.total_depth_moe)])
        else:
            self.batched_fc2_bias = torch.nn.ParameterList([None for _ in range(self.total_depth_moe)])

        self.mlp_fc1_w = \
            torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.hidden_size_list_mlp[i], \
                self.input_dim_list_mlp[i])) for i in range(self.total_depth_mlp)])
        self.mlp_fc2_w = \
            torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.hidden_size_list_mlp[i], \
                self.output_dim_list_mlp[i])) for i in range(self.total_depth_mlp)])
        if self.has_fc1_bias:
            self.mlp_fc1_b = \
                torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.hidden_size_list_mlp[i])) for i in range(self.total_depth_mlp)])
        else:
            self.mlp_fc1_b = torch.nn.ParameterList([None for _ in range(self.total_depth_mlp)])
        if self.has_fc2_bias:
            self.mlp_fc2_b = \
                torch.nn.ParameterList([torch.nn.Parameter(torch.empty(self.output_dim_list_mlp[i])) for i in range(self.total_depth_mlp)])
        else:
            self.mlp_fc2_b = torch.nn.ParameterList([None for _ in range(self.total_depth_mlp)])

        self.dispatch_params = EasyDict()
        
        if data_centric:
            self.dispatch_params.batched_fc1_w_cache = None
            self.dispatch_params.batched_fc2_w_cache = None
            self.dispatch_params.batched_fc1_bias_cache = None
            self.dispatch_params.prev_fc1_w = None
            self.dispatch_params.prev_fc2_w = None

            self.dispatch_params.batched_fc1_w_cache_pool = torch.zeros(self.num_global_experts, self.hidden_size_bound, self.input_dim_bound)
            self.dispatch_params.batched_fc2_w_cache_pool = torch.zeros(self.num_global_experts, self.hidden_size_bound, self.output_dim_bound)
            if self.has_fc1_bias:
                self.dispatch_params.batched_fc1_bias_cache_pool = torch.zeros(self.num_global_experts, self.hidden_size_bound)
            else:
                self.dispatch_params.batched_fc1_bias_cache_pool = None

        if activation_fn_with_self is not None:
            assert activation_fn is None, "Option `activation_fn_with_self` has been specified, please keep exactly one of them."
            activation_fn = lambda x: activation_fn_with_self(x, self)
        if activation_fn is None:
            activation_fn = lambda x: F.relu(x)
        
        self.dispatch_params.activation_fn = activation_fn
        self.dispatch_params.stage = 0
        self.dispatch_params.world_size = self.world_size
        self.dispatch_params.local_rank = self.local_rank
        self.dispatch_params.has_fc1_bias = mlp_fc1_bias
        self.dispatch_params.has_fc2_bias = mlp_fc2_bias
        self.dispatch_params.input_dim_list_moe = self.input_dim_list_moe
        self.dispatch_params.output_dim_list_moe = self.output_dim_list_moe
        self.dispatch_params.hidden_size_list_moe = self.hidden_size_list_moe
        self.dispatch_params.local_dim_list = self.local_dim_list
        self.dispatch_params.local_dim_list_all = self.local_dim_list_all
        self.dispatch_params.ids_list_global = self.ids_list_global
        self.dispatch_params.num_global_experts = int(num_global_experts)
        self.dispatch_params.fc1_w_op = None
        self.dispatch_params.fc1_b_op = None
        self.dispatch_params.fc2_w_op = None
        self.dispatch_params.data_centric = data_centric
        self.dispatch_params.fc1_w_dtypes = [''] * self.total_depth_moe
        self.dispatch_params.fc2_w_dtypes = [''] * self.total_depth_moe
        self.dispatch_params.fc1_b_dtypes = [''] * self.total_depth_moe
        self.dispatch_params.fc2_b_dtypes = [''] * self.total_depth_moe

        self._init_moe()

        if isinstance(gate_type, str):
            assert re.match(r'^Top[0-9]+Gate$', gate_type), "Unrecognized gate_type: %s" % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}

        if not isinstance(gate_type, list):
            gate_type = [gate_type]

        self.gates = []
        self.gate_len = len(gate_type)
        for j in range(self.total_depth_moe):
            for gi, single_gate_type in enumerate(gate_type):
                gate_type_ = single_gate_type['type']
                single_gate_type.pop('type')
                assert re.match(r'[a-zA-Z0-9\_]+', gate_type_), "Gate type must only include digits, letters and underline characters."

                if seeds is not None and seeds[0] is not None:
                    torch.manual_seed(seeds[0] + gi)
                try:
                    single_gate = importlib.import_module(f'...gates.{gate_type_}', __name__)
                except ModuleNotFoundError:
                    raise Exception("Unrecognized gate_type: %s" % gate_type_)

                gate_module = single_gate.Gate(model_dim=self.input_dim_list_moe[j], num_global_experts=self.num_global_experts, **single_gate_type)
                if not hasattr(gate_module, 'gate_noise'):
                    gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)

                single_gate_type['type'] = 'top'
                self.gates += [gate_module]

        self.gates = ModuleList(self.gates)

        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

    def _init_moe(self):
        for i in range(self.total_depth_moe):
            trunc_normal_(self.batched_fc1_w[i].data, std=0.005)
            trunc_normal_(self.batched_fc2_w[i].data, std=0.005)
            if self.has_fc1_bias:
                torch.nn.init.constant_(self.batched_fc1_bias[i].data, 0)
            if self.has_fc2_bias:
                torch.nn.init.constant_(self.batched_fc2_bias[i].data, 0)

        for i in range(self.total_depth_mlp):
            trunc_normal_(self.mlp_fc1_w[i].data, std=0.005)
            trunc_normal_(self.mlp_fc2_w[i].data, std=0.005)
            if self.has_fc1_bias:
                torch.nn.init.constant_(self.mlp_fc1_b[i].data, 0)
            if self.has_fc2_bias:
                torch.nn.init.constant_(self.mlp_fc2_b[i].data, 0)

    def dim_split(self, dims_list, rto):
        tmp_dims_list_global = [[] for _ in range(self.world_size)]
        tmp_ids_list_global = [[] for _ in range(self.world_size)]
        tmp_dims_bound_list = []

        for dim_size in dims_list:
            tmp_dims_list = [int(item * dim_size) for item in self.mlp_proportion]
            this_idx = 0
            while (sum(tmp_dims_list) < dim_size):
                tmp_dims_list[this_idx] += 1
                this_idx = (this_idx + 1) % self.world_size

            start_idx = 0
            for i in range(self.world_size):
                tmp_dims_list_global[i].append(int(rto * tmp_dims_list[i]))
                tmp_ids_list_global[i].append([*range(start_idx, start_idx + int(rto * tmp_dims_list[i]))])
                start_idx = start_idx + int(rto * tmp_dims_list[i])

            tmp_dims_bound_list.append(int(rto * max(tmp_dims_list)))

        return tmp_dims_list_global, tmp_ids_list_global, tmp_dims_bound_list

    def dim_chunk(self, hidden_size_list, world_size):
        tmp_dims_list_global = [[] for _ in range(world_size)]
        tmp_ids_list_global = [[] for _ in range(world_size)]

        for hidden_size in hidden_size_list:
            # Scatter hidden_size to all devices
            base_dim = hidden_size // world_size
            portion_1, portion_2 = (hidden_size - base_dim * world_size), world_size - (hidden_size - base_dim * world_size)
            hidden_dims_list = [(base_dim + 1) for _ in range(portion_1)] + [base_dim for _ in range(portion_2)]
            start_idx = 0
            for i in range(world_size):
                tmp_dims_list_global[i].append(hidden_dims_list[i])
                tmp_ids_list_global[i].append([*range(start_idx, start_idx+hidden_dims_list[i])])
                start_idx = start_idx + hidden_dims_list[i]

        return tmp_dims_list_global, tmp_ids_list_global

    def cuda_cache(self):
        if self.dispatch_params.data_centric:
            self.dispatch_params.batched_fc1_w_cache_pool = self.dispatch_params.batched_fc1_w_cache_pool.cuda()
            self.dispatch_params.batched_fc2_w_cache_pool = self.dispatch_params.batched_fc2_w_cache_pool.cuda()
            if self.has_fc1_bias:
                self.dispatch_params.batched_fc1_bias_cache_pool = self.dispatch_params.batched_fc1_bias_cache_pool.cuda()

    def cache_gather(self, depth_idx=0):
        # Split the cache pool

        self.dispatch_params.batched_fc1_w_cache_pool.zero_()
        # Split a cache list from the cache pool
        self.dispatch_params.batched_fc1_w_cache = \
            [self.dispatch_params.batched_fc1_w_cache_pool[:, self.ids_list_global[i][depth_idx], \
                :self.input_dim_list_moe[depth_idx]].contiguous() for i in range(self.world_size)]
        self.dispatch_params.fc1_w_op = dist.all_gather(self.dispatch_params.batched_fc1_w_cache, \
            self.batched_fc1_w[depth_idx].data[:, :self.local_dim_list[depth_idx], :].contiguous(), group=dist.group.WORLD, async_op=True)

        self.dispatch_params.batched_fc2_w_cache_pool.zero_()
        self.dispatch_params.batched_fc2_w_cache = \
            [self.dispatch_params.batched_fc2_w_cache_pool[:, self.ids_list_global[i][depth_idx], \
                :self.output_dim_list_moe[depth_idx]].contiguous() for i in range(self.world_size)]
        self.dispatch_params.fc2_w_op = dist.all_gather(self.dispatch_params.batched_fc2_w_cache, \
            self.batched_fc2_w[depth_idx].data[:, :self.local_dim_list[depth_idx], :].contiguous(), group=dist.group.WORLD, async_op=True)

        if self.has_fc1_bias:
            self.dispatch_params.batched_fc1_bias_cache_pool.zero_()
            self.dispatch_params.batched_fc1_bias_cache = \
                [self.dispatch_params.batched_fc1_bias_cache_pool[:, self.ids_list_global[i][depth_idx]].contiguous() for i in range(self.world_size)]
            self.dispatch_params.fc1_b_op = dist.all_gather(self.dispatch_params.batched_fc1_bias_cache, \
                self.batched_fc1_bias[depth_idx].data[:, :self.local_dim_list[depth_idx]].contiguous(), group=dist.group.WORLD, async_op=True)

    def load_ckpt(self, load_moe, src_rank = 0):
        # load_moe is an instance of MoE_Buffer class, which is always kept on CPU
        for depth_idx in range(self.total_depth_moe):
            # fetch the params on each device
            fc1_w_tmp = torch.zeros(self.num_global_experts, self.local_dim_list[depth_idx], self.input_dim_list_moe[depth_idx]).cuda()
            fc2_w_tmp = torch.zeros(self.num_global_experts, self.local_dim_list[depth_idx], self.output_dim_list_moe[depth_idx]).cuda()
            fc1_bias_tmp = torch.zeros(self.num_global_experts, self.local_dim_list[depth_idx]).cuda() if self.has_fc1_bias else None
            fc2_bias_tmp = torch.zeros(self.num_global_experts, self.output_dim_list_moe[depth_idx]).cuda() if self.has_fc2_bias else None

            # prepare a param list to scatter
            if self.local_rank == src_rank:
                # split the parameters
                scatter_list_fc1_w = [torch.zeros(self.num_global_experts, \
                    self.local_dim_list[depth_idx], self.input_dim_list_moe[depth_idx]) for _ in range(self.world_size)]
                scatter_list_fc2_w = [torch.zeros(self.num_global_experts, \
                    self.local_dim_list[depth_idx], self.output_dim_list_moe[depth_idx]) for _ in range(self.world_size)]
                if self.has_fc1_bias:
                    scatter_list_fc1_b = [torch.zeros(self.num_global_experts, \
                        self.local_dim_list[depth_idx]) for _ in range(self.world_size)]
                else:
                    scatter_list_fc1_b = [None for _ in range(self.world_size)]
                if self.has_fc2_bias:
                    scatter_list_fc2_b = [torch.zeros(self.num_global_experts, \
                        self.output_dim_list_moe[depth_idx]) for _ in range(self.world_size)]
                else:
                    scatter_list_fc2_b = [None for _ in range(self.world_size)]

                for i in range(self.world_size):
                    if self.has_fc2_bias:
                        scatter_list_fc2_b[i] = load_moe.batched_fc2_bias[depth_idx].data
                    scatter_list_fc1_w[i][:, :len(self.ids_list_global[i][depth_idx]), :] = \
                        load_moe.batched_fc1_w[depth_idx].data[:, self.ids_list_global[i][depth_idx], :]
                    scatter_list_fc2_w[i][:, :len(self.ids_list_global[i][depth_idx]), :] = \
                        load_moe.batched_fc2_w[depth_idx].data[:, self.ids_list_global[i][depth_idx], :]
                    if self.has_fc1_bias:
                        scatter_list_fc1_b[i][:, :len(self.ids_list_global[i][depth_idx])] = \
                            load_moe.batched_fc1_bias[depth_idx].data[:, self.ids_list_global[i][depth_idx]]
            
                # load the params to src_rank
                scatter_list_fc1_w = [item.cuda() for item in scatter_list_fc1_w]
                scatter_list_fc2_w = [item.cuda() for item in scatter_list_fc2_w]
                if self.has_fc1_bias:
                    scatter_list_fc1_b = [item.cuda() for item in scatter_list_fc1_b]
                if self.has_fc2_bias:
                    scatter_list_fc2_b = [item.cuda() for item in scatter_list_fc2_b]
            else:
                scatter_list_fc1_w, scatter_list_fc2_w, scatter_list_fc1_b, scatter_list_fc2_b = None, None, None, None

            dist.scatter(fc1_w_tmp, scatter_list_fc1_w, src=src_rank)
            dist.scatter(fc2_w_tmp, scatter_list_fc2_w, src=src_rank)
            dist.scatter(fc1_bias_tmp, scatter_list_fc1_b, src=src_rank)
            dist.scatter(fc2_bias_tmp, scatter_list_fc2_b, src=src_rank)

            # load the params
            self.batched_fc1_w[depth_idx].data[:, :self.local_dim_list[depth_idx], :] = fc1_w_tmp
            self.batched_fc2_w[depth_idx].data[:, :self.local_dim_list[depth_idx], :] = fc2_w_tmp
            if self.has_fc1_bias:
                self.batched_fc1_bias[depth_idx].data[:, :self.local_dim_list[depth_idx]] = fc1_bias_tmp
            else:
                self.batched_fc1_bias[depth_idx].data = fc1_bias_tmp
            self.batched_fc2_bias[depth_idx].data = fc2_bias_tmp

    def forward(self, depth_idx: int, input: Tensor, gate_index=0, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1):       
        # Refactor depth_idx into moe_depth_idx or mlp_depth_idx.
        moe_flag = depth_idx in self.moe_idx_list
        depth_idx_mlp, depth_idx_moe = -1, -1
        if moe_flag:
            depth_idx_moe = self.moe_idx_list.index(depth_idx)
        else:
            depth_idx_mlp = self.mlp_idx_list.index(depth_idx)
        
        if (depth_idx_moe == 0) and (self.dispatch_params.data_centric):
            self.cache_gather(depth_idx_moe)
        
        if self.skip_moe:
            result_output = input
            return self.result_func(result_output) if self.result_func is not None else result_output, None

        original_shape, original_dtype  = input.shape, input.dtype
        assert len(original_shape) >= 2, "Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim"

        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        if torch.is_autocast_enabled():
            x = cast_if_autocast_enabled(x)
        else:
            for p in self.parameters():
                x = x.to(p.dtype)
                break
        
        if not moe_flag:
            y = MLP_Implement.apply(self.dispatch_params, x, \
                    self.mlp_fc1_w[depth_idx_mlp], self.mlp_fc2_w[depth_idx_mlp], \
                    self.mlp_fc1_b[depth_idx_mlp], self.mlp_fc2_b[depth_idx_mlp]).to(original_dtype)
            y = y.view(list(original_shape[:-reserve_dims]) + list(y.shape[-reserve_dims:])).to(original_dtype)
            return self.result_func(y) if self.result_func is not None else y, 0
        
        gctx = self.gates[depth_idx_moe * self.gate_len + gate_index]
        if a2a_ffn_overlap_degree is not None:
            self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        a2a_ffn_overlap_degree = self.a2a_ffn_overlap_degree

        top_k = top_k or gctx.top_k

        def routing():
            logits = gctx(x)

            if self.training and gctx.gate_noise > 0:
                logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
            else:
                logits_w_noise = logits

            scores = F.softmax(logits_w_noise, dim=1)
            if self.is_gshard_loss:
                _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)
            else:
                _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(
                    F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1),
                    self.num_global_experts, gctx.gate_noise)

            return logits.dtype, simplified_extract_critical(scores, top_k = top_k, loss_fn = _loss_fn)

        if x.is_cuda:
            logits_dtype, (indices_, l_aux) = routing()
            with torch.cuda.amp.autocast(enabled=True):
                logits_dtype, (indices_, l_aux) = routing()
        else:
            logits_dtype, (indices_, l_aux) = routing()

        # Expert computation using combined fast_encode and fast_decode
        data = x.to(logits_dtype)
        assert data.is_contiguous(), "Input tensor for encode/decode should be in contiguous memory format."
        assert depth_idx_moe >= 0 and depth_idx_moe < self.total_depth_moe

        self.dispatch_params.indices_ = [x.to(torch.int32).view(-1) for x in indices_]
        self.dispatch_params.stage = depth_idx_moe

        if depth_idx_moe >= 1:
            # prepare the previous local params
            self.dispatch_params.prev_fc1_w = self.batched_fc1_w[depth_idx_moe-1].data[:, :self.local_dim_list[depth_idx_moe-1], :].contiguous()
            self.dispatch_params.prev_fc2_w = self.batched_fc2_w[depth_idx_moe-1].data[:, :self.local_dim_list[depth_idx_moe-1], :].contiguous()
        
        param_1_w = self.batched_fc1_w[depth_idx_moe].data[:, :self.local_dim_list[depth_idx_moe], :].contiguous()
        param_2_w = self.batched_fc2_w[depth_idx_moe].data[:, :self.local_dim_list[depth_idx_moe], :].contiguous()
        if self.has_fc1_bias:
            param_1_b = self.batched_fc1_bias[depth_idx_moe].data[:, :self.local_dim_list[depth_idx_moe]].contiguous()
        else:
            param_1_b = self.batched_fc1_bias[depth_idx_moe]
        param_2_b = self.batched_fc2_bias[depth_idx_moe]

        y = FusedMoE_Shared_CudaCore.apply(self.dispatch_params, data, \
                param_1_w, param_2_w, param_1_b, param_2_b).to(original_dtype)

        if (depth_idx_moe < self.total_depth_moe - 1) and (self.dispatch_params.data_centric):
            self.cache_gather(depth_idx_moe + 1)
        y = y.view(list(original_shape[:-reserve_dims]) + list(y.shape[-reserve_dims:])).to(original_dtype)

        return self.result_func(y) if self.result_func is not None else y, l_aux
