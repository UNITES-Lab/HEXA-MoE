# HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with zero computation redundancy

Code for the paper "HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with zero computation redundancy"

- Authors: Shuqing Luo, Jie Peng, Pingzhi Li, Hanrui Wang and Tianlong Chen

## Overview



## Code & Usage

We provide HEXA-MoE implementations with both [Triton](./hexa_moe_triton/) and [CUDA](./hexa_moe_cuda/). The programming interfaces for both are the same:

```
    import torch.nn.functional as F
    from hexa_moe import moe as hmoe

    # In the class for model definition
    _gate_type = {'type': 'top', 'k': 1, 'gate_noise': 1.0, 'fp32_gate': True}
    self.cascaded_moe = t_moe.MoE_Cascaded(
        gate_type=_gate_type,
        model_dim_list=[128,128,128,128],
        moe_idx_list=[2,3],
        mlp_ratio=4,
        mlp_proportion=None, # Or a list with length world_size and sum 1
        num_global_experts=8,
        total_depth=4,
        data_centric=True,
        mlp_fc1_bias=True,
        mlp_fc2_bias=True,
        activation_fn=lambda x: F.gelu(x)
    )

    # In forward(self, xxx) function in the class
    for depth_idx in range(self.total_depth):
        x, cur_l_aux = self.cascaded_moe(depth_idx, x)
```

We define all the FFNs/MoEs of a model in a single class to facilitate the pipeline-shared cache.