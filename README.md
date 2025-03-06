# HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

Code for the paper "HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with zero computation redundancy"

[![Arxiv](https://img.shields.io/badge/arXiv-2411.01288-B31B1B.svg)][#arxiv-paper-package]

[#arxiv-paper-package]: https://arxiv.org/abs/2411.01288

- Authors: [Shuqing Luo](https://luoshuqing2001.github.io/), [Jie Peng](https://openreview.net/profile?id=~Jie_Peng4), [Pingzhi Li](https://pingzhili.github.io/), [Hanrui Wang](https://www.hanruiwang.com/) and [Tianlong Chen](https://tianlong-chen.github.io/)

## Overview

In this paper, we propose to reformulate MoE computing with expert-specific operators as an alternative to general matrix multiplication (GeMM) or grouped GeMM. Current MoE librares are all built upon GeMM, where dispatch and combine operations are required to calibrate the workload of each expert. However, this requires token padding or discarding, which hampers the computation efficiency as well as model performance. Moreover, current MoE libraries mainly adopt expert parallelism to distribute MoE layer parameters to different devices due to its sheer size, which depends on homogeneous devices. However, if we wanna adapt MoE computing to heterogeneous devices, expert parallelism turn to be complicated and impractical. This is because we need to re-arrange computation allocation on different devices to utilize total computing capacity better, but this is difficult for expert parallelism due to its inherent dynamism. To achieve both heterogeneous-awareness and high computation efficiency for MoE model, we propose to re-factor MoE computing with expert-specific operators, and adapt it to data- and model- centric configurations considering different workload scale. We present a comparison between conventional GeMM-based and our expert-specific operators-based MoE computing for both forward and backward propagation. Experiments with Swin-MoE on homogeneous devices show that our method can reduce 10%-48% memory consumption while achieve 0.5-4.3× speed up compared to Tutel and MegaBlocks. Experiments on heterogeneous devices demonstrate that our method can substaintially minimize the average latency. 

<div align="middle">
    <img src=./figures/formula.jpg width=96% />
</div>

## Code & Usage

We provide HEXA-MoE implementations with both [Triton](./hexa_moe_triton/) and [CUDA](./hexa_moe_cuda/). The programming interfaces for both are the same:

```
    import torch.nn.functional as F
    from hexa_moe import moe as hmoe

    # In the class for model definition
    _gate_type = {'type': 'top', 'k': 1, 'gate_noise': 1.0, 'fp32_gate': True}
    self.cascaded_moe = hmoe.MoE_Cascaded(
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
