## HEXA-MoE implementation with Triton

Our code is tested with Python 3.8, CUDA 12.1 and PyTorch >= 2.0.0

```
    conda create -n hexa_moe_38 python=3.8 -y
    conda activate hexa_moe_38
    conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    pip install triton
    pip install constant
    pip install easydict
```

Install triton implemented HEXA-MoE

```
    cd hexa_moe_triton
    python setup.py install
```