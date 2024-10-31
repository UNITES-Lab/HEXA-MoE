## HEXA-MoE implementation with CUDA

Our code is tested with Python 3.7, CUDA 11.7 and PyTorch <= 1.13.1

```
    conda create -n hexa_moe_37 python=3.7 -y
    conda activate hexa_moe_37
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
    pip install easydict
```

Install CUDA implemented HEXA-MoE

```
    cd hexa_moe_cuda
    python setup.py install
```