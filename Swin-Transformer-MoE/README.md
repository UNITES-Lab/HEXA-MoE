## Swin-Transformer-MoE with T-MoE

### ImageNet-1k preparation

We take ImageNet-1k as the training dataset, organizing the folder as

```
    ImageNet
    │
    └───imagenet-1k
    │   │   train_map.txt
    │   │   val_map.txt
    │   │
    │   └───train
    │   │    │   ...
    │   │    
    │   └───val    
    │        │   ...
    │
    └───imagenet-22k
```

where [train_map.txt](https://pan.baidu.com/s/1O_hZXnG1ytsIUo65ZbBGQQ?pwd=1234), [val_map.txt](https://pan.baidu.com/s/1F05A2_9LqwWwjisM81LkFg?pwd=1234) and [val](https://pan.baidu.com/s/1Pht5eaI9z80fncsZEAYI-A?pwd=1234) can be downloaded from BaiduNetDisk (1234), while folder train has to be prepared following below:

```
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
    cd ImageNet/imagenet-1k
    mkdir train && tar -xvf ILSVRC2012_img_train.tar -C train && for x in `ls train/*tar`; do fn=train/`basename $x .tar`; mkdir $fn; tar -xvf $x -C $fn; rm -f $fn.tar; done
```

### Training from scratch

After installing the Swin-Transformer dependences and T-MoE library with either [CUDA](https://github.com/luoshuqing2001/tmoe/tree/main/tmoe_cuda) or [Triton](https://github.com/luoshuqing2001/tmoe/tree/main/tmoe_triton) implementation, you can start training on 4 GPUs via

```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --nnode=1  main_moe_cascaded.py --cfg configs/swinmoe/swin_moe_small_patch4_window12_192_4gpu_1k.yaml --data-path ImageNet/imagenet-1k/ --batch-size 100
```