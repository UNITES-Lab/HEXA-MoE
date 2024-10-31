from .swin_transformer_moe_cascaded import SwinTransformerMoE_Cascaded

from hexa_moe import moe as hmoe

def build_model(config):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin_moe_cascaded':
        model = SwinTransformerMoE_Cascaded(
                        img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
                        depths=config.MODEL.SWIN_MOE.DEPTHS,
                        num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
                        num_global_experts=config.MODEL.SWIN_MOE.NUM_GLOBAL_EXPERTS,
                        window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MOE.APE,
                        patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
                        mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
                        init_std=config.MODEL.SWIN_MOE.INIT_STD,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                        moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
                        num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
                        top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
                        capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
                        cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
                        normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
                        use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
                        is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
                        gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
                        cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
                        cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
                        moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
                        aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT)

        model_dim_list = []
        for i in range(len(config.MODEL.SWIN_MOE.DEPTHS)):
            for _ in range(config.MODEL.SWIN_MOE.DEPTHS[i]):
                model_dim_list.append(config.MODEL.SWIN_MOE.EMBED_DIM * 2 ** i)

        # Re-Index the moe_blocks list
        counter_base = 0
        moe_idx_list = []
        for i in range(len(config.MODEL.SWIN_MOE.DEPTHS)):
            for item in config.MODEL.SWIN_MOE.MOE_BLOCKS[i]:
                if item >= 0:
                    moe_idx_list.append(counter_base + item)
            counter_base += config.MODEL.SWIN_MOE.DEPTHS[i]

        model_dim_list_moe = [model_dim_list[i] for i in moe_idx_list]

        moe_cascaded = hmoe.MoE_Buffer(
                        model_dim_list_moe=model_dim_list_moe,
                        num_global_experts=config.MODEL.SWIN_MOE.NUM_GLOBAL_EXPERTS,
                        mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                        total_depth_moe=len(moe_idx_list),
                        mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    if model_type == 'swin_moe_cascaded':
        return model, moe_cascaded
    else:
        return model
