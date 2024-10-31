#include <cuda_fp16.h>

void launch_count(
    int *tokens_per_expert,
    int *expanded_tokens_per_expert,
    int *routings,
    int num_tokens,
    int num_routings,
    int device_idx
);

void launch_assign(
    int *token_idx_list,
    int *expert_idx_list,
    const int *expanded_tokens_per_expert,
    int *routings,
    int num_tokens,
    int num_routings,
    int device_idx
);

void launch_esmm_accum_shared_tensor_full(
    float *result,
    float *tokens,
    float *weights,
    float *bias,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_esmm_accum_shared_tensor_mix(
    float *result,
    __half *tokens,
    float *weights,
    float *bias,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_esmm_accum_shared_tensor_half(
    float *result,
    __half *tokens,
    __half *weights,
    __half *bias,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_fused_grad_bias_full(
    float *j_b,
    float *j_x,
    float *j_w,
    float *j_y,
    float *w,
    float *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_fused_grad_no_bias_full(
    float *j_x,
    float *j_w,
    float *j_y,
    float *w,
    float *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_fused_grad_bias_mix(
    float *j_b,
    float *j_x,
    float *j_w,
    __half *j_y,
    float *w,
    __half *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_fused_grad_no_bias_mix(
    float *j_x,
    float *j_w,
    __half *j_y,
    float *w,
    __half *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_fused_grad_bias_half(
    float *j_b,
    float *j_x,
    float *j_w,
    __half *j_y,
    __half *w,
    __half *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);

void launch_fused_grad_no_bias_half(
    float *j_x,
    float *j_w,
    __half *j_y,
    __half *w,
    __half *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens,
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx
);