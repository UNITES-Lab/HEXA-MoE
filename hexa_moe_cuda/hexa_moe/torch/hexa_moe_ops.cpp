#include <torch/extension.h>
#include <stdio.h>
#include "tmoe.h"
#include <iostream>
#include <cuda_fp16.h>

using namespace std;

void TOKEN_COUNT(
        torch::Tensor &tokens_per_expert,
        torch::Tensor &expanded_tokens_per_expert,
        torch::Tensor &routings,
        int64_t num_tokens,
        int64_t num_routings) {
    int device_idx=0;
    launch_count(
        (int *)tokens_per_expert.data_ptr(),
        (int *)expanded_tokens_per_expert.data_ptr(),
        (int *)routings.data_ptr(),
        num_tokens,
        num_routings,
        device_idx
    );
}

void TOKEN_ASSIGN(
        torch::Tensor &token_idx_list,
        torch::Tensor &expert_idx_list,
        const torch::Tensor &expanded_tokens_per_expert,
        torch::Tensor &routings,
        int64_t num_tokens,
        int64_t num_routings) {
    int device_idx=0;
    launch_assign(
        (int *)token_idx_list.data_ptr(),
        (int *)expert_idx_list.data_ptr(),
        (const int *)expanded_tokens_per_expert.data_ptr(),
        (int *)routings.data_ptr(),
        num_tokens,
        num_routings,
        device_idx
    );
}

void ESMM_TENSOR_CORE(
        torch::Tensor &result,
        torch::Tensor &tokens,
        torch::Tensor &weights,
        torch::Tensor &bias,
        const torch::Tensor &token_idx_list,
        const torch::Tensor &expert_idx_list,
        const torch::Tensor &num_expanded_tokens,
        bool x_dtype_full,
        bool w_dtype_full,
        bool b_dtype_full,
        bool m2s,
        int64_t num_tokens,
        int64_t top_k,
        int64_t num_in_dims,
        int64_t num_out_dims,
        int64_t num_routings) {
    int device_idx=0;
    if ((x_dtype_full) && (w_dtype_full) && (b_dtype_full)) {
        launch_esmm_accum_shared_tensor_full(
            (float *)result.data_ptr(),
            (float *)tokens.data_ptr(),
            (float *)weights.data_ptr(),
            (float *)bias.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else if ((!x_dtype_full) && (w_dtype_full) && (b_dtype_full)) {
        launch_esmm_accum_shared_tensor_mix(
            (float *)result.data_ptr(),
            (__half *)tokens.data_ptr(),
            (float *)weights.data_ptr(),
            (float *)bias.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else if ((!x_dtype_full) && (!w_dtype_full) && (!b_dtype_full)) {
        launch_esmm_accum_shared_tensor_half(
            (float *)result.data_ptr(),
            (__half *)tokens.data_ptr(),
            (__half *)weights.data_ptr(),
            (__half *)bias.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else {
        cout << "Unsupported data type combination!" << endl;
    }
}

void FUSED_GRAD(
        torch::Tensor &j_b,
        torch::Tensor &j_x,
        torch::Tensor &j_w,
        torch::Tensor &j_y,
        torch::Tensor &w,
        torch::Tensor &x,
        const torch::Tensor &token_idx_list,
        const torch::Tensor &expert_idx_list,
        const torch::Tensor &num_expanded_tokens,
        bool x_dtype_full,
        bool w_dtype_full,
        bool m2s,
        int64_t num_tokens,
        int64_t top_k,
        int64_t num_in_dims,
        int64_t num_out_dims,
        int64_t num_routings) {
    int device_idx=0;
    if ((x_dtype_full) && (w_dtype_full)) {
        launch_fused_grad_bias_full(
            (float *)j_b.data_ptr(),
            (float *)j_x.data_ptr(),
            (float *)j_w.data_ptr(),
            (float *)j_y.data_ptr(),
            (float *)w.data_ptr(),
            (float *)x.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else if ((!x_dtype_full) && (w_dtype_full)) {
        launch_fused_grad_bias_mix(
            (float *)j_b.data_ptr(),
            (float *)j_x.data_ptr(),
            (float *)j_w.data_ptr(),
            (__half *)j_y.data_ptr(),
            (float *)w.data_ptr(),
            (__half *)x.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else if ((!x_dtype_full) && (!w_dtype_full)) {
        launch_fused_grad_bias_half(
            (float *)j_b.data_ptr(),
            (float *)j_x.data_ptr(),
            (float *)j_w.data_ptr(),
            (__half *)j_y.data_ptr(),
            (__half *)w.data_ptr(),
            (__half *)x.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else {
        cout << "Unsupported data type combination!" << endl;
    }
}

void FUSED_GRAD_NO_BIAS(
        torch::Tensor &j_x,
        torch::Tensor &j_w,
        torch::Tensor &j_y,
        torch::Tensor &w,
        torch::Tensor &x,
        const torch::Tensor &token_idx_list,
        const torch::Tensor &expert_idx_list,
        const torch::Tensor &num_expanded_tokens,
        bool x_dtype_full,
        bool w_dtype_full,
        bool m2s,
        int64_t num_tokens,
        int64_t top_k,
        int64_t num_in_dims,
        int64_t num_out_dims,
        int64_t num_routings) {
    int device_idx=0;
    if ((x_dtype_full) && (w_dtype_full)) {
        launch_fused_grad_no_bias_full(
            (float *)j_x.data_ptr(),
            (float *)j_w.data_ptr(),
            (float *)j_y.data_ptr(),
            (float *)w.data_ptr(),
            (float *)x.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else if ((!x_dtype_full) && (w_dtype_full)) {
        launch_fused_grad_no_bias_mix(
            (float *)j_x.data_ptr(),
            (float *)j_w.data_ptr(),
            (__half *)j_y.data_ptr(),
            (float *)w.data_ptr(),
            (__half *)x.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else if ((!x_dtype_full) && (!w_dtype_full)) {
        launch_fused_grad_no_bias_half(
            (float *)j_x.data_ptr(),
            (float *)j_w.data_ptr(),
            (__half *)j_y.data_ptr(),
            (__half *)w.data_ptr(),
            (__half *)x.data_ptr(),
            (const int *)token_idx_list.data_ptr(),
            (const int *)expert_idx_list.data_ptr(),
            (const int *)num_expanded_tokens.data_ptr(),
            m2s,
            num_tokens,
            top_k,
            num_in_dims,
            num_out_dims,
            num_routings,
            device_idx
        );
    } else {
        cout << "Unsupported data type combination!" << endl;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("TOKEN_COUNT",
          &TOKEN_COUNT,
          "TOKEN_COUNT kernel warpper");
    m.def("TOKEN_ASSIGN",
          &TOKEN_ASSIGN,
          "TOKEN_ASSIGN kernel warpper");
    m.def("ESMM_TENSOR_CORE",
          &ESMM_TENSOR_CORE,
          "ESMM_TENSOR_CORE kernel warpper");
    m.def("FUSED_GRAD",
          &FUSED_GRAD,
          "FUSED_GRAD kernel warpper");
    m.def("FUSED_GRAD_NO_BIAS",
          &FUSED_GRAD_NO_BIAS,
          "FUSED_GRAD_NO_BIAS kernel warpper");
}

TORCH_LIBRARY(tmoe_cuda, m) {
    m.def("TOKEN_COUNT", TOKEN_COUNT);
    m.def("TOKEN_ASSIGN", TOKEN_ASSIGN);
    m.def("ESMM_TENSOR_CORE", ESMM_TENSOR_CORE);
    m.def("FUSED_GRAD", FUSED_GRAD);
    m.def("FUSED_GRAD_NO_BIAS", FUSED_GRAD_NO_BIAS);
}