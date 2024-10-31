#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define TILE 4 // Assuming TILE == UNROLL

const int threads_per_chunk = BLOCK_SIZE / (WARP_SIZE / BLOCK_SIZE); // 8

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct __align__(8) Half4
{
    __half x;
    __half y;
    __half z;
    __half w;
};

__forceinline__ __host__ __device__ Half4 make_Half4(const float x, const float y, const float z, float w)
{
    Half4 h4;

    h4.x = __float2half(x);
    h4.y = __float2half(y);
    h4.z = __float2half(z);
    h4.w = __float2half(w);

    return h4;
}


__global__ void fused_grad_bias_kernel_vectorized_full(
    float *j_b,
    float *j_x,
    float *j_w,
    float *j_y,
    float *w,
    float *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *expert_start_list,
    bool m2s,
    int delta_1,
    int num_tokens,
    int num_in_dims,
    int num_out_dims,
    int num_routings
) {
    float zero_pad_float4[4] = {0.0f};
    __half zero_pad_half4[4] = {__float2half(0.0f)};

    int ThreadTok = threadIdx.x % BLOCK_SIZE;
    int ThreadOut = threadIdx.x / BLOCK_SIZE;
    int ThreadTile = threadIdx.y;

    int input_offset = 0, output_offset = 0, k_idx = blockIdx.x / num_routings, \
        loc_start=expert_start_list[k_idx * (num_routings+1) + blockIdx.x % num_routings], \
        loc_end=expert_start_list[k_idx * (num_routings+1) + blockIdx.x % num_routings + 1];
    int num_expanded_tokens = loc_end - loc_start;
    if (m2s) {
        output_offset = k_idx*num_tokens*num_out_dims;
    } else {
        input_offset = k_idx*num_tokens*num_in_dims;
    }

    __shared__ __half smem_tok[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ __half smem_op[BLOCK_SIZE * (TILE * BLOCK_SIZE)];
    __shared__ float smem_result[BLOCK_SIZE * (TILE * BLOCK_SIZE)];

    int task_id = -1; // 0 for ESMM, 1 for ESTMM, 2 for ESS.
    int delta_2 = num_out_dims % BLOCK_SIZE == 0? int(num_out_dims / BLOCK_SIZE): 1+int(num_out_dims / BLOCK_SIZE);
    if (blockIdx.z < delta_1)
        task_id = 0;
    else if (blockIdx.z < delta_1 + delta_2)
        task_id = 1;
    else
        task_id = 2;

    int tok_idx_0 = -1, exp_idx_0 = -1;
    if (blockIdx.z < delta_1) {
        if (blockIdx.z * BLOCK_SIZE + ThreadTok < num_expanded_tokens) {
            tok_idx_0 = token_idx_list[loc_start + blockIdx.z * BLOCK_SIZE + ThreadTok];
            exp_idx_0 = expert_idx_list[loc_start + blockIdx.z * BLOCK_SIZE];
        }
    }

    int exp_idx_1 = blockIdx.x % num_routings, loc_exp_start = -1, loc_exp_end = -1;
    loc_exp_start = expert_start_list[k_idx * (num_routings+1) + exp_idx_1]; // less than num_expanded_tokens
    loc_exp_end = expert_start_list[k_idx * (num_routings+1) + exp_idx_1 + 1];

    if ((task_id == 0) && (blockIdx.x % num_routings == 0)) {
        // Define the constants for ESMM
        int TileIn = blockIdx.y;

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
        for (int i = 0; i < num_out_dims; i += BLOCK_SIZE) {
            // load smem_tok with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx_0 < 0) || (tok_idx_0 >= num_tokens)) {
                    reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                } else {
                    int out_dims_base = i + ThreadOut * threads_per_chunk + j * 4;
                    if (out_dims_base + 3 < num_out_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base])[0];
                        Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base + 2 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base + 1];
                        float tmp_float_z = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base + 2];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base + 1 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base + 1];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else {
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                    }
                }
            }

            // load smem_op with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((exp_idx_0 < 0) || (exp_idx_0 >= num_routings)) {
                    reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                } else {
                    int out_dims_base = i + ThreadTok;
                    if (out_dims_base >= num_out_dims) {
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                    } else {
                        int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                        if (in_dims_base + 3 < num_in_dims) {
                            float4 tmp_float4 = reinterpret_cast<float4 *>(&w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base])[0];
                            Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else if (in_dims_base + 2 < num_in_dims) {
                            float tmp_float_x = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base];
                            float tmp_float_y = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base + 1];
                            float tmp_float_z = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base + 2];
                            Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else if (in_dims_base + 1 < num_in_dims) {
                            float tmp_float_x = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base];
                            float tmp_float_y = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base + 1];
                            Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else if (in_dims_base < num_in_dims) {
                            float tmp_float_x = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base];
                            Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else {
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                                reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                        }
                    }
                }
            }

            wmma::load_matrix_sync(a_frag, smem_tok, BLOCK_SIZE);
            wmma::load_matrix_sync(b_frag, smem_op + ThreadTile * BLOCK_SIZE, TILE * BLOCK_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(smem_result + ThreadTile * BLOCK_SIZE, c_frag, TILE * BLOCK_SIZE, wmma::mem_row_major);

        // write back to global memory with vectorized memory access
        if (m2s) {
            if ((tok_idx_0 >= 0) && (tok_idx_0 < num_tokens)) {
                int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk;
                for (int i = 0; i < threads_per_chunk; i++) {
                    if (in_dims_base + i < num_in_dims) {
                        atomicAdd(j_x + input_offset + tok_idx_0 * num_in_dims + in_dims_base + i, \
                            smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i]);
                        // at::native::fastAtomicAdd(j_x, input_offset + tok_idx_0 * num_in_dims + in_dims_base + i, num_tokens * num_in_dims, \
                        //     smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i], true);
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx_0 >= 0) && (tok_idx_0 < num_tokens)) {
                    int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (in_dims_base + 3 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        float4 tmp_float4_ori = reinterpret_cast<float4 *>(&j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base])[0];
                        float4 tmp_float4_dst = make_float4(tmp_float4_ori.x + tmp_float4.x, tmp_float4_ori.y + tmp_float4.y, \
                                                            tmp_float4_ori.z + tmp_float4.z, tmp_float4_ori.w + tmp_float4.w);
                        reinterpret_cast<float4 *>(&j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base])[0] = tmp_float4_dst;
                    } else if (in_dims_base + 2 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base] += tmp_float4.x;
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base + 1] += tmp_float4.y;
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base + 2] += tmp_float4.z;
                    } else if (in_dims_base + 1 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base] += tmp_float4.x;
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base + 1] += tmp_float4.y;
                    } else if (in_dims_base < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base] += tmp_float4.x;
                    }
                }
            }
        }
    } else if (task_id == 1) {
        // Define the constants for ESTMM
        int BlockOut = blockIdx.z;
        int TileIn = blockIdx.y;
        int exp_idx = expert_idx_list[loc_exp_start];

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
        for (int i = loc_exp_start; i < loc_exp_end; i += BLOCK_SIZE) {
            int tok_idx = token_idx_list[i + ThreadTok];

            // load smem_tok with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx < 0) || (tok_idx >= num_tokens)) {
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                } else {
                    int out_dims_base = BlockOut * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (out_dims_base + 3 < num_out_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&j_y[output_offset + tok_idx * num_out_dims + out_dims_base])[0];
                        Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else if (out_dims_base + 2 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 1];
                        float tmp_float_z = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 2];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else if (out_dims_base + 1 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 1];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else if (out_dims_base < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else {
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    }
                }
            }

            // load smem_op with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx < 0) || (tok_idx >= num_tokens)) {
                    reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(&zero_pad_half4)[0];
                } else {
                    int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (in_dims_base + 3 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&x[input_offset + tok_idx * num_in_dims + in_dims_base])[0];
                        Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else if (in_dims_base + 2 < num_in_dims) {
                        float tmp_float_x = x[input_offset + tok_idx * num_in_dims + in_dims_base];
                        float tmp_float_y = x[input_offset + tok_idx * num_in_dims + in_dims_base + 1];
                        float tmp_float_z = x[input_offset + tok_idx * num_in_dims + in_dims_base + 2];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else if (in_dims_base + 1 < num_in_dims) {
                        float tmp_float_x = x[input_offset + tok_idx * num_in_dims + in_dims_base];
                        float tmp_float_y = x[input_offset + tok_idx * num_in_dims + in_dims_base + 1];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else if (in_dims_base < num_in_dims) {
                        float tmp_float_x = x[input_offset + tok_idx * num_in_dims + in_dims_base];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else {
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<Half4 *>(&zero_pad_half4)[0];
                    }
                }
            }

            wmma::load_matrix_sync(a_frag, smem_tok, BLOCK_SIZE);
            wmma::load_matrix_sync(b_frag, smem_op + ThreadTile * BLOCK_SIZE, TILE * BLOCK_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(smem_result + ThreadTile * BLOCK_SIZE, c_frag, TILE * BLOCK_SIZE, wmma::mem_row_major);

        // write back to global memory with vectorized memory access
        // gradients for W are accumulative in all cases
        int out_dims_base = BlockOut * BLOCK_SIZE + ThreadTok;
        if (out_dims_base < num_out_dims) {
            int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk;
            for (int i = 0; i < threads_per_chunk; i++) {
                if (in_dims_base + i < num_in_dims) {
                    atomicAdd(j_w + exp_idx * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base, \
                        smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i]);
                    // at::native::fastAtomicAdd(j_w, exp_idx * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base, \
                    //     num_routings * num_in_dims * num_out_dims, \
                    //     smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i], true);
                }
            }
        }
    } else if ((task_id == 2) && (blockIdx.y == 0)) {
        int TileOut = blockIdx.z;
        float acc = 0.0f;
        int exp_idx = expert_idx_list[loc_exp_start];

#pragma unroll
        for (int i = loc_exp_start; i < loc_exp_end; i += BLOCK_SIZE) {
            int tok_idx = token_idx_list[i + ThreadTok];

            // load tokens to smem_result with vectorized memory access
            for (int j = 0; j < 2; j++) {
                if ((tok_idx < 0) || (tok_idx >= num_tokens)) {
                    reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<float4 *>(&zero_pad_float4)[0];
                } else {
                    int out_dims_base = TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (out_dims_base + 3 < num_out_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&j_y[output_offset + tok_idx * num_out_dims + out_dims_base])[0];
                        reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            tmp_float4;
                    } else if (out_dims_base + 2 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 1];
                        float tmp_float_z = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 2];
                        float4 tmp_float4 = make_float4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            tmp_float4;
                    } else if (out_dims_base + 1 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 1];
                        float4 tmp_float4 = make_float4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            tmp_float4;
                    } else if (out_dims_base < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float4 tmp_float4 = make_float4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            tmp_float4;
                    } else {
                        reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<float4 *>(&zero_pad_float4)[0];
                    }
                }
            }
            __syncthreads();

            if (ThreadOut == 0) {
                if (TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadTok < num_out_dims) {
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        acc += smem_result[ThreadTile * BLOCK_SIZE + ThreadTok];
                    }
                }
            }
        }

        if (ThreadOut == 0) {
            if (TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadTok < num_out_dims) {
                j_b[exp_idx * num_out_dims + TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadTok] += acc;
            }
        }
    }
}

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
    int device_idx=0
) {
    /*
        We build the CUDA kernel with a unified manner 
        for memory access and computation.

        Inputs:
            j_y [num_tokens, num_out_dims]
            w [num_experts, num_out_dims, num_in_dims]
            x [num_tokens, num_in_dims]
        
        Outputs:
            j_b [num_experts, num_outputs]
            j_x [num_tokens, num_in_dims]
            j_w [num_experts, num_out_dims, num_in_dims]

        We re-organiza the blocks to fuse the 3 operators in parallel
        Assuming that TILE == UNROLL
            For ESMM:  (num_expanded_tokens // BLOCK_SIZE, num_in_dims // (TILE * BLOCK_SIZE), 1)
            For ESTMM: (num_routings,                      num_in_dims // (TILE * BLOCK_SIZE), num_out_dims // BLOCK_SIZE)
            For ESS:   (num_routings,                      1,                                  num_out_dims // (TILE * BLOCK_SIZE))
            
            Padding and flipping:
            For ESMM:  (num_routings, num_in_dims // (TILE * BLOCK_SIZE), num_expanded_tokens // BLOCK_SIZE)
            For ESTMM: (num_routings, num_in_dims // (TILE * BLOCK_SIZE), num_out_dims // BLOCK_SIZE)
            For ESS:   (num_routings, num_in_dims // (TILE * BLOCK_SIZE), num_out_dims // (TILE * BLOCK_SIZE))
    */
    int n_expd_toks = 0, loc_start = 0, loc_end = 0;

    int *num_expanded_tokens_cpu = new int [top_k + 1]();
    cudaMemcpy(num_expanded_tokens_cpu, num_expanded_tokens, sizeof(int) * (top_k + 1), cudaMemcpyDeviceToHost);

    int *exp_idx_list = new int [num_expanded_tokens_cpu[top_k]]();
    cudaMemcpy(exp_idx_list, expert_idx_list, sizeof(int) * num_expanded_tokens_cpu[top_k], cudaMemcpyDeviceToHost);

    int *exp_start_list = new int [top_k * (num_routings + 1)]();
    for (int j = 0; j < top_k; j++) {
        loc_end = num_expanded_tokens_cpu[j + 1];
        loc_start = num_expanded_tokens_cpu[j];
        if (loc_end - loc_start > n_expd_toks) {
            n_expd_toks = loc_end - loc_start;
        }

        int current_exp = 0;
        exp_start_list[j * (num_routings + 1) + current_exp] = loc_start;
        for (int k = loc_start; k < loc_end; k += BLOCK_SIZE) {
            if ((exp_idx_list[k] >= 0) && (exp_idx_list[k] < num_routings) && (exp_idx_list[k] == current_exp + 1)) {
                current_exp = exp_idx_list[k];
                exp_start_list[j * (num_routings + 1) + current_exp] = k;
            } else if ((exp_idx_list[k] >= 0) && (exp_idx_list[k] < num_routings) && (exp_idx_list[k] != current_exp)) {
                for (int n = current_exp + 1; n <= exp_idx_list[k]; n++) {
                    exp_start_list[j * (num_routings + 1) + n] = k;
                }
                current_exp = exp_idx_list[k];
            }
        }
        if (current_exp != num_routings - 1) {
            printf("Err out of bound, %d.\n", current_exp);
        }
        for (int m = current_exp + 1; m <= num_routings; m++) 
            exp_start_list[j * (num_routings + 1) + m] = loc_end;
    }

    int *expert_start_list;
    cudaMalloc((void**)(&expert_start_list), sizeof(int) * top_k * (num_routings + 1));
    cudaMemcpy(expert_start_list, exp_start_list, sizeof(int) * top_k * (num_routings + 1), cudaMemcpyHostToDevice);
    
    delete exp_start_list;

    int grid_dim_y = 0, grid_dim_z_1 = 0, grid_dim_z_2 = 0, grid_dim_z_3 = 0;

    grid_dim_y = int(num_in_dims / (TILE * BLOCK_SIZE));
    if (num_in_dims % (TILE * BLOCK_SIZE) != 0)
        grid_dim_y += 1;
    
    grid_dim_z_1 = int(n_expd_toks / BLOCK_SIZE);
    if (n_expd_toks % BLOCK_SIZE != 0)
        grid_dim_z_1 += 1;
    
    grid_dim_z_2 = int(num_out_dims / BLOCK_SIZE);
    if (num_out_dims % BLOCK_SIZE != 0)
        grid_dim_z_2 += 1;

    grid_dim_z_3 = int(num_out_dims / (TILE * BLOCK_SIZE));
    if (num_out_dims % (TILE * BLOCK_SIZE) != 0)
        grid_dim_z_3 += 1;

    dim3 Block(WARP_SIZE, TILE);
    dim3 Grid(num_routings * top_k, grid_dim_y, grid_dim_z_1 + grid_dim_z_2 + grid_dim_z_3);

    fused_grad_bias_kernel_vectorized_full<<<Block, Grid>>>(
        j_b, j_x, j_w, j_y, w, x, \
        token_idx_list, expert_idx_list, expert_start_list, \
        m2s, grid_dim_z_1, num_tokens, num_in_dims, num_out_dims, num_routings
    );

    delete exp_idx_list;
    delete num_expanded_tokens_cpu;
}


__global__ void fused_grad_no_bias_kernel_vectorized_full(
    float *j_x,
    float *j_w,
    float *j_y,
    float *w,
    float *x,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *expert_start_list,
    bool m2s,
    int delta_1,
    int num_tokens,
    int num_in_dims,
    int num_out_dims,
    int num_routings
) {
    float zero_pad_float4[4] = {0.0f};
    __half zero_pad_half4[4] = {__float2half(0.0f)};

    int ThreadTok = threadIdx.x % BLOCK_SIZE;
    int ThreadOut = threadIdx.x / BLOCK_SIZE;
    int ThreadTile = threadIdx.y;

    int input_offset = 0, output_offset = 0, k_idx = blockIdx.x / num_routings, \
        loc_start=expert_start_list[k_idx * (num_routings+1) + blockIdx.x % num_routings], \
        loc_end=expert_start_list[k_idx * (num_routings+1) + blockIdx.x % num_routings + 1];
    int num_expanded_tokens = loc_end - loc_start;
    if (m2s) {
        output_offset = k_idx*num_tokens*num_out_dims;
    } else {
        input_offset = k_idx*num_tokens*num_in_dims;
    }

    __shared__ __half smem_tok[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ __half smem_op[BLOCK_SIZE * (TILE * BLOCK_SIZE)];
    __shared__ float smem_result[BLOCK_SIZE * (TILE * BLOCK_SIZE)];

    int task_id = -1; // 0 for ESMM, 1 for ESTMM, 2 for ESS.
    int delta_2 = num_out_dims % BLOCK_SIZE == 0? int(num_out_dims / BLOCK_SIZE): 1+int(num_out_dims / BLOCK_SIZE);
    if (blockIdx.z < delta_1)
        task_id = 0;
    else if (blockIdx.z < delta_1 + delta_2)
        task_id = 1;
    else
        task_id = 2;

    int tok_idx_0 = -1, exp_idx_0 = -1;
    if (blockIdx.z < delta_1) {
        if (blockIdx.z * BLOCK_SIZE + ThreadTok < num_expanded_tokens) {
            tok_idx_0 = token_idx_list[loc_start + blockIdx.z * BLOCK_SIZE + ThreadTok];
            exp_idx_0 = expert_idx_list[loc_start + blockIdx.z * BLOCK_SIZE];
        }
    }

    int exp_idx_1 = blockIdx.x % num_routings, loc_exp_start = -1, loc_exp_end = -1;
    loc_exp_start = expert_start_list[k_idx * (num_routings+1) + exp_idx_1]; // less than num_expanded_tokens
    loc_exp_end = expert_start_list[k_idx * (num_routings+1) + exp_idx_1 + 1];

    if ((task_id == 0) && (blockIdx.x % num_routings == 0)) {
        // Define the constants for ESMM
        int TileIn = blockIdx.y;

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
        for (int i = 0; i < num_out_dims; i += BLOCK_SIZE) {
            // load smem_tok with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx_0 < 0) || (tok_idx_0 >= num_tokens)) {
                    reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                } else {
                    int out_dims_base = i + ThreadOut * threads_per_chunk + j * 4;
                    if (out_dims_base + 3 < num_out_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base])[0];
                        Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base + 2 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base + 1];
                        float tmp_float_z = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base + 2];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base + 1 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base + 1];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx_0 * num_out_dims + out_dims_base];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else {
                        reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                    }
                }
            }

            // load smem_op with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((exp_idx_0 < 0) || (exp_idx_0 >= num_routings)) {
                    reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                } else {
                    int out_dims_base = i + ThreadTok;
                    if (out_dims_base >= num_out_dims) {
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                    } else {
                        int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                        if (in_dims_base + 3 < num_in_dims) {
                            float4 tmp_float4 = reinterpret_cast<float4 *>(&w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base])[0];
                            Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else if (in_dims_base + 2 < num_in_dims) {
                            float tmp_float_x = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base];
                            float tmp_float_y = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base + 1];
                            float tmp_float_z = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base + 2];
                            Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else if (in_dims_base + 1 < num_in_dims) {
                            float tmp_float_x = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base];
                            float tmp_float_y = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base + 1];
                            Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else if (in_dims_base < num_in_dims) {
                            float tmp_float_x = w[exp_idx_0 * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base];
                            Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                        } else {
                            reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                                reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                        }
                    }
                }
            }

            wmma::load_matrix_sync(a_frag, smem_tok, BLOCK_SIZE);
            wmma::load_matrix_sync(b_frag, smem_op + ThreadTile * BLOCK_SIZE, TILE * BLOCK_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(smem_result + ThreadTile * BLOCK_SIZE, c_frag, TILE * BLOCK_SIZE, wmma::mem_row_major);

        // write back to global memory with vectorized memory access
        if (m2s) {
            if ((tok_idx_0 >= 0) && (tok_idx_0 < num_tokens)) {
                int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk;
                for (int i = 0; i < threads_per_chunk; i++) {
                    if (in_dims_base + i < num_in_dims) {
                        atomicAdd(j_x + input_offset + tok_idx_0 * num_in_dims + in_dims_base + i, \
                            smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i]);
                        // at::native::fastAtomicAdd(j_x, input_offset + tok_idx_0 * num_in_dims + in_dims_base + i, num_tokens * num_in_dims, \
                        //     smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i], true);
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx_0 >= 0) && (tok_idx_0 < num_tokens)) {
                    int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (in_dims_base + 3 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        float4 tmp_float4_ori = reinterpret_cast<float4 *>(&j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base])[0];
                        float4 tmp_float4_dst = make_float4(tmp_float4_ori.x + tmp_float4.x, tmp_float4_ori.y + tmp_float4.y, \
                                                            tmp_float4_ori.z + tmp_float4.z, tmp_float4_ori.w + tmp_float4.w);
                        reinterpret_cast<float4 *>(&j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base])[0] = tmp_float4_dst;
                    } else if (in_dims_base + 2 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base] += tmp_float4.x;
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base + 1] += tmp_float4.y;
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base + 2] += tmp_float4.z;
                    } else if (in_dims_base + 1 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base] += tmp_float4.x;
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base + 1] += tmp_float4.y;
                    } else if (in_dims_base < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                        j_x[input_offset + tok_idx_0 * num_in_dims + in_dims_base] += tmp_float4.x;
                    }
                }
            }
        }
    } else if (task_id == 1) {
        // Define the constants for ESTMM
        int BlockOut = blockIdx.z;
        int TileIn = blockIdx.y;
        int exp_idx = expert_idx_list[loc_exp_start];

        wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
        for (int i = loc_exp_start; i < loc_exp_end; i += BLOCK_SIZE) {
            int tok_idx = token_idx_list[i + ThreadTok];

            // load smem_tok with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx < 0) || (tok_idx >= num_tokens)) {
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                } else {
                    int out_dims_base = BlockOut * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (out_dims_base + 3 < num_out_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&j_y[output_offset + tok_idx * num_out_dims + out_dims_base])[0];
                        Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else if (out_dims_base + 2 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 1];
                        float tmp_float_z = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 2];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else if (out_dims_base + 1 < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        float tmp_float_y = j_y[output_offset + tok_idx * num_out_dims + out_dims_base + 1];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else if (out_dims_base < num_out_dims) {
                        float tmp_float_x = j_y[output_offset + tok_idx * num_out_dims + out_dims_base];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = tmp_half4.x;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = tmp_half4.y;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = tmp_half4.z;
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = tmp_half4.w;
                    } else {
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 1) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 2) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                        smem_tok[(ThreadOut * threads_per_chunk + j * 4 + 3) * BLOCK_SIZE + ThreadTok] = __float2half(0.0f);
                    }
                }
            }

            // load smem_op with vectorized memory access
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if ((tok_idx < 0) || (tok_idx >= num_tokens)) {
                    reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(&zero_pad_half4)[0];
                } else {
                    int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (in_dims_base + 3 < num_in_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&x[input_offset + tok_idx * num_in_dims + in_dims_base])[0];
                        Half4 tmp_half4 = make_Half4(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else if (in_dims_base + 2 < num_in_dims) {
                        float tmp_float_x = x[input_offset + tok_idx * num_in_dims + in_dims_base];
                        float tmp_float_y = x[input_offset + tok_idx * num_in_dims + in_dims_base + 1];
                        float tmp_float_z = x[input_offset + tok_idx * num_in_dims + in_dims_base + 2];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else if (in_dims_base + 1 < num_in_dims) {
                        float tmp_float_x = x[input_offset + tok_idx * num_in_dims + in_dims_base];
                        float tmp_float_y = x[input_offset + tok_idx * num_in_dims + in_dims_base + 1];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else if (in_dims_base < num_in_dims) {
                        float tmp_float_x = x[input_offset + tok_idx * num_in_dims + in_dims_base];
                        Half4 tmp_half4 = make_Half4(tmp_float_x, 0.0f, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = \
                            tmp_half4;
                    } else {
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                            reinterpret_cast<Half4 *>(&zero_pad_half4)[0];
                    }
                }
            }

            wmma::load_matrix_sync(a_frag, smem_tok, BLOCK_SIZE);
            wmma::load_matrix_sync(b_frag, smem_op + ThreadTile * BLOCK_SIZE, TILE * BLOCK_SIZE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(smem_result + ThreadTile * BLOCK_SIZE, c_frag, TILE * BLOCK_SIZE, wmma::mem_row_major);

        // write back to global memory with vectorized memory access
        // gradients for W are accumulative in all cases
        int out_dims_base = BlockOut * BLOCK_SIZE + ThreadTok;
        if (out_dims_base < num_out_dims) {
            int in_dims_base = TileIn * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk;
            for (int i = 0; i < threads_per_chunk; i++) {
                if (in_dims_base + i < num_in_dims) {
                    atomicAdd(j_w + exp_idx * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base, \
                        smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i]);
                    // at::native::fastAtomicAdd(j_w, exp_idx * num_out_dims * num_in_dims + out_dims_base * num_in_dims + in_dims_base, \
                    //     num_routings * num_in_dims * num_out_dims, \
                    //     smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i], true);
                }
            }
        }
    }
}

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
    int device_idx=0
) {
    /*
        We build the CUDA kernel with a unified manner 
        for memory access and computation.

        Inputs:
            j_y [num_tokens, num_out_dims]
            w [num_experts, num_out_dims, num_in_dims]
            x [num_tokens, num_in_dims]
        
        Outputs:
            j_b [num_experts, num_outputs]
            j_x [num_tokens, num_in_dims]
            j_w [num_experts, num_out_dims, num_in_dims]

        We re-organiza the blocks to fuse the 3 operators in parallel
        Assuming that TILE == UNROLL
            For ESMM:  (num_expanded_tokens // BLOCK_SIZE, num_in_dims // (TILE * BLOCK_SIZE), 1)
            For ESTMM: (num_routings,                      num_in_dims // (TILE * BLOCK_SIZE), num_out_dims // BLOCK_SIZE)
            
            Padding and flipping:
            For ESMM:  (num_routings, num_in_dims // (TILE * BLOCK_SIZE), num_expanded_tokens // BLOCK_SIZE)
            For ESTMM: (num_routings, num_in_dims // (TILE * BLOCK_SIZE), num_out_dims // BLOCK_SIZE)
    */
    int n_expd_toks = 0, loc_start = 0, loc_end = 0;

    int *num_expanded_tokens_cpu = new int [top_k + 1]();
    cudaMemcpy(num_expanded_tokens_cpu, num_expanded_tokens, sizeof(int) * (top_k + 1), cudaMemcpyDeviceToHost);

    int *exp_idx_list = new int [num_expanded_tokens_cpu[top_k]]();
    cudaMemcpy(exp_idx_list, expert_idx_list, sizeof(int) * num_expanded_tokens_cpu[top_k], cudaMemcpyDeviceToHost);

    int *exp_start_list = new int [top_k * (num_routings + 1)]();
    for (int j = 0; j < top_k; j++) {
        loc_end = num_expanded_tokens_cpu[j + 1];
        loc_start = num_expanded_tokens_cpu[j];
        if (loc_end - loc_start > n_expd_toks) {
            n_expd_toks = loc_end - loc_start;
        }

        int current_exp = 0;
        exp_start_list[j * (num_routings + 1) + current_exp] = loc_start;
        for (int k = loc_start; k < loc_end; k += BLOCK_SIZE) {
            if ((exp_idx_list[k] >= 0) && (exp_idx_list[k] < num_routings) && (exp_idx_list[k] == current_exp + 1)) {
                current_exp = exp_idx_list[k];
                exp_start_list[j * (num_routings + 1) + current_exp] = k;
            } else if ((exp_idx_list[k] >= 0) && (exp_idx_list[k] < num_routings) && (exp_idx_list[k] != current_exp)) {
                for (int n = current_exp + 1; n <= exp_idx_list[k]; n++) {
                    exp_start_list[j * (num_routings + 1) + n] = k;
                }
                current_exp = exp_idx_list[k];
            }
        }
        if (current_exp != num_routings - 1) {
            printf("Err out of bound, %d.\n", current_exp);
        }
        for (int m = current_exp + 1; m <= num_routings; m++) 
            exp_start_list[j * (num_routings + 1) + m] = loc_end;
    }

    int *expert_start_list;
    cudaMalloc((void**)(&expert_start_list), sizeof(int) * top_k * (num_routings + 1));
    cudaMemcpy(expert_start_list, exp_start_list, sizeof(int) * top_k * (num_routings + 1), cudaMemcpyHostToDevice);
    
    delete exp_start_list;

    int grid_dim_y = 0, grid_dim_z_1 = 0, grid_dim_z_2 = 0;

    grid_dim_y = int(num_in_dims / (TILE * BLOCK_SIZE));
    if (num_in_dims % (TILE * BLOCK_SIZE) != 0)
        grid_dim_y += 1;
    
    grid_dim_z_1 = int(n_expd_toks / BLOCK_SIZE);
    if (n_expd_toks % BLOCK_SIZE != 0)
        grid_dim_z_1 += 1;
    
    grid_dim_z_2 = int(num_out_dims / BLOCK_SIZE);
    if (num_out_dims % BLOCK_SIZE != 0)
        grid_dim_z_2 += 1;

    dim3 Block(WARP_SIZE, TILE);
    dim3 Grid(num_routings * top_k, grid_dim_y, grid_dim_z_1 + grid_dim_z_2);

    fused_grad_no_bias_kernel_vectorized_full<<<Block, Grid>>>(
        j_x, j_w, j_y, w, x, \
        token_idx_list, expert_idx_list, expert_start_list, \
        m2s, grid_dim_z_1, num_tokens, num_in_dims, num_out_dims, num_routings
    );

    delete exp_idx_list;
    delete num_expanded_tokens_cpu;
}