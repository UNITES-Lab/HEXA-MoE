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
#define TILE 4

const int threads_per_chunk = BLOCK_SIZE / (WARP_SIZE / BLOCK_SIZE); // 8
const int chunks_per_row = WARP_SIZE / BLOCK_SIZE; // 2

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

__forceinline__ __host__ __device__ Half4 make_Half4_from_float(const float x, const float y, const float z, float w)
{
    Half4 h4;

    h4.x = __float2half(x);
    h4.y = __float2half(y);
    h4.z = __float2half(z);
    h4.w = __float2half(w);

    return h4;
}

__forceinline__ __host__ __device__ Half4 make_Half4_from_half(const __half x, const __half y, const __half z, __half w)
{
    Half4 h4;

    h4.x = x;
    h4.y = y;
    h4.z = z;
    h4.w = w;

    return h4;
}

__global__ void esmm_shared_tensor_kernel_vectorized_accum_mix(
    float *result,
    __half *tokens,
    float *weights,
    float *bias,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens_list,
    bool m2s,
    int num_tokens,
    int num_in_dims,
    int num_out_dims,
    int num_routings
) {
    float zero_pad_float4[4] = {0.0f};
    __half zero_pad_half4[4] = {__float2half(0.0f)};

    int BlockTok = blockIdx.x;
    int TileOut = blockIdx.y;

    int ThreadTok = threadIdx.x % BLOCK_SIZE;
    int ThreadOut = threadIdx.x / BLOCK_SIZE;
    int ThreadTile = threadIdx.y;

    int input_offset = 0, output_offset = 0, k_idx = blockIdx.z;
    int num_expanded_tokens = num_expanded_tokens_list[k_idx + 1] - num_expanded_tokens_list[k_idx];
    int loc_start = num_expanded_tokens_list[k_idx];
    if (m2s) {
        input_offset = k_idx*num_tokens*num_in_dims;
    } else {
        output_offset = k_idx*num_tokens*num_out_dims;
    }

    int tok_idx = -1, exp_idx = -1;
    if (BlockTok * BLOCK_SIZE + ThreadTok < num_expanded_tokens) {
        tok_idx = token_idx_list[loc_start + BlockTok * BLOCK_SIZE + ThreadTok];
        exp_idx = expert_idx_list[loc_start + BlockTok * BLOCK_SIZE];
    }

    __shared__ __half smem_tok[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ __half smem_op[BLOCK_SIZE * TILE * BLOCK_SIZE];
    __shared__ float smem_bias[TILE * BLOCK_SIZE];
    __shared__ float smem_result[BLOCK_SIZE * TILE * BLOCK_SIZE];

    int out_dims_base = TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadTok;
    if ((out_dims_base < num_out_dims) && (ThreadOut == 0)) {
        if ((bias != nullptr) && (exp_idx >= 0) && (exp_idx < num_routings)) {
            smem_bias[ThreadTile * BLOCK_SIZE + ThreadTok] = bias[exp_idx * num_out_dims + out_dims_base];
        } else {
            smem_bias[ThreadTile * BLOCK_SIZE + ThreadTok] = 0.0f;
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int i = 0; i < num_in_dims; i += BLOCK_SIZE) {
        // load smem_tok with vectorized memory access
#pragma unroll
        for (int j = 0; j < 2; j++) {
            if ((tok_idx < 0) || (tok_idx >= num_tokens)) {
                reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                    reinterpret_cast<Half4 *>(zero_pad_half4)[0];
            } else {
                int in_dims_base = i + ThreadOut * threads_per_chunk + j * 4;
                if (in_dims_base + 3 < num_in_dims) {
                    Half4 tmp_half4 = reinterpret_cast<Half4 *>(&tokens[input_offset + tok_idx * num_in_dims + in_dims_base])[0];
                    reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                } else if (in_dims_base + 2 < num_in_dims) {
                    __half tmp_float_x = tokens[input_offset + tok_idx * num_in_dims + in_dims_base];
                    __half tmp_float_y = tokens[input_offset + tok_idx * num_in_dims + in_dims_base + 1];
                    __half tmp_float_z = tokens[input_offset + tok_idx * num_in_dims + in_dims_base + 2];
                    Half4 tmp_half4 = make_Half4_from_half(tmp_float_x, tmp_float_y, tmp_float_z, __float2half(0.0f));
                    reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                } else if (in_dims_base + 1 < num_in_dims) {
                    __half tmp_float_x = tokens[input_offset + tok_idx * num_in_dims + in_dims_base];
                    __half tmp_float_y = tokens[input_offset + tok_idx * num_in_dims + in_dims_base + 1];
                    Half4 tmp_half4 = make_Half4_from_half(tmp_float_x, tmp_float_y, __float2half(0.0f), __float2half(0.0f));
                    reinterpret_cast<Half4 *>(&smem_tok[ThreadTok * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                } else if (in_dims_base < num_in_dims) {
                    __half tmp_float_x = tokens[input_offset + tok_idx * num_in_dims + in_dims_base];
                    Half4 tmp_half4 = make_Half4_from_half(tmp_float_x, __float2half(0.0f), __float2half(0.0f), __float2half(0.0f));
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
            if ((exp_idx < 0) || (exp_idx >= num_routings)) {
                reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                    reinterpret_cast<Half4 *>(zero_pad_half4)[0];
            } else {
                int in_dims_base = i + ThreadTok;
                if (in_dims_base >= num_in_dims) {
                    reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] =\
                        reinterpret_cast<Half4 *>(zero_pad_half4)[0];
                } else {
                    int out_dims_base = TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                    if (out_dims_base + 3 < num_out_dims) {
                        float4 tmp_float4 = reinterpret_cast<float4 *>(&weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base])[0];
                        Half4 tmp_half4 = make_Half4_from_float(tmp_float4.x, tmp_float4.y, tmp_float4.z, tmp_float4.w);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base + 2 < num_out_dims) {
                        float tmp_float_x = weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base];
                        float tmp_float_y = weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base + 1];
                        float tmp_float_z = weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base + 2];
                        Half4 tmp_half4 = make_Half4_from_float(tmp_float_x, tmp_float_y, tmp_float_z, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base + 1 < num_out_dims) {
                        float tmp_float_x = weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base];
                        float tmp_float_y = weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base + 1];
                        Half4 tmp_half4 = make_Half4_from_float(tmp_float_x, tmp_float_y, 0.0f, 0.0f);
                        reinterpret_cast<Half4 *>(&smem_op[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0] = tmp_half4;
                    } else if (out_dims_base < num_out_dims) {
                        float tmp_float_x = weights[exp_idx * num_in_dims * num_out_dims + in_dims_base * num_out_dims + out_dims_base];
                        Half4 tmp_half4 = make_Half4_from_float(tmp_float_x, 0.0f, 0.0f, 0.0f);
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
        // Using fastatomicadd to merge different routings
        if ((tok_idx >= 0) && (tok_idx < num_tokens)) {
            int out_dims_base = TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk;
            for (int i = 0; i < threads_per_chunk; i++) {
                if (out_dims_base + i < num_out_dims) {
                    float tmp_bias = smem_bias[ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i];
                    float tmp_result = smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + i];
                    tmp_result = tmp_result + tmp_bias;
                    at::native::fastAtomicAdd(result, output_offset + tok_idx * num_out_dims + out_dims_base + i, \
                        num_tokens * num_out_dims, tmp_result, true);
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            if ((tok_idx >= 0) && (tok_idx < num_tokens)) {
                int out_dims_base = TileOut * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4;
                if (out_dims_base + 3 < num_out_dims) {
                    float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    float4 tmp_bias = reinterpret_cast<float4 *>(&smem_bias[ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    float4 tmp_float4_ori = reinterpret_cast<float4 *>(&result[output_offset + tok_idx * num_out_dims + out_dims_base])[0];
                    float4 tmp_float4_dst = make_float4(tmp_float4_ori.x + tmp_float4.x + tmp_bias.x, tmp_float4_ori.y + tmp_float4.y + tmp_bias.y, \
                                                        tmp_float4_ori.z + tmp_float4.z + tmp_bias.z, tmp_float4_ori.w + tmp_float4.w + tmp_bias.w);
                    reinterpret_cast<float4 *>(&result[output_offset + tok_idx * num_out_dims + out_dims_base])[0] = tmp_float4_dst;
                } else if (out_dims_base + 2 < num_out_dims) {
                    float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    float4 tmp_bias = reinterpret_cast<float4 *>(&smem_bias[ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    result[output_offset + tok_idx * num_out_dims + out_dims_base] += tmp_float4.x + tmp_bias.x;
                    result[output_offset + tok_idx * num_out_dims + out_dims_base + 1] += tmp_float4.y + tmp_bias.y;
                    result[output_offset + tok_idx * num_out_dims + out_dims_base + 2] += tmp_float4.z + tmp_bias.z;
                } else if (out_dims_base + 1 < num_out_dims) {
                    float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    float4 tmp_bias = reinterpret_cast<float4 *>(&smem_bias[ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    result[output_offset + tok_idx * num_out_dims + out_dims_base] += tmp_float4.x + tmp_bias.x;
                    result[output_offset + tok_idx * num_out_dims + out_dims_base + 1] += tmp_float4.y + tmp_bias.y;
                } else if (out_dims_base < num_out_dims) {
                    float4 tmp_float4 = reinterpret_cast<float4 *>(&smem_result[ThreadTok * TILE * BLOCK_SIZE + ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    float4 tmp_bias = reinterpret_cast<float4 *>(&smem_bias[ThreadTile * BLOCK_SIZE + ThreadOut * threads_per_chunk + j * 4])[0];
                    result[output_offset + tok_idx * num_out_dims + out_dims_base] += tmp_float4.x + tmp_bias.x;
                }
            }
        }
    }
}


void launch_esmm_accum_shared_tensor_mix(
    float *result,
    __half *tokens,
    float *weights,
    float *bias,
    const int *token_idx_list,
    const int *expert_idx_list,
    const int *num_expanded_tokens, // length top_k + 1 and start with 0
    bool m2s,
    int num_tokens,
    int top_k,
    int num_in_dims,
    int num_out_dims,
    int num_routings,
    int device_idx=0
) {
    int grid_dim_tok = 0, grid_dim_out = 0, n_expd_toks = 0, loc_start = 0, loc_end = 0;

    int *num_expanded_tokens_cpu = new int [top_k + 1]();
    cudaMemcpy(num_expanded_tokens_cpu, num_expanded_tokens, sizeof(int) * (top_k + 1), cudaMemcpyDeviceToHost);

    int *exp_idx_list = new int [num_expanded_tokens_cpu[top_k]]();
    cudaMemcpy(exp_idx_list, expert_idx_list, sizeof(int) * num_expanded_tokens_cpu[top_k], cudaMemcpyDeviceToHost);

    // Parallel for top-k routing using thread grids
    for (int j = 0; j < top_k; j++) {
        loc_end = num_expanded_tokens_cpu[j + 1];
        loc_start = num_expanded_tokens_cpu[j];
        if (loc_end - loc_start > n_expd_toks) {
            n_expd_toks = loc_end - loc_start;
        }
    }

    grid_dim_tok = int(n_expd_toks / BLOCK_SIZE);
    grid_dim_out = int(num_out_dims / (TILE * BLOCK_SIZE));

    if (n_expd_toks % BLOCK_SIZE != 0)
        grid_dim_tok += 1;
    if (num_out_dims % (TILE * BLOCK_SIZE) != 0)
        grid_dim_out += 1;

    dim3 Block(WARP_SIZE, TILE);
    dim3 Grid(grid_dim_tok, grid_dim_out, top_k);

    esmm_shared_tensor_kernel_vectorized_accum_mix<<<Grid, Block>>>(
        result, tokens, weights, bias, \
        token_idx_list, expert_idx_list, num_expanded_tokens, \
        m2s, num_tokens, num_in_dims, num_out_dims, num_routings
    );

    delete num_expanded_tokens_cpu;
}