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

#define BLOCK_SIZE 16
#define UNROLL_SIZE 32

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

__global__ void count_tokens_vectorized(
    int *tokens_per_expert,
    int *routings,
    int list_start_idx,
    int num_tokens,
    int num_routings
) {
    int chunk_start_idx = list_start_idx + threadIdx.x * BLOCK_SIZE;
    int tok_chunk[BLOCK_SIZE] = {-1};
    __shared__ int toks_per_exp[BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];
    if (threadIdx.x == 0) {
        int tmp_start = 0;
        while (tmp_start < num_routings) {
            if (tmp_start + 3 < num_routings) {
                int4 tmp_int4 = reinterpret_cast<int4 *>(&tokens_per_expert[tmp_start])[0];
                reinterpret_cast<int4 *>(&toks_per_exp[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 2 < num_routings) {
                int tmp_int_0 = tokens_per_expert[tmp_start];
                int tmp_int_1 = tokens_per_expert[tmp_start + 1];
                int tmp_int_2 = tokens_per_expert[tmp_start + 2];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, tmp_int_2, 0);
                reinterpret_cast<int4 *>(&toks_per_exp[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 1 < num_routings) {
                int tmp_int_0 = tokens_per_expert[tmp_start];
                int tmp_int_1 = tokens_per_expert[tmp_start + 1];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, 0, 0);
                reinterpret_cast<int4 *>(&toks_per_exp[tmp_start])[0] = tmp_int4;
            } else {
                int tmp_int_0 = tokens_per_expert[tmp_start];
                int4 tmp_int4 = make_int4(tmp_int_0, 0, 0, 0);
                reinterpret_cast<int4 *>(&toks_per_exp[tmp_start])[0] = tmp_int4;
            }
            tmp_start += 4;
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < int(BLOCK_SIZE / 4); i++) {
        if (chunk_start_idx + i * 4 + 3 < num_tokens) {
            int4 tmp_int4 = reinterpret_cast<int4 *>(&routings[chunk_start_idx + i * 4])[0];
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        } else if (chunk_start_idx + i * 4 + 2 < num_tokens) {
            int tmp_int_0 = routings[chunk_start_idx + i * 4];
            int tmp_int_1 = routings[chunk_start_idx + i * 4 + 1];
            int tmp_int_2 = routings[chunk_start_idx + i * 4 + 2];
            int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, tmp_int_2, -1);
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        } else if (chunk_start_idx + i * 4 + 1 < num_tokens) {
            int tmp_int_0 = routings[chunk_start_idx + i * 4];
            int tmp_int_1 = routings[chunk_start_idx + i * 4 + 1];
            int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, -1, -1);
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        } else if (chunk_start_idx + i * 4 < num_tokens) {
            int tmp_int_0 = routings[chunk_start_idx + i * 4];
            int4 tmp_int4 = make_int4(tmp_int_0, -1, -1, -1);
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        }
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (tok_chunk[i] >= 0) {
            atomicAdd(toks_per_exp + tok_chunk[i], 1);
            // at::native::fastAtomicAdd(toks_per_exp, tok_chunk[i], num_routings, 1, true);
        }
    }
    __syncthreads();

    // write back to global memory
    if (threadIdx.x == 0) {
        int tmp_start = 0;
        while (tmp_start < num_routings) {
            if (tmp_start + 3 < num_routings) {
                int4 tmp_int4 = reinterpret_cast<int4 *>(&toks_per_exp[tmp_start])[0];
                reinterpret_cast<int4 *>(&tokens_per_expert[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 2 < num_routings) {
                int tmp_int_0 = toks_per_exp[tmp_start];
                int tmp_int_1 = toks_per_exp[tmp_start + 1];
                int tmp_int_2 = toks_per_exp[tmp_start + 2];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, tmp_int_2, 0);
                reinterpret_cast<int4 *>(&tokens_per_expert[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 1 < num_routings) {
                int tmp_int_0 = toks_per_exp[tmp_start];
                int tmp_int_1 = toks_per_exp[tmp_start + 1];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, 0, 0);
                reinterpret_cast<int4 *>(&tokens_per_expert[tmp_start])[0] = tmp_int4;
            } else {
                int tmp_int_0 = toks_per_exp[tmp_start];
                int4 tmp_int4 = make_int4(tmp_int_0, 0, 0, 0);
                reinterpret_cast<int4 *>(&tokens_per_expert[tmp_start])[0] = tmp_int4;
            }
            tmp_start += 4;
        }
    }
}

void launch_count(
    int *tokens_per_expert,
    int *expanded_tokens_per_expert,
    int *routings,
    int num_tokens,
    int num_routings,
    int device_idx
) {
    cudaDeviceProp prop;
    int num_devices=1, num_max_threads=0, num_threads=0, list_start_idx=0;

    HANDLE_ERROR(cudaGetDeviceCount(&num_devices));
    if (device_idx < num_devices)
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_idx));
    else
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    num_max_threads = prop.maxThreadsPerBlock;

    if (num_tokens < num_max_threads * BLOCK_SIZE) {
        list_start_idx = 0;
        num_threads = int(num_tokens / BLOCK_SIZE);
        if (num_tokens % BLOCK_SIZE != 0)
            num_threads += 1;
        dim3 Threads(num_threads);
        count_tokens_vectorized<<<1, Threads>>>(tokens_per_expert, routings, list_start_idx, num_tokens, num_routings);
    } else {
        list_start_idx = 0;
        while (list_start_idx < num_tokens) {
            num_threads = min(int((num_tokens - list_start_idx) / BLOCK_SIZE), num_max_threads);
            if ((num_threads < num_max_threads) && ((num_tokens - list_start_idx) % BLOCK_SIZE != 0))
                num_threads += 1;
            dim3 Threads(num_threads);
            count_tokens_vectorized<<<1, Threads>>>(tokens_per_expert, routings, list_start_idx, num_tokens, num_routings);
            list_start_idx += num_max_threads * BLOCK_SIZE;
        }
    }

    int *cum_start_exp = new int [num_routings + 1];
    cudaMemcpy(cum_start_exp, tokens_per_expert, sizeof(int) * (num_routings), cudaMemcpyDeviceToHost);
    int start_idx = 0, tmp = 0;
    for (int i = 0; i < num_routings; i++) {
        if (cum_start_exp[i] % BLOCK_SIZE != 0)
            cum_start_exp[i] = (1 + int(cum_start_exp[i] / BLOCK_SIZE)) * BLOCK_SIZE;
    }
    
    for (int i = 0; i < num_routings + 1; i++) {
        if (i < num_routings)
            tmp = cum_start_exp[i];
        cum_start_exp[i] = start_idx;
        if (i < num_routings)
            start_idx += tmp;
    }
    cudaMemcpy(expanded_tokens_per_expert, cum_start_exp, sizeof(int) * (num_routings + 1), cudaMemcpyHostToDevice);
    delete cum_start_exp;
}

__global__ void assign_tokens_vectorized(
    int *token_idx_list,
    int *expert_idx_list,
    int *expert_start_ids,
    int *routings,
    int list_start_idx,
    int num_tokens,
    int num_routings
) {
    int chunk_start_idx = list_start_idx + threadIdx.x * BLOCK_SIZE;
    int tok_chunk[BLOCK_SIZE] = {-1};
    __shared__ int exp_start_ids[BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];
    if (threadIdx.x == 0) {
        int tmp_start = 0;
        while (tmp_start < num_routings) {
            if (tmp_start + 3 < num_routings) {
                int4 tmp_int4 = reinterpret_cast<int4 *>(&expert_start_ids[tmp_start])[0];
                reinterpret_cast<int4 *>(&exp_start_ids[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 2 < num_routings) {
                int tmp_int_0 = expert_start_ids[tmp_start];
                int tmp_int_1 = expert_start_ids[tmp_start + 1];
                int tmp_int_2 = expert_start_ids[tmp_start + 2];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, tmp_int_2, 0);
                reinterpret_cast<int4 *>(&exp_start_ids[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 1 < num_routings) {
                int tmp_int_0 = expert_start_ids[tmp_start];
                int tmp_int_1 = expert_start_ids[tmp_start + 1];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, 0, 0);
                reinterpret_cast<int4 *>(&exp_start_ids[tmp_start])[0] = tmp_int4;
            } else {
                int tmp_int_0 = expert_start_ids[tmp_start];
                int4 tmp_int4 = make_int4(tmp_int_0, 0, 0, 0);
                reinterpret_cast<int4 *>(&exp_start_ids[tmp_start])[0] = tmp_int4;
            }
            tmp_start += 4;
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < int(BLOCK_SIZE / 4); i++) {
        if (chunk_start_idx + i * 4 + 3 < num_tokens) {
            int4 tmp_int4 = reinterpret_cast<int4 *>(&routings[chunk_start_idx + i * 4])[0];
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        } else if (chunk_start_idx + i * 4 + 2 < num_tokens) {
            int tmp_int_0 = routings[chunk_start_idx + i * 4];
            int tmp_int_1 = routings[chunk_start_idx + i * 4 + 1];
            int tmp_int_2 = routings[chunk_start_idx + i * 4 + 2];
            int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, tmp_int_2, -1);
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        } else if (chunk_start_idx + i * 4 + 1 < num_tokens) {
            int tmp_int_0 = routings[chunk_start_idx + i * 4];
            int tmp_int_1 = routings[chunk_start_idx + i * 4 + 1];
            int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, -1, -1);
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        } else if (chunk_start_idx + i * 4 < num_tokens) {
            int tmp_int_0 = routings[chunk_start_idx + i * 4];
            int4 tmp_int4 = make_int4(tmp_int_0, -1, -1, -1);
            reinterpret_cast<int4 *>(&tok_chunk[i * 4])[0] = tmp_int4;
        }
    }

#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (tok_chunk[i] >= 0) {
            int write_idx = atomicAdd(&exp_start_ids[tok_chunk[i]], 1);
            int token_idx = chunk_start_idx + i;
            token_idx_list[write_idx] = token_idx;
            expert_idx_list[write_idx] = routings[token_idx];
        }
    }
    __syncthreads();

    // write back to global memory
    if (threadIdx.x == 0) {
        int tmp_start = 0;
        while (tmp_start < num_routings) {
            if (tmp_start + 3 < num_routings) {
                int4 tmp_int4 = reinterpret_cast<int4 *>(&exp_start_ids[tmp_start])[0];
                reinterpret_cast<int4 *>(&expert_start_ids[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 2 < num_routings) {
                int tmp_int_0 = exp_start_ids[tmp_start];
                int tmp_int_1 = exp_start_ids[tmp_start + 1];
                int tmp_int_2 = exp_start_ids[tmp_start + 2];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, tmp_int_2, 0);
                reinterpret_cast<int4 *>(&expert_start_ids[tmp_start])[0] = tmp_int4;
            } else if (tmp_start + 1 < num_routings) {
                int tmp_int_0 = exp_start_ids[tmp_start];
                int tmp_int_1 = exp_start_ids[tmp_start + 1];
                int4 tmp_int4 = make_int4(tmp_int_0, tmp_int_1, 0, 0);
                reinterpret_cast<int4 *>(&expert_start_ids[tmp_start])[0] = tmp_int4;
            } else {
                int tmp_int_0 = exp_start_ids[tmp_start];
                int4 tmp_int4 = make_int4(tmp_int_0, 0, 0, 0);
                reinterpret_cast<int4 *>(&expert_start_ids[tmp_start])[0] = tmp_int4;
            }
            tmp_start += 4;
        }
    }
}

void launch_assign(
    int *token_idx_list,
    int *expert_idx_list,
    const int *expanded_tokens_per_expert,
    int *routings,
    int num_tokens,
    int num_routings,
    int device_idx
) {
    cudaDeviceProp prop;
    int num_devices=1, num_max_threads=0, num_threads=0, list_start_idx=0;

    HANDLE_ERROR(cudaGetDeviceCount(&num_devices));
    if (device_idx < num_devices)
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device_idx));
    else
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    num_max_threads = prop.maxThreadsPerBlock;

    int *expert_start_idx = new int [num_routings];
    cudaMemcpy(expert_start_idx, expanded_tokens_per_expert, sizeof(int) * (num_routings), cudaMemcpyDeviceToHost);
    int *exp_start_ids;
    cudaMalloc((void**)(&exp_start_ids), sizeof(int) * (num_routings));
    cudaMemcpy(exp_start_ids, expert_start_idx, sizeof(int) * (num_routings), cudaMemcpyHostToDevice);
    delete expert_start_idx;

    if (num_tokens < num_max_threads * BLOCK_SIZE) {
        list_start_idx = 0;
        num_threads = int(num_tokens / BLOCK_SIZE);
        if (num_tokens % BLOCK_SIZE != 0)
            num_threads += 1;
        dim3 Threads(num_threads);
        assign_tokens_vectorized<<<1, Threads>>>(token_idx_list, expert_idx_list, exp_start_ids, routings, \
            list_start_idx, num_tokens, num_routings);
    } else {
        list_start_idx = 0;
        while (list_start_idx < num_tokens) {
            num_threads = min(int((num_tokens - list_start_idx) / BLOCK_SIZE), num_max_threads);
            if ((num_threads < num_max_threads) && ((num_tokens - list_start_idx) % BLOCK_SIZE != 0))
                num_threads += 1;
            dim3 Threads(num_threads);
            assign_tokens_vectorized<<<1, Threads>>>(token_idx_list, expert_idx_list, exp_start_ids, routings, \
                list_start_idx, num_tokens, num_routings);
            list_start_idx += num_max_threads * BLOCK_SIZE;
        }
    }

    cudaFree(exp_start_ids);
}