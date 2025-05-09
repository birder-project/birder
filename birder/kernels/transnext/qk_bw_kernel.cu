/*!
**************************************************************************************************
* TransNeXt
* Taken from:
* https://github.com/DaiShiResearch/TransNeXt/blob/main/swattention_extension
* Licensed under the Apache License, Version 2.0
**************************************************************************************************
*/

#include <torch/extension.h>
#include <cmath>

template <typename scalar_t>
__global__ void qk_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> keys,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_queries,
    int height,
    int width,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (keys.size(0)* keys.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < keys.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < keys.size(3)){
                const int b = x / keys.size(1);
                const int h = x - b * keys.size(1);
                const int i = y / width;
                const int j = y - i * width;
                const int start_i = i-(kernel_size-1)/2;
                const int start_j = j-(kernel_size-1)/2;
                scalar_t updt = scalar_t(0);
                int k_offset=0;

                #pragma unroll
                for (int current_i=start_i; current_i<(start_i+kernel_size); ++current_i){
                    #pragma unroll
                    for (int current_j=start_j; current_j<(start_j+kernel_size); ++current_j){
                        if (((current_i>=0) && (current_i<height))&& ((current_j>=0) && (current_j<width))){
                            const int current_offset=current_i*width+current_j;
                            updt += d_attn_weight[b][h][y][k_offset] * keys[b][h][current_offset][z]; 
                        }
                        ++k_offset;
                    }
                }
                d_queries[b][h][y][z]=updt; 

            }

        }
    }
}


template <typename scalar_t>
__global__ void qk_inverse_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> queries,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_keys,
    int height,
    int width,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_keys.size(0)* d_keys.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_keys.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_keys.size(3)){
                const int b = x / d_keys.size(1);
                const int h = x - b * d_keys.size(1);
                const int i = y / width;
                const int j = y - i * width;
                const int q_start_i = i-kernel_size/2;
                const int q_end_i = i+1+(kernel_size-1)/2;
                const int q_start_j = j-kernel_size/2;
                const int q_end_j = j+1+(kernel_size-1)/2;
                scalar_t updt = scalar_t(0);
                int k_offset=kernel_size*kernel_size;
                #pragma unroll
                for (int current_i=q_start_i; current_i<q_end_i; ++current_i){
                    #pragma unroll
                    for (int current_j=q_start_j; current_j<q_end_j; ++current_j){
                        --k_offset;
                        if (((current_i>=0) && (current_i<height))&& ((current_j>=0) && (current_j<width))){
                            const int current_offset=current_i*width+current_j;
                            updt += d_attn_weight[b][h][current_offset][k_offset] * queries[b][h][current_offset][z]; 
                        }            
                    }
                }
                d_keys[b][h][y][z]=updt; 

            }

        }
    }
}


std::vector<torch::Tensor> qk_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    TORCH_CHECK((cuda_threads>0)&&(cuda_threads<=1024),"The value of CUDA_NUM_THREADS should between 1 and 1024");
    TORCH_CHECK(queries.size(0) == keys.size(0), "Query and Key should have same Batch_Size");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Query and Key should have same Head Nums");
    TORCH_CHECK(queries.size(2) == keys.size(2), "Query and Key should have same Pixel Nums");
    TORCH_CHECK(queries.size(3) == keys.size(3), "Query and Key should have same Head Dims");
    const int B= queries.size(0), N = queries.size(1), L = queries.size(2), C = queries.size(3);

    const int attention_span = kernel_size* kernel_size;
    const int DIMTHREADS = min(cuda_threads, C);
    const int PIXELTHREADS = min(int(cuda_threads / DIMTHREADS), L);
    const int BATCHTHREADS = max(1, cuda_threads / (PIXELTHREADS * DIMTHREADS));

    torch::Tensor d_queries = torch::empty({B, N, L, C}, queries.options());
    torch::Tensor d_keys = torch::empty({B, N, L, C}, keys.options());

    const dim3 threads(BATCHTHREADS, PIXELTHREADS, DIMTHREADS);
    const dim3 blocks(((B*N)+threads.x-1)/threads.x, (L+threads.y-1)/threads.y, (C+threads.z-1)/threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.scalar_type(), "qk_bw_cu",
    ([&] {
        qk_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),        
            height,
            width,
            kernel_size
        );
        qk_inverse_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),        
            height,
            width,
            kernel_size
        );
    }));

    return {d_queries,d_keys};
}
