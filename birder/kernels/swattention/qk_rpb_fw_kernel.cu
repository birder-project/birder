/*!
**************************************************************************************************
* TransNeXt
* Taken from:
* https://github.com/DaiShiResearch/TransNeXt/blob/main/swattention_extension
* Licensed under the Apache License, Version 2.0
*
* Modified by:
* Ofer Hasson - 2026-07-20
**************************************************************************************************
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

template <typename scalar_t>
__global__ void qk_rpb_fw_kernel(const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> queries,
                                 const at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> keys,
                                 const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> rpb,
                                 at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> attn_weight, int height,
                                 int width, int kernel_size) {
    using acc_t = at::opmath_type<scalar_t>;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (queries.size(0) * queries.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < queries.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < kernel_size * kernel_size) {
                const int b = x / queries.size(1);
                const int h = x - b * queries.size(1);
                const int ki = z / kernel_size;
                const int kj = z - ki * kernel_size;
                const int i = y / width;
                const int j = y - i * width;
                const int ni = i + ki - (kernel_size - 1) / 2;
                const int nj = j + kj - (kernel_size - 1) / 2;

                acc_t updt = acc_t(0);
                if (((ni >= 0) && (ni < height)) && ((nj >= 0) && (nj < width))) {
                    const int key_y = ni * width + nj;
#pragma unroll
                    for (int dim_offset = 0; dim_offset < queries.size(3); ++dim_offset)
                        updt += static_cast<acc_t>(queries[b][h][y][dim_offset]) *
                                static_cast<acc_t>(keys[b][h][key_y][dim_offset]);
                    updt += static_cast<acc_t>(rpb[h][z]);
                } else {
                    updt = acc_t(-INFINITY);
                }
                attn_weight[b][h][y][z] = static_cast<scalar_t>(updt);
            }
        }
    }
}

at::Tensor qk_rpb_fw_cu(const at::Tensor &queries, const at::Tensor &keys, const at::Tensor &rpb, int height, int width,
                        int kernel_size, int cuda_threads) {
    at::DeviceGuard guard(queries.device());

    TORCH_CHECK(queries.size(0) == keys.size(0), "Query and Key should have same Batch_Size");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Query and Key should have same Head Nums");
    TORCH_CHECK(queries.size(2) == keys.size(2), "Query and Key should have same Pixel Nums");
    TORCH_CHECK(queries.size(3) == keys.size(3), "Query and Key should have same Head Dims");
    TORCH_CHECK(rpb.size(0) == keys.size(1), "Relative_Position_Bias should have same Head Dims with Query and Key");

    const int batch = queries.size(0), num_heads = queries.size(1), num_tokens = queries.size(2);

    const int attention_span = kernel_size * kernel_size;
    TORCH_CHECK(rpb.size(1) == attention_span, "Last dim of Relative_Position_Bias should equal Kernel_Size^2");
    const int kernel_threads = std::min(cuda_threads, attention_span);
    const int pixel_threads = std::min(int(cuda_threads / kernel_threads), num_tokens);
    const int batch_threads = std::max(1, cuda_threads / (pixel_threads * kernel_threads));

    at::Tensor attn_weight = at::empty({batch, num_heads, num_tokens, attention_span}, queries.options());

    const dim3 threads(batch_threads, pixel_threads, kernel_threads);
    const dim3 blocks(((batch * num_heads) + threads.x - 1) / threads.x, (num_tokens + threads.y - 1) / threads.y,
                      (attention_span + threads.z - 1) / threads.z);
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, queries.scalar_type(), "qk_rpb_fw_cu", ([&] {
            qk_rpb_fw_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
                queries.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                keys.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
                rpb.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                attn_weight.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(), height, width, kernel_size);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));

    return attn_weight;
}
