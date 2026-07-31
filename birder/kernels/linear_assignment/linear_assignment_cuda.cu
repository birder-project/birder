/*
  Implementation is based on the algorithm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952

* Modified by:
* Ofer Hasson - 2026-03-27
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOptions.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/zeros.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int cooperative_block_size = 128;
constexpr int warp_size = 32;
constexpr int warps_per_block = cooperative_block_size / warp_size;
constexpr std::size_t shared_memory_reserve = 1024;

template <typename scalar_t>
struct min_candidate {
    scalar_t value;
    int remaining_index;
    int unmatched;
};

template <typename scalar_t>
__device__ __forceinline__ min_candidate<scalar_t> better_candidate(min_candidate<scalar_t> lhs,
                                                                    min_candidate<scalar_t> rhs) {
    if (rhs.remaining_index < 0) {
        return lhs;
    }
    if (lhs.remaining_index < 0) {
        return rhs;
    }
    if (rhs.value < lhs.value) {
        return rhs;
    }
    if (lhs.value < rhs.value) {
        return lhs;
    }
    if (rhs.unmatched != lhs.unmatched) {
        return rhs.unmatched != 0 ? rhs : lhs;
    }

    // Match the serial traversal exactly: the last unassigned column wins an
    // equal-cost tie, while the first assigned column remains selected.
    if (lhs.unmatched != 0) {
        return rhs.remaining_index > lhs.remaining_index ? rhs : lhs;
    }
    return rhs.remaining_index < lhs.remaining_index ? rhs : lhs;
}

template <typename scalar_t>
__device__ __forceinline__ min_candidate<scalar_t> warp_reduce_candidate(min_candidate<scalar_t> candidate) {
    constexpr unsigned int mask = 0xffffffffU;
    const int lane = threadIdx.x % warp_size;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        min_candidate<scalar_t> other{
            __shfl_down_sync(mask, candidate.value, offset),
            __shfl_down_sync(mask, candidate.remaining_index, offset),
            __shfl_down_sync(mask, candidate.unmatched, offset),
        };
        if (lane + offset < warp_size) {
            candidate = better_candidate(candidate, other);
        }
    }

    return candidate;
}

template <typename scalar_t>
__global__ void solve_cuda_kernel_batch_cooperative(int bs, int nr, int nc, const scalar_t *__restrict__ cost,
                                                    int64_t cost_batch_stride, int64_t cost_row_stride,
                                                    int64_t *__restrict__ col4row_out,
                                                    int64_t *__restrict__ row4col_out, scalar_t infinity) {
    const int batch_index = blockIdx.x;
    if (batch_index >= bs) {
        return;
    }

    extern __shared__ __align__(16) unsigned char cooperative_shared_memory[];
    auto *shared_scalars = reinterpret_cast<scalar_t *>(cooperative_shared_memory);
    scalar_t *u = shared_scalars;
    scalar_t *v = u + nr;
    scalar_t *shortest_path_costs = v + nc;

    auto *shared_ints = reinterpret_cast<int *>(shortest_path_costs + nc);
    int *path = shared_ints;
    int *col4row = path + nc;
    int *row4col = col4row + nr;
    int *remaining = row4col + nc;

    auto *shared_bytes_tail = reinterpret_cast<std::uint8_t *>(remaining + nc);
    std::uint8_t *visited_rows = shared_bytes_tail;
    std::uint8_t *visited_cols = visited_rows + nr;

    __shared__ scalar_t warp_values[warps_per_block];
    __shared__ int warp_indices[warps_per_block];
    __shared__ int warp_unmatched[warps_per_block];
    __shared__ scalar_t min_value;
    __shared__ int selected_remaining_index;
    __shared__ int current_row;
    __shared__ int sink;
    __shared__ int num_remaining;

    const int thread_index = threadIdx.x;
    const int lane = thread_index % warp_size;
    const int warp_index = thread_index / warp_size;
    const scalar_t *matrix_cost = cost + static_cast<int64_t>(batch_index) * cost_batch_stride;

    for (int index = thread_index; index < nr; index += cooperative_block_size) {
        u[index] = 0;
        col4row[index] = -1;
    }
    for (int index = thread_index; index < nc; index += cooperative_block_size) {
        v[index] = 0;
        row4col[index] = -1;
    }
    __syncthreads();

    for (int cur_row = 0; cur_row < nr; ++cur_row) {
        for (int index = thread_index; index < nr; index += cooperative_block_size) {
            visited_rows[index] = 0;
        }
        for (int index = thread_index; index < nc; index += cooperative_block_size) {
            visited_cols[index] = 0;
            remaining[index] = nc - index - 1;
            shortest_path_costs[index] = infinity;
        }
        if (thread_index == 0) {
            min_value = 0;
            current_row = cur_row;
            sink = -1;
            num_remaining = nc;
        }
        __syncthreads();

        while (sink == -1) {
            if (thread_index == 0) {
                visited_rows[current_row] = 1;
            }
            __syncthreads();

            const int row = current_row;
            const scalar_t base_r = min_value - u[row];
            const scalar_t *cost_row = matrix_cost + static_cast<int64_t>(row) * cost_row_stride;
            min_candidate<scalar_t> thread_candidate{infinity, -1, 0};
            for (int remaining_index = thread_index; remaining_index < num_remaining;
                 remaining_index += cooperative_block_size) {
                const int column = remaining[remaining_index];
                const scalar_t reduced_cost = base_r + cost_row[column] - v[column];
                scalar_t shortest_cost = shortest_path_costs[column];
                if (reduced_cost < shortest_cost) {
                    path[column] = row;
                    shortest_cost = reduced_cost;
                    shortest_path_costs[column] = shortest_cost;
                }

                const min_candidate<scalar_t> candidate{
                    shortest_cost,
                    remaining_index,
                    row4col[column] == -1 ? 1 : 0,
                };
                thread_candidate = better_candidate(thread_candidate, candidate);
            }

            thread_candidate = warp_reduce_candidate(thread_candidate);
            if (lane == 0) {
                warp_values[warp_index] = thread_candidate.value;
                warp_indices[warp_index] = thread_candidate.remaining_index;
                warp_unmatched[warp_index] = thread_candidate.unmatched;
            }
            __syncthreads();

            if (warp_index == 0) {
                min_candidate<scalar_t> block_candidate{infinity, -1, 0};
                if (lane < warps_per_block) {
                    block_candidate = {
                        warp_values[lane],
                        warp_indices[lane],
                        warp_unmatched[lane],
                    };
                }
                block_candidate = warp_reduce_candidate(block_candidate);
                if (lane == 0) {
                    min_value = block_candidate.value;
                    selected_remaining_index = block_candidate.remaining_index;
                }
            }
            __syncthreads();

            if (min_value == infinity) {
                if (thread_index == 0) {
                    CUDA_KERNEL_ASSERT(false && "Infeasible matrix");
                }
                return;
            }

            if (thread_index == 0) {
                const int column = remaining[selected_remaining_index];
                if (row4col[column] == -1) {
                    sink = column;
                } else {
                    current_row = row4col[column];
                }

                visited_cols[column] = 1;
                --num_remaining;
                remaining[selected_remaining_index] = remaining[num_remaining];
            }
            __syncthreads();
        }

        if (thread_index == 0) {
            u[cur_row] += min_value;
        }
        for (int row = thread_index; row < nr; row += cooperative_block_size) {
            if (visited_rows[row] != 0 && row != cur_row) {
                u[row] += min_value - shortest_path_costs[col4row[row]];
            }
        }
        for (int column = thread_index; column < nc; column += cooperative_block_size) {
            if (visited_cols[column] != 0) {
                v[column] -= min_value - shortest_path_costs[column];
            }
        }
        __syncthreads();

        if (thread_index == 0) {
            int row = -1;
            int column = sink;
            while (row != cur_row) {
                row = path[column];
                row4col[column] = row;
                const int previous_column = col4row[row];
                col4row[row] = column;
                column = previous_column;
            }
        }
        __syncthreads();
    }

    int64_t *matrix_col4row = col4row_out + static_cast<int64_t>(batch_index) * nr;
    int64_t *matrix_row4col = row4col_out + static_cast<int64_t>(batch_index) * nc;
    for (int row = thread_index; row < nr; row += cooperative_block_size) {
        matrix_col4row[row] = col4row[row];
    }
    for (int column = thread_index; column < nc; column += cooperative_block_size) {
        matrix_row4col[column] = row4col[column];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void array_fill(scalar_t *start, scalar_t *stop, scalar_t value) {
    for (; start < stop; ++start) {
        *start = value;
    }
}

template <typename scalar_t>
__device__ __forceinline__ int
augmenting_path_cuda_serial(int nr, int nc, int row, const scalar_t *__restrict__ cost, int64_t cost_row_stride,
                            scalar_t *__restrict__ u, scalar_t *__restrict__ v, int *__restrict__ path,
                            int64_t *__restrict__ row4col, scalar_t *__restrict__ shortest_path_costs,
                            std::uint8_t *__restrict__ visited_rows, std::uint8_t *__restrict__ visited_cols,
                            int *__restrict__ remaining, scalar_t *min_value_out, scalar_t infinity) {
    scalar_t min_value = 0;
    int num_remaining = nc;
    for (int index = 0; index < nc; ++index) {
        visited_cols[index] = 0;
        remaining[index] = nc - index - 1;
        shortest_path_costs[index] = infinity;
    }
    array_fill(visited_rows, visited_rows + nr, static_cast<std::uint8_t>(0));

    int sink = -1;
    while (sink == -1) {
        int selected_remaining_index = -1;
        scalar_t lowest = infinity;
        visited_rows[row] = 1;

        const scalar_t *cost_row = cost + static_cast<int64_t>(row) * cost_row_stride;
        const scalar_t base_r = min_value - u[row];
        for (int remaining_index = 0; remaining_index < num_remaining; ++remaining_index) {
            const int column = remaining[remaining_index];
            const scalar_t reduced_cost = base_r + cost_row[column] - v[column];
            if (reduced_cost < shortest_path_costs[column]) {
                path[column] = row;
                shortest_path_costs[column] = reduced_cost;
            }
            if (shortest_path_costs[column] < lowest ||
                (shortest_path_costs[column] == lowest && row4col[column] == -1)) {
                lowest = shortest_path_costs[column];
                selected_remaining_index = remaining_index;
            }
        }

        min_value = lowest;
        if (min_value == infinity) {
            return -1;
        }

        const int column = remaining[selected_remaining_index];
        if (row4col[column] == -1) {
            sink = column;
        } else {
            row = static_cast<int>(row4col[column]);
        }

        visited_cols[column] = 1;
        remaining[selected_remaining_index] = remaining[--num_remaining];
    }

    *min_value_out = min_value;
    return sink;
}

template <typename scalar_t>
__device__ __forceinline__ void
solve_cuda_serial(int nr, int nc, const scalar_t *__restrict__ cost, int64_t cost_row_stride, scalar_t *__restrict__ u,
                  scalar_t *__restrict__ v, scalar_t *__restrict__ shortest_path_costs, int *__restrict__ path,
                  int64_t *__restrict__ col4row, int64_t *__restrict__ row4col, std::uint8_t *__restrict__ visited_rows,
                  std::uint8_t *__restrict__ visited_cols, int *__restrict__ remaining, scalar_t infinity) {
    scalar_t min_value;
    for (int cur_row = 0; cur_row < nr; ++cur_row) {
        const int sink = augmenting_path_cuda_serial(nr, nc, cur_row, cost, cost_row_stride, u, v, path, row4col,
                                                     shortest_path_costs, visited_rows, visited_cols, remaining,
                                                     &min_value, infinity);
        CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");

        u[cur_row] += min_value;
        for (int row = 0; row < nr; ++row) {
            if (visited_rows[row] != 0 && row != cur_row) {
                u[row] += min_value - shortest_path_costs[col4row[row]];
            }
        }
        for (int column = 0; column < nc; ++column) {
            if (visited_cols[column] != 0) {
                v[column] -= min_value - shortest_path_costs[column];
            }
        }

        int row = -1;
        int column = sink;
        while (row != cur_row) {
            row = path[column];
            row4col[column] = row;
            const int previous_column = static_cast<int>(col4row[row]);
            col4row[row] = column;
            column = previous_column;
        }
    }
}

template <typename scalar_t>
__global__ void
solve_cuda_kernel_batch_serial(int bs, int nr, int nc, const scalar_t *__restrict__ cost, int64_t cost_batch_stride,
                               int64_t cost_row_stride, scalar_t *__restrict__ u, scalar_t *__restrict__ v,
                               scalar_t *__restrict__ shortest_path_costs, int *__restrict__ path,
                               int64_t *__restrict__ col4row, int64_t *__restrict__ row4col,
                               std::uint8_t *__restrict__ visited_rows, std::uint8_t *__restrict__ visited_cols,
                               int *__restrict__ remaining, scalar_t infinity) {
    const int batch_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_index >= bs) {
        return;
    }

    solve_cuda_serial(
        nr, nc, cost + static_cast<int64_t>(batch_index) * cost_batch_stride, cost_row_stride,
        u + static_cast<int64_t>(batch_index) * nr, v + static_cast<int64_t>(batch_index) * nc,
        shortest_path_costs + static_cast<int64_t>(batch_index) * nc, path + static_cast<int64_t>(batch_index) * nc,
        col4row + static_cast<int64_t>(batch_index) * nr, row4col + static_cast<int64_t>(batch_index) * nc,
        visited_rows + static_cast<int64_t>(batch_index) * nr, visited_cols + static_cast<int64_t>(batch_index) * nc,
        remaining + static_cast<int64_t>(batch_index) * nc, infinity);
}

template <typename scalar_t>
void solve_cuda_batch_serial(const at::TensorOptions &cost_options, int bs, int nr, int nc, const scalar_t *cost,
                             int64_t cost_batch_stride, int64_t cost_row_stride, int64_t *col4row, int64_t *row4col) {
    const auto infinity = std::numeric_limits<scalar_t>::infinity();
    const auto int_options = cost_options.dtype(c10::ScalarType::Int);
    const auto uint8_options = cost_options.dtype(c10::ScalarType::Byte);

    const int64_t batch_rows = static_cast<int64_t>(bs) * nr;
    const int64_t batch_columns = static_cast<int64_t>(bs) * nc;
    at::Tensor u = at::zeros({batch_rows}, cost_options);
    at::Tensor v = at::zeros({batch_columns}, cost_options);
    at::Tensor shortest_path_costs = at::empty({batch_columns}, cost_options);
    at::Tensor path = at::empty({batch_columns}, int_options);
    at::Tensor visited_rows = at::empty({batch_rows}, uint8_options);
    at::Tensor visited_cols = at::empty({batch_columns}, uint8_options);
    at::Tensor remaining = at::empty({batch_columns}, int_options);

    constexpr int block_size = 32;
    const int grid_size = (bs + block_size - 1) / block_size;
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    solve_cuda_kernel_batch_serial<<<grid_size, block_size, 0, stream.stream()>>>(
        bs, nr, nc, cost, cost_batch_stride, cost_row_stride, u.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
        shortest_path_costs.data_ptr<scalar_t>(), path.data_ptr<int>(), col4row, row4col,
        visited_rows.data_ptr<std::uint8_t>(), visited_cols.data_ptr<std::uint8_t>(), remaining.data_ptr<int>(),
        infinity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void solve_cuda_batch_cooperative(int bs, int nr, int nc, const scalar_t *cost, int64_t cost_batch_stride,
                                  int64_t cost_row_stride, int64_t *col4row, int64_t *row4col,
                                  std::size_t shared_memory_size) {
    const auto infinity = std::numeric_limits<scalar_t>::infinity();
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    solve_cuda_kernel_batch_cooperative<<<bs, cooperative_block_size, shared_memory_size, stream.stream()>>>(
        bs, nr, nc, cost, cost_batch_stride, cost_row_stride, col4row, row4col, infinity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::size_t cooperative_shared_memory_size(const at::Tensor &cost, int nr, int nc) {
    const std::size_t scalar_count = static_cast<std::size_t>(nr) + 2 * static_cast<std::size_t>(nc);
    const std::size_t int_count = static_cast<std::size_t>(nr) + 3 * static_cast<std::size_t>(nc);
    const std::size_t byte_count = static_cast<std::size_t>(nr) + static_cast<std::size_t>(nc);
    return scalar_count * cost.element_size() + int_count * sizeof(int) + byte_count * sizeof(std::uint8_t);
}

std::vector<at::Tensor> batch_linear_assignment_cuda_impl(const at::Tensor &cost) {
    const auto sizes = cost.sizes();
    const auto strides = cost.strides();
    const int batch_size = static_cast<int>(sizes[0]);
    const int nr = static_cast<int>(sizes[1]);
    const int nc = static_cast<int>(sizes[2]);
    const int64_t cost_batch_stride = strides[0];
    const int64_t cost_row_stride = strides[1];
    const auto index_options = cost.options().dtype(c10::ScalarType::Long);

    if (batch_size == 0 || nr == 0) {
        return {
            at::full({batch_size, nr}, -1, index_options),
            at::full({batch_size, nc}, -1, index_options),
        };
    }

    const std::size_t shared_memory_size = cooperative_shared_memory_size(cost, nr, nc);
    const auto *device_properties = at::cuda::getCurrentDeviceProperties();
    const bool use_cooperative =
        shared_memory_size + shared_memory_reserve <= static_cast<std::size_t>(device_properties->sharedMemPerBlock);

    at::Tensor col4row;
    at::Tensor row4col;
    if (use_cooperative) {
        col4row = at::empty({batch_size, nr}, index_options);
        row4col = at::empty({batch_size, nc}, index_options);
    } else {
        col4row = at::full({batch_size, nr}, -1, index_options);
        row4col = at::full({batch_size, nc}, -1, index_options);
    }

    AT_DISPATCH_FLOATING_TYPES(cost.scalar_type(), "solve_cuda_batch", [&] {
        if (use_cooperative) {
            solve_cuda_batch_cooperative<scalar_t>(batch_size, nr, nc, cost.data_ptr<scalar_t>(), cost_batch_stride,
                                                   cost_row_stride, col4row.data_ptr<int64_t>(),
                                                   row4col.data_ptr<int64_t>(), shared_memory_size);
        } else {
            solve_cuda_batch_serial<scalar_t>(cost.options(), batch_size, nr, nc, cost.data_ptr<scalar_t>(),
                                              cost_batch_stride, cost_row_stride, col4row.data_ptr<int64_t>(),
                                              row4col.data_ptr<int64_t>());
        }
    });

    return {col4row, row4col};
}

} // namespace

std::vector<at::Tensor> batch_linear_assignment_cuda(const at::Tensor &cost) {
    at::DeviceGuard guard(cost.device());
    const auto sizes = cost.sizes();
    TORCH_CHECK(sizes.size() == 3, "Cost matrix must have shape (B, W, T).");
    TORCH_CHECK(sizes[0] <= std::numeric_limits<int>::max() && sizes[1] <= std::numeric_limits<int>::max() &&
                    sizes[2] <= std::numeric_limits<int>::max(),
                "Cost matrix dimensions exceed the CUDA kernel's index range.");

    if (sizes[1] <= sizes[2]) {
        return batch_linear_assignment_cuda_impl(cost.contiguous());
    }

    // The shortest augmenting path implementation requires rows <= columns.
    // Materializing this transpose also makes the repeated 300-column scans
    // contiguous and therefore coalesced in the cooperative kernel.
    const at::Tensor transposed_cost = cost.transpose(1, 2).contiguous();
    auto assignment = batch_linear_assignment_cuda_impl(transposed_cost);
    return {assignment[1], assignment[0]};
}
